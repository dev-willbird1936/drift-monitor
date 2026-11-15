// drift-watch: run the drift monitor live next to an ordinary Pi session.
//
// After every agent turn (one assistant message plus its tool results) the session so far is sent to the local
// scoring server (harness/score_server.py). The score shows in the footer. When it crosses the threshold and an
// advisor model is configured, the advisor is asked for a note, which is passed to the agent before its next step.
// Scoring runs in the background: the agent never waits for the monitor.
//
//   pi -e harness/pi_drift_watch.ts
//
// Environment (all optional):
//   DRIFT_SCORE_URL         scoring server (default http://127.0.0.1:8765)
//   DRIFT_THRESHOLD         alert threshold, 0-100 (default 30; 40-50 for weak agents)
//   DRIFT_ADVISOR           advisor as provider/model, e.g. claude-bridge-1/claude-haiku-5-5. Empty: watch only.
//   DRIFT_ADVISOR_THINKING  advisor reasoning level (default high)
//   DRIFT_MAX_ADVICE        advisor calls per session (default 2)
//   DRIFT_LOG               append every score and advisor reply to this JSONL file
//
// Command: /drift (show the trajectory)  /drift on|off  /drift threshold N  /drift advisor provider/model|none
import * as fs from "node:fs";

const URL_ = (process.env.DRIFT_SCORE_URL ?? "http://127.0.0.1:8765").replace(/\/$/, "");
const NOTE_TAG = "drift-advisor";

type Step = { thinking: string; say: string; calls: { tool: string; args: any }[]; results: { text: string; is_error: boolean }[] };

function textOf(content: any): string {
  if (typeof content === "string") return content;
  return (content ?? []).filter((b: any) => b?.type === "text").map((b: any) => b.text ?? "").join("\n");
}

// The observer's view of the session: the user's first message is the goal, later user messages are updates,
// each assistant message plus its tool results is one step. Extension messages (advisor notes) are not included.
function sessionFromBranch(branch: any[]) {
  let goal = "";
  const updates: { after_step: number; text: string }[] = [];
  const steps: Step[] = [];
  const byCall = new Map<string, Step>();
  for (const e of branch) {
    if (e?.type !== "message") continue;
    const m = e.message ?? {};
    if (m.role === "user") {
      const t = textOf(m.content).trim();
      if (!t) continue;
      if (!goal) goal = t;
      else updates.push({ after_step: steps.length, text: t });
    } else if (m.role === "assistant") {
      if (m.stopReason === "error" || m.stopReason === "aborted") continue;
      const s: Step = { thinking: "", say: "", calls: [], results: [] };
      for (const b of m.content ?? []) {
        if (b.type === "thinking" && b.thinking) s.thinking += (s.thinking ? " " : "") + b.thinking;
        else if (b.type === "text" && b.text?.trim()) s.say += (s.say ? " " : "") + b.text;
        else if (b.type === "toolCall") {
          s.calls.push({ tool: b.name, args: b.arguments ?? {} });
          byCall.set(b.id, s);
        }
      }
      steps.push(s);
    } else if (m.role === "toolResult") {
      const s = byCall.get(m.toolCallId) ?? steps[steps.length - 1];
      if (s) s.results.push({ text: textOf(m.content), is_error: !!m.isError });
    }
  }
  return { goal, constraints: [] as string[], updates, steps, tier: 5 };
}

async function post(path: string, body: any, signal?: AbortSignal) {
  const r = await fetch(URL_ + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  const j: any = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}

export default function (pi: any) {
  let enabled = true;
  let threshold = Number(process.env.DRIFT_THRESHOLD ?? 30);
  let advisor = process.env.DRIFT_ADVISOR ?? "";
  const advisorThinking = process.env.DRIFT_ADVISOR_THINKING ?? "high";
  const maxAdvice = Number(process.env.DRIFT_MAX_ADVICE ?? 2);
  const logPath = process.env.DRIFT_LOG ?? "";
  let history: any[] = [];
  let nAdvice = 0, lastAdviceStep = -99;
  let busy = false;
  let pending: { snap: any; ctx: any } | null = null;
  let settling = false;          // the agent has stopped and is waiting for our last score before it settles
  const undelivered: string[] = []; // advisor notes to hand over when the agent settles

  const log = (rec: any) => {
    if (logPath) fs.appendFileSync(logPath, JSON.stringify({ t: Date.now(), ...rec }) + "\n");
  };
  const status = (ctx: any, s: string) => {
    try {
      if (ctx.hasUI) ctx.ui.setStatus("drift", s);
    } catch { /* the session ended or was replaced */ }
  };

  async function askAdvisor(snap: any, score: number, ctx: any): Promise<string | null> {
    const slash = advisor.indexOf("/");
    const model = slash > 0 ? ctx.modelRegistry.find(advisor.slice(0, slash), advisor.slice(slash + 1)) : undefined;
    if (!model) throw new Error(`advisor model ${advisor} not found`);
    const p = await post("/advisor_prompt", { session: snap, score });
    const stream = ctx.modelRegistry.streamSimple(model, {
      systemPrompt: p.system,
      messages: [{ role: "user", content: [{ type: "text", text: p.prompt }], timestamp: Date.now() }],
    }, { reasoning: advisorThinking });
    const msg: any = await stream.result();
    const reply = textOf(msg?.content).trim();
    log({ step: snap.steps.length, advisor, advisor_reply: reply.slice(0, 1500) });
    if (!reply || reply.slice(0, 40).includes("NO_ACTION")) return null;
    return p.note_prefix + reply.slice(0, 1200);
  }

  async function process_(snap: any, ctx: any) {
    const k = snap.steps.length;
    const r = await post("/score", { session: snap });
    const alert = r.score >= threshold;
    history.push({ step: k, score: r.score, spread: r.spread, alert });
    log({ step: k, score: r.score, spread: r.spread, alert, ms: r.ms });
    status(ctx, `drift ${Math.round(r.score)}${alert ? " ⚠" : ""}`);
    if (!(alert && advisor && nAdvice < maxAdvice && k - lastAdviceStep >= 2)) return;
    nAdvice++;
    lastAdviceStep = k;
    status(ctx, `drift ${Math.round(r.score)} ⚠ asking advisor`);
    const note = await askAdvisor(snap, r.score, ctx);
    history[history.length - 1].advice = note ? "note" : "NO_ACTION";
    status(ctx, `drift ${Math.round(r.score)}${note ? " · advisor note sent" : " · advisor: on course"}`);
    if (!note) return;
    let running = false;
    try { running = !ctx.isIdle(); } catch { /* stale ctx */ }
    if (running && !settling) {
      // mid-run: delivered after the current tool calls, before the agent's next model call
      pi.sendMessage({ customType: NOTE_TAG, content: note, display: true }, { deliverAs: "steer" });
    } else if (settling) {
      undelivered.push(note); // agent_before_settle hands it over and asks for one more turn
    } else {
      pi.sendMessage({ customType: NOTE_TAG, content: note, display: true }, { deliverAs: "followUp", triggerTurn: true });
    }
  }

  async function drain() {
    busy = true;
    while (pending) {
      const { snap, ctx } = pending;
      pending = null;
      try {
        await process_(snap, ctx);
      } catch (e: any) {
        status(ctx, `drift: ${String(e?.message ?? e).slice(0, 60)}`);
        log({ error: String(e?.message ?? e).slice(0, 300) });
      }
    }
    busy = false;
  }

  // The agent is about to stop. Wait for the score of its last step (and the advisor, if that fires) so a final
  // drifting step is not missed, then hand over any note and ask for one more turn.
  pi.on("agent_before_settle", async (event: any) => {
    if (!enabled) return;
    settling = true;
    const t0 = Date.now();
    try {
      while ((busy || pending) && Date.now() - t0 < 90_000) await new Promise((r) => setTimeout(r, 150));
    } finally {
      settling = false;
    }
    if (!undelivered.length) return;
    const notes = undelivered.splice(0);
    return {
      entries: [...event.entries, ...notes.map((n) => ({ type: "custom_message", customType: NOTE_TAG, content: n, display: true }))],
      continue: true,
    };
  });

  pi.on("session_start", async (_e: any, ctx: any) => {
    history = []; nAdvice = 0; lastAdviceStep = -99;
    status(ctx, enabled ? `drift watch on${advisor ? "" : " (no advisor)"}` : "drift watch off");
  });

  pi.on("turn_end", async (_event: any, ctx: any) => {
    if (!enabled) return;
    const snap = sessionFromBranch(ctx.sessionManager.getBranch());
    if (!snap.goal || !snap.steps.length) return;
    pending = { snap, ctx }; // newest snapshot wins: it already contains every earlier step
    if (!busy) void drain();
  });

  pi.registerCommand("drift", {
    description: "Drift monitor: show scores, or on|off|threshold N|advisor provider/model|none",
    handler: async (args: string, ctx: any) => {
      const [cmd, val] = (args ?? "").trim().split(/\s+/);
      if (cmd === "on" || cmd === "off") enabled = cmd === "on";
      else if (cmd === "threshold" && val) threshold = Number(val);
      else if (cmd === "advisor" && val) advisor = val === "none" ? "" : val;
      const lines = history.map((h) => `step ${String(h.step).padStart(2)}  ${h.score.toFixed(1).padStart(5)}  ${"#".repeat(Math.round(h.score / 4))}${h.alert ? "  ALERT" : ""}${h.advice ? `  advisor: ${h.advice}` : ""}`);
      const head = `drift watch ${enabled ? "on" : "off"} · threshold ${threshold} · advisor ${advisor || "none"} · ${nAdvice}/${maxAdvice} advisor calls · server ${URL_}`;
      if (ctx.hasUI) ctx.ui.notify([head, ...lines].join("\n"), "info");
    },
  });
}
