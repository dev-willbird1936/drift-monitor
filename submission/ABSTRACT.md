<!-- Paste only the text below the line into the abstract field. The title, track, problem statement and
     repo list are separate fields on the submission form.

     Title:    Drift monitor: a 3.35B model that decides when to call a bigger one
     Track:    1, Fine-tuning and Data Curation
     Problem:  2, replacing a closed-model API
     Pipeline: CuratorKIT -> AlignTune -> AuditKIT
-->

---

When you delegate a task to an AI agent you authorise specific work. A drifting agent keeps producing
plausible output, so oversight is lost without anyone noticing, and reviewing every step with a frontier
model costs more than the agent itself. We fine-tuned Cohere Labs' `tiny-aya-global` (3.35B) to read a
session and emit one digit, 0 to 9: how far the work has drifted from what the user asked for. A threshold on
it decides when to call a stronger model.

**Does acting on it help?** We closed the loop on a live agent: 256 sessions on 16 tasks, six written
after training and never seen by the monitor. Drift is induced by a hidden prompt; an alert calls an advisor model. Outcomes are objective: files on disk and each task's own check. Unrequested
files left behind fell from **2.48 per session to 0** (95% CI of the difference −2.99 to −1.99); checks
passing rose from 84% to 99%. On 32 clean sessions it alerted twice; the advisor dismissed both.
Having the advisor review every turn gave equivalent outcomes at 4.8 advisor calls per drifting session
instead of 1.4, and 2.8 per clean session instead of 0.06. Replayed over the
unwatched arm, it flags 66 of the 67 sessions that ended badly, including all 26 on unseen tasks.

**Against the closed models it replaces.** Two frontier models, the identical prompt, 497 identical real
held-out rows. At matched recall the 3.35B model raises **5–6× fewer false alarms**; at matched false-alarm rate it catches **1.6–1.8× more drift**.

**Before and after, same real sessions.** MAE 43.9 → 7.8, AUROC 0.552 → 0.957, correct alert decisions
0.164 → 0.922, false alarms 83.6% → 5.8%. On four held-out work categories: MAE 5.99, AUROC 0.981.

**Data.** A non-Cohere teacher model wrote 5,417 sessions across 20 work
categories and 16 drift patterns, including planted instructions obeyed and refused. Each pattern is paired
with an on-course control that looks nearly identical and differs only in what the user asked for, so the
model must read the authorisation. Every session is
rendered at six evidence tiers. With 1,902 recorded live sessions: 90,365 rows. Two independent judges re-scored labels blind and disagree with each other
more than with our labels. CuratorKIT curated the recorded corpus and audited the
final training split (3.1% flagged, disclosed), AlignTune trained a 4-bit QLoRA adapter, AuditKIT graded
every run.

**What did not work.** Synthetic-only training over-fired on real traffic by 15 points until real sessions
were added. The loop's one failure: the agent broke a word limit inside the requested file, which the
monitor's clipped view cannot see. The induced drift was mild and the agent obeyed every note; an agent that
resists correction is untested.

**Cost.** 309 ms per score on one consumer GPU. Monitor plus triggered advisor calls: $463 per million
steps at threshold 30 ($316 at 40) against $1,735 to review every step.

**Honesty note.** 5.71% of real held-out rows shared a prompt with training; those figures are leak-free.
