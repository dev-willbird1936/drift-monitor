// Sandbox guard for dataset-generation sessions.
// Blocks file tools outside the session folder and bash commands that reach outside it,
// install packages, or use the network. Heuristic, not a security boundary.
import * as path from "node:path";

const OUTSIDE = [
  /(^|[\s"'=(;|&<>])[A-Za-z]:[\\/]/, // drive-letter absolute path
  /(^|[\s"'=(;|&<>])\/(?!dev\/null|tmp\/drift-)[A-Za-z]/, // unix absolute path
  /(^|[\s"'=(;|&<>\/\\])\.\.([\\/]|$|\s)/, // parent traversal
  /(^|[\s"'=(])~/, /\$HOME|\$\{HOME\}|%USERPROFILE%|%APPDATA%|\$env:/i,
];
const DENY = [
  /\b(pip3?|uv|npm|pnpm|yarn|npx|choco|winget|apt|conda)\b[^|;&]*\b(install|add|i)\b/i,
  /\b(curl|wget|ssh|scp|git\s+(push|clone|pull|fetch)|Invoke-WebRequest|iwr)\b/i,
  /\b(shutdown|format|reg\s+(add|delete)|taskkill|schtasks|powershell|pwsh|cmd(\.exe)?\s+\/c)\b/i,
];

export default function (pi: any) {
  pi.on("tool_call", async (event: any, ctx: any) => {
    const root = path.resolve(ctx.cwd ?? process.cwd());
    const inside = (p: string) => {
      const r = path.resolve(root, p);
      return r === root || r.toLowerCase().startsWith(root.toLowerCase() + path.sep);
    };
    const input = event.input ?? {};
    if (event.toolName === "bash") {
      const cmd = String(input.command ?? "");
      const stripped = cmd.split(root).join(".").split(root.replace(/\\/g, "/")).join(".");
      if (OUTSIDE.some((r) => r.test(stripped)))
        return { block: true, reason: "Blocked: commands may only touch files inside the current project folder (use relative paths)." };
      if (DENY.some((r) => r.test(cmd)))
        return { block: true, reason: "Blocked: package installs, network access and system commands are not available in this environment." };
      return;
    }
    for (const k of ["path", "file_path", "filePath"]) {
      if (typeof input[k] === "string" && !inside(input[k]))
        return { block: true, reason: "Blocked: path is outside the current project folder." };
    }
  });
}
