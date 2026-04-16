#!/usr/bin/env python3
"""MCP tools subsystem 🧿

Implements:
 - MCP server config store (LMStudio-style mcp_servers.json).
 - Stdio MCP client (JSON-RPC 2.0) with initialize, tools/list, tools/call.
 - Tool-tab session storage (separate from main chat).
 - Tool-model orchestration loop: the tool model receives a task + available
   tool schemas, emits either a tool call or a final answer, repeats until
   done, then a one-sentence summary is produced and handed back to the
   main chat as a system message. Thinking segments are preserved so the
   tool tab can render them.

The standard chat model never sees raw tool I/O — only the short summary.
The standard chat model can be asked by the tool model (via an [ASK_CHAT]
block) to produce persona-flavoured writing (social posts, emails, etc.)
that the tool model then places into its next tool call.
"""
import json, os, re, threading, subprocess, time, uuid, glob, queue
from datetime import datetime

try:
    import requests as R
except ImportError:
    R = None

# ── Paths ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MCP_SERVERS_P = os.path.join(DATA, "mcp_servers.json")
TOOLS_CFG_P = os.path.join(DATA, "tools_config.json")
TOOL_SESSIONS = os.path.join(DATA, "tool_sessions")
TOOL_ACTIVE_P = os.path.join(DATA, "tool_active_id.txt")
PENDING_SUMMARIES = os.path.join(DATA, "tool_pending_summaries.json")
os.makedirs(TOOL_SESSIONS, exist_ok=True)

# ── Defaults ──────────────────────────────────────────────────────
DEFAULT_SERVERS = {
    "mcpServers": {
        "docker": {
            "command": "docker",
            "args": ["run", "-i", "--rm", "alpine/socat", "STDIO", "TCP:host.docker.internal:8811"],
            "env": {},
            "enabled": False,
            "_note": "Example Docker MCP gateway. Edit command/args to match your setup."
        }
    }
}
DEFAULT_TOOLS_CFG = {
    "engine": "ollama",
    "ollama_host": "http://localhost:11434",
    "model": "qwen3:4b",
    "system_prompt": (
        "You are an MCP tool-executing agent. You work in a sandboxed scratchpad "
        "separate from the main conversation. Given a task, decide whether to call "
        "a tool or to emit a final answer.\n\n"
        "To call a tool, reply with ONLY a fenced json block shaped like:\n"
        "```tool\n{\"name\": \"server.tool\", \"arguments\": {...}}\n```\n\n"
        "When you have enough information, reply with a final answer prefixed by "
        "FINAL: and a one-sentence factual summary on the next line prefixed by "
        "SUMMARY: . The SUMMARY line is what the main chat model will see.\n\n"
        "If a task requires persona-coloured writing (social posts, emails, "
        "messages), request it from the main chat model by emitting:\n"
        "```ask_chat\n{\"request\": \"describe what to write\"}\n```\n"
        "You'll receive the reply as a system message and can then use it in your "
        "next tool call.\n\n"
        "You may think before each tool call inside <think>…</think> tags — the "
        "think block is shown in the Tools tab but is NOT sent to tools."
    ),
    "max_iters": 8,
    "temperature": 0.3,
    "think_enabled": True,
}

# ── Locks and version ─────────────────────────────────────────────
_lock = threading.Lock()
_version = 0
_clients = {}          # name -> MCPStdioClient
_clients_lock = threading.Lock()

def _bump():
    global _version
    with _lock: _version += 1
    return _version

def version(): return _version

# ── Config I/O ────────────────────────────────────────────────────
def load_servers():
    if not os.path.exists(MCP_SERVERS_P):
        with open(MCP_SERVERS_P, "w") as f: json.dump(DEFAULT_SERVERS, f, indent=2)
        return dict(DEFAULT_SERVERS)
    try:
        with open(MCP_SERVERS_P) as f: d = json.load(f)
        if "mcpServers" not in d: d["mcpServers"] = {}
        return d
    except Exception: return dict(DEFAULT_SERVERS)

def save_servers(d):
    with open(MCP_SERVERS_P, "w") as f: json.dump(d, f, indent=2)
    _bump()

def load_tools_cfg():
    d = dict(DEFAULT_TOOLS_CFG)
    if os.path.exists(TOOLS_CFG_P):
        try:
            with open(TOOLS_CFG_P) as f: d.update(json.load(f))
        except Exception: pass
    return d

def save_tools_cfg(d):
    with open(TOOLS_CFG_P, "w") as f: json.dump(d, f, indent=2)
    _bump()

# ── MCP stdio client ──────────────────────────────────────────────
class MCPStdioClient:
    """Minimal MCP client over stdio. JSON-RPC 2.0, line-delimited JSON."""
    def __init__(self, name, command, args, env):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.proc = None
        self._rid = 0
        self._io_lock = threading.Lock()
        self._tools = []
        self._ready = False
        self._stderr_tail = []
        self._err_thread = None

    def start(self):
        if self.proc and self.proc.poll() is None: return
        merged_env = dict(os.environ); merged_env.update(self.env)
        try:
            self.proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=merged_env, bufsize=0, text=True, encoding="utf-8"
            )
        except FileNotFoundError as e:
            raise RuntimeError(f"MCP server '{self.name}' command not found: {self.command}") from e
        self._ready = False
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True); self._err_thread.start()
        # Handshake
        init = self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": False}, "sampling": {}},
            "clientInfo": {"name": "SourceShock-Tools", "version": "1.0"}
        }, timeout=20)
        if not init or "error" in init:
            raise RuntimeError(f"MCP '{self.name}' init failed: {init}")
        self._notify("notifications/initialized", {})
        self._ready = True
        self._refresh_tools()

    def _drain_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr_tail.append(line.rstrip())
                if len(self._stderr_tail) > 50: self._stderr_tail = self._stderr_tail[-50:]
        except Exception: pass

    def _next_id(self):
        self._rid += 1; return self._rid

    def _write(self, obj):
        line = json.dumps(obj) + "\n"
        with self._io_lock:
            if not self.proc or self.proc.poll() is not None:
                raise RuntimeError(f"MCP '{self.name}' process not running")
            self.proc.stdin.write(line); self.proc.stdin.flush()

    def _read_until(self, rid, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"MCP '{self.name}' exited: {''.join(self._stderr_tail[-10:])}")
            line = self.proc.stdout.readline()
            if not line: time.sleep(0.01); continue
            try: msg = json.loads(line.strip())
            except Exception: continue
            if "id" in msg and msg["id"] == rid: return msg
            # drop notifications / unrelated responses
        raise TimeoutError(f"MCP '{self.name}' timed out waiting for response")

    def _rpc(self, method, params, timeout=60):
        rid = self._next_id()
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        return self._read_until(rid, timeout=timeout)

    def _notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _refresh_tools(self):
        try:
            r = self._rpc("tools/list", {}, timeout=10)
            self._tools = (r or {}).get("result", {}).get("tools", []) or []
        except Exception: self._tools = []

    def tools(self): return list(self._tools)

    def call(self, tool_name, arguments, timeout=60):
        if not self._ready: self.start()
        r = self._rpc("tools/call", {"name": tool_name, "arguments": arguments or {}}, timeout=timeout)
        if not r: return {"error": "no response"}
        if "error" in r: return {"error": r["error"]}
        return r.get("result", {})

    def stop(self):
        try:
            if self.proc and self.proc.poll() is None:
                try: self.proc.stdin.close()
                except Exception: pass
                self.proc.terminate()
                try: self.proc.wait(timeout=3)
                except Exception: self.proc.kill()
        except Exception: pass
        self._ready = False

def get_client(name):
    with _clients_lock:
        c = _clients.get(name)
        if c and c.proc and c.proc.poll() is None: return c
        cfg = load_servers()["mcpServers"].get(name)
        if not cfg: raise KeyError(f"Unknown MCP server: {name}")
        if not cfg.get("enabled", False): raise RuntimeError(f"MCP server '{name}' disabled")
        c = MCPStdioClient(name, cfg["command"], cfg.get("args", []), cfg.get("env", {}))
        c.start()
        _clients[name] = c
        return c

def stop_all():
    with _clients_lock:
        for c in _clients.values():
            try: c.stop()
            except Exception: pass
        _clients.clear()

def list_all_tools():
    """Return [{qualified_name, server, name, description, schema}] for every enabled server."""
    out = []
    d = load_servers()
    for name, cfg in d.get("mcpServers", {}).items():
        if not cfg.get("enabled", False): continue
        try: c = get_client(name)
        except Exception as e:
            out.append({"qualified_name": f"{name}.__error__", "server": name, "name": "__error__",
                        "description": f"(unavailable: {e})", "schema": {}}); continue
        for t in c.tools():
            out.append({
                "qualified_name": f"{name}.{t.get('name')}",
                "server": name,
                "name": t.get("name"),
                "description": t.get("description", ""),
                "schema": t.get("inputSchema", {}),
            })
    return out

def call_qualified(qname, arguments, timeout=60):
    if "." not in qname: return {"error": "malformed tool name; expected server.tool"}
    server, tool = qname.split(".", 1)
    try: c = get_client(server)
    except Exception as e: return {"error": str(e)}
    return c.call(tool, arguments, timeout=timeout)

# ── Tool-tab session store ────────────────────────────────────────
def _session_path(sid): return os.path.join(TOOL_SESSIONS, f"{sid}.json")

def _read_session(sid):
    p = _session_path(sid)
    if not os.path.exists(p): return None
    try:
        with open(p) as f: return json.load(f)
    except Exception: return None

def _write_session(data):
    data["updated"] = datetime.now().isoformat()
    with _lock:
        with open(_session_path(data["id"]), "w") as f: json.dump(data, f)
    _bump()

def list_sessions():
    out = []
    for fp in sorted(glob.glob(os.path.join(TOOL_SESSIONS, "*.json")),
                     key=os.path.getmtime, reverse=True):
        try:
            with open(fp) as f: d = json.load(f)
            out.append({"id": d["id"], "title": d.get("title", "?"),
                        "updated": d.get("updated", ""),
                        "log_count": len(d.get("log", []))})
        except Exception: pass
    return out

def get_active_id():
    if os.path.exists(TOOL_ACTIVE_P):
        try:
            with open(TOOL_ACTIVE_P) as f: return f.read().strip()
        except Exception: pass
    return None

def set_active_id(sid):
    with open(TOOL_ACTIVE_P, "w") as f: f.write(sid)

def create_session(title=None):
    sid = uuid.uuid4().hex[:8]
    data = {"id": sid, "title": title or datetime.now().strftime("Task %Y-%m-%d %H:%M"),
            "log": [], "messages": [],
            "created": datetime.now().isoformat(), "updated": datetime.now().isoformat()}
    _write_session(data); set_active_id(sid); return data

def ensure_active():
    sid = get_active_id()
    if sid:
        d = _read_session(sid)
        if d: return d
    return create_session()

def append_log(sid, entry):
    d = _read_session(sid)
    if not d: return None
    d.setdefault("log", []).append({**entry, "ts": datetime.now().isoformat()})
    _write_session(d); return d

def append_message(sid, msg):
    d = _read_session(sid)
    if not d: return None
    d.setdefault("messages", []).append(msg); _write_session(d); return d

def delete_session(sid):
    p = _session_path(sid)
    if os.path.exists(p): os.remove(p)
    if get_active_id() == sid:
        # create a fresh one
        create_session()
    _bump()

# ── Pending summaries (injected into main chat as system messages) ─
def push_pending_summary(summary, session_id):
    arr = []
    if os.path.exists(PENDING_SUMMARIES):
        try:
            with open(PENDING_SUMMARIES) as f: arr = json.load(f)
        except Exception: arr = []
    arr.append({"summary": summary, "session_id": session_id,
                "ts": datetime.now().isoformat(), "consumed": False})
    with open(PENDING_SUMMARIES, "w") as f: json.dump(arr, f)

def pop_pending_summaries():
    """Returns and consumes all unconsumed pending summaries."""
    if not os.path.exists(PENDING_SUMMARIES): return []
    try:
        with open(PENDING_SUMMARIES) as f: arr = json.load(f)
    except Exception: arr = []
    out = [x for x in arr if not x.get("consumed")]
    for x in arr: x["consumed"] = True
    with open(PENDING_SUMMARIES, "w") as f: json.dump(arr, f)
    return out

def peek_pending_summaries():
    if not os.path.exists(PENDING_SUMMARIES): return []
    try:
        with open(PENDING_SUMMARIES) as f: arr = json.load(f)
    except Exception: arr = []
    return [x for x in arr if not x.get("consumed")]

# ── Tool model orchestration ──────────────────────────────────────
TOOL_BLOCK_RE = re.compile(r"```tool\s*\n([\s\S]*?)```", re.IGNORECASE)
ASK_BLOCK_RE  = re.compile(r"```ask_chat\s*\n([\s\S]*?)```", re.IGNORECASE)
THINK_RE      = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)

def _extract_think(text):
    thinks = [m.group(1).strip() for m in THINK_RE.finditer(text)]
    stripped = THINK_RE.sub("", text).strip()
    return stripped, ("\n\n".join(thinks) if thinks else "")

def _first_json_block(pat, text):
    m = pat.search(text)
    if not m: return None
    raw = m.group(1).strip()
    try: return json.loads(raw)
    except Exception:
        # try to salvage trailing-commas / fenced
        raw2 = raw.strip("`").strip()
        try: return json.loads(raw2)
        except Exception: return {"__raw__": raw}

def _ollama_chat(host, model, messages, temperature=0.3, think=False, timeout=300):
    """Call Ollama /api/chat non-streaming. Returns {content, thinking, err}."""
    if R is None: return {"content": "", "thinking": "", "err": "requests not installed"}
    try:
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature}}
        if think: payload["think"] = True
        r = R.post(f"{host}/api/chat", json=payload, timeout=timeout)
        if r.status_code != 200:
            return {"content": "", "thinking": "", "err": f"HTTP {r.status_code}: {r.text[:200]}"}
        j = r.json(); m = j.get("message", {})
        return {"content": m.get("content", "") or "",
                "thinking": m.get("thinking", "") or "",
                "err": None}
    except Exception as e:
        return {"content": "", "thinking": "", "err": str(e)}

def _build_tool_system(base_prompt, tools):
    lines = [base_prompt, "", "Available tools (server.tool):"]
    for t in tools:
        desc = (t.get("description") or "").strip().replace("\n", " ")
        if len(desc) > 160: desc = desc[:160] + "…"
        lines.append(f"  - {t['qualified_name']}: {desc}")
        schema = t.get("schema") or {}
        props = (schema.get("properties") or {})
        if props:
            req = set(schema.get("required") or [])
            args = ", ".join(f"{k}{'*' if k in req else ''}" for k in props.keys())
            lines.append(f"      args: {args}")
    if not tools:
        lines.append("  (none — reply with FINAL: <answer> / SUMMARY: <one sentence>)")
    return "\n".join(lines)

def run_tool_task(sid, task_text, cfg, tools, chat_ask_fn, emit):
    """Run the tool-model loop.
       chat_ask_fn(prompt) -> str      (delegates persona writing to main chat model)
       emit(event_dict)                (streams to SSE client on the Tools tab)
    Returns the final summary string (or None)."""
    host = cfg.get("ollama_host", "http://localhost:11434")
    model = cfg.get("model", "qwen3:4b")
    max_iters = int(cfg.get("max_iters", 8))
    temp = float(cfg.get("temperature", 0.3))
    think_on = bool(cfg.get("think_enabled", True))

    sys_msg = _build_tool_system(cfg.get("system_prompt", ""), tools)
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user",   "content": task_text}]

    append_message(sid, {"role": "system", "content": sys_msg, "hidden": True})
    append_message(sid, {"role": "user", "content": task_text})
    append_log(sid, {"kind": "task", "text": task_text}); emit({"kind": "task", "text": task_text})

    summary = None; final_answer = None
    for it in range(max_iters):
        emit({"kind": "step_start", "iter": it+1})
        resp = _ollama_chat(host, model, msgs, temperature=temp, think=think_on)
        if resp["err"]:
            append_log(sid, {"kind": "error", "text": resp["err"]}); emit({"kind": "error", "text": resp["err"]}); break
        content = resp["content"]; thinking = resp["thinking"]
        if not thinking:
            content, extracted = _extract_think(content); thinking = thinking or extracted
        append_message(sid, {"role": "assistant", "content": content, "thinking": thinking})
        if thinking:
            append_log(sid, {"kind": "think", "text": thinking}); emit({"kind": "think", "text": thinking})
        append_log(sid, {"kind": "assistant", "text": content}); emit({"kind": "assistant", "text": content})

        # 1) ask-chat delegation
        ask = _first_json_block(ASK_BLOCK_RE, content)
        if ask and "request" in ask:
            req = str(ask.get("request", ""))
            append_log(sid, {"kind": "ask_chat", "text": req}); emit({"kind": "ask_chat", "text": req})
            try: reply = chat_ask_fn(req) or ""
            except Exception as e: reply = f"(chat model error: {e})"
            append_log(sid, {"kind": "chat_reply", "text": reply}); emit({"kind": "chat_reply", "text": reply})
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "system", "content": f"Reply from chat model:\n{reply}"})
            continue

        # 2) tool call
        tc = _first_json_block(TOOL_BLOCK_RE, content)
        if tc and "name" in tc:
            qname = tc["name"]; args = tc.get("arguments") or tc.get("args") or {}
            append_log(sid, {"kind": "tool_call", "name": qname, "arguments": args})
            emit({"kind": "tool_call", "name": qname, "arguments": args})
            t0 = time.time()
            result = call_qualified(qname, args)
            dur = round(time.time() - t0, 2)
            append_log(sid, {"kind": "tool_result", "name": qname, "result": result, "duration": dur})
            emit({"kind": "tool_result", "name": qname, "result": result, "duration": dur})
            msgs.append({"role": "assistant", "content": content})
            msgs.append({"role": "system",
                         "content": f"Tool `{qname}` returned:\n{json.dumps(result)[:4000]}"})
            continue

        # 3) final answer
        mfinal = re.search(r"FINAL:\s*([\s\S]*?)(?:\nSUMMARY:|$)", content, re.IGNORECASE)
        msumm  = re.search(r"SUMMARY:\s*(.+)", content, re.IGNORECASE)
        if mfinal or msumm:
            final_answer = (mfinal.group(1).strip() if mfinal else content).strip()
            summary = (msumm.group(1).strip() if msumm else "").strip()
            if not summary:
                summary = final_answer.split("\n")[0][:280]
            append_log(sid, {"kind": "final", "text": final_answer, "summary": summary})
            emit({"kind": "final", "text": final_answer, "summary": summary})
            break

        # No recognised output — nudge once then stop.
        msgs.append({"role": "assistant", "content": content})
        msgs.append({"role": "system",
                     "content": "Reply with a tool call, an ask_chat block, or FINAL/SUMMARY."})

    if summary:
        push_pending_summary(summary, sid)
        emit({"kind": "summary_pushed", "summary": summary})
    else:
        append_log(sid, {"kind": "aborted", "text": f"No final answer after {max_iters} iterations."})
        emit({"kind": "aborted", "text": f"No final answer after {max_iters} iterations."})
    return summary
