from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from io import BytesIO
import base64
import copy
import threading
from typing import Any, Mapping


@dataclass
class ObserverAgentState:
    agent_id: int
    label: str
    mode: str = "autonomous"
    task: str | None = None
    decision_index: int = 0
    primitive_steps: int = 0
    pace: str = "normal"
    report: str = ""
    focus: str | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    body_feedback: dict[str, Any] | None = None
    spoken: list[str] = field(default_factory=list)
    heard: list[dict[str, Any]] = field(default_factory=list)
    used_vision: bool = False
    fallback: bool = False
    image_data_uri: str | None = None
    updated_at: str = ""


class ObserverDashboard:
    """
    Read-only local observer UI for a social embodied session.

    This object never contributes information to either agent's prompt. It only
    receives copies of externally observable runtime state after decisions have
    already been made. In particular, it does not expose or request hidden model
    chain-of-thought.
    """

    def __init__(
        self,
        labels: Mapping[int, str],
        host: str = "127.0.0.1",
        port: int = 8765,
        refresh_ms: int = 350,
        max_events: int = 200,
        image_width: int = 640,
        jpeg_quality: int = 75,
    ):
        self.labels = {int(k): str(v) for k, v in labels.items()}
        self.host = str(host)
        self.port = int(port)
        self.refresh_ms = max(100, int(refresh_ms))
        self.max_events = max(20, int(max_events))
        self.image_width = max(160, int(image_width))
        self.jpeg_quality = max(35, min(95, int(jpeg_quality)))
        self._lock = threading.RLock()
        self._states: dict[int, ObserverAgentState] = {
            agent_id: ObserverAgentState(agent_id=agent_id, label=label)
            for agent_id, label in self.labels.items()
        }
        self._events: list[dict[str, Any]] = []
        self._server = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _frame_data_uri(self, thor: Any) -> str | None:
        event = thor.raw_event()
        frame = None if event is None else getattr(event, "frame", None)
        if frame is None:
            return None
        try:
            from PIL import Image

            image = Image.fromarray(frame)
            if image.width > self.image_width:
                ratio = self.image_width / image.width
                image = image.resize((self.image_width, max(1, round(image.height * ratio))))
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return "data:image/jpeg;base64," + encoded
        except Exception:
            # The observer must never interfere with the embodied session.
            return None

    def append_event(
        self,
        kind: str,
        text: str,
        *,
        agent_id: int | None = None,
        peer_id: int | None = None,
    ) -> None:
        item = {
            "timestamp": self._now(),
            "kind": str(kind),
            "text": str(text)[:1000],
            "agent_id": None if agent_id is None else int(agent_id),
            "agent_label": None if agent_id is None else self.labels.get(int(agent_id), str(agent_id)),
            "peer_id": None if peer_id is None else int(peer_id),
            "peer_label": None if peer_id is None else self.labels.get(int(peer_id), str(peer_id)),
        }
        with self._lock:
            self._events.append(item)
            self._events = self._events[-self.max_events :]

    def refresh_frames(self, loops: Mapping[int, Any]) -> None:
        """Refresh every camera from the same current shared-world event."""
        now = self._now()
        captured = {int(agent_id): self._frame_data_uri(loop.thor) for agent_id, loop in loops.items()}
        with self._lock:
            for agent_id, data_uri in captured.items():
                if agent_id not in self._states:
                    continue
                if data_uri is not None:
                    self._states[agent_id].image_data_uri = data_uri
                self._states[agent_id].updated_at = now

    def update_agent(
        self,
        agent_id: int,
        loop: Any,
        *,
        mode: str,
        task: str | None,
        decision: Any | None = None,
        record: Mapping[str, Any] | None = None,
        heard: list[Any] | None = None,
    ) -> None:
        agent_id = int(agent_id)
        state = self._states[agent_id]
        record = dict(record or {})
        semantic_actions = record.get("semantic_actions") or []
        heard_rows = []
        for item in heard or []:
            if hasattr(item, "for_llm"):
                heard_rows.append(item.for_llm())
            elif isinstance(item, Mapping):
                heard_rows.append(dict(item))

        with self._lock:
            state.mode = str(mode)
            state.task = task
            state.decision_index = int(getattr(loop, "decision_index", state.decision_index))
            state.primitive_steps = int(getattr(loop.executor, "total_steps", state.primitive_steps))
            state.pace = str(record.get("pace", getattr(decision, "pace", "normal")))
            state.report = str(record.get("say", getattr(decision, "say", "")))[:1000]
            state.focus = record.get("attention", getattr(decision, "attention", None))
            state.actions = copy.deepcopy(list(semantic_actions))
            state.body_feedback = copy.deepcopy(record.get("body_feedback"))
            state.spoken = list(record.get("spoken") or getattr(decision, "spoken", None) or [])[-8:]
            state.heard = heard_rows[-8:]
            state.used_vision = bool(record.get("used_vision", getattr(decision, "used_vision", False)))
            state.fallback = bool(record.get("fallback", getattr(decision, "fallback", False)))
            state.updated_at = self._now()

        action_summary = " -> ".join(
            str(item.get("kind", "?")) + (f'(\"{item.get("target")}\")' if item.get("target") else "")
            for item in state.actions
        ) or "Observe"
        report_summary = state.report.strip() or "(no outward report)"
        self.append_event("decision", f"{action_summary}: {report_summary}", agent_id=agent_id)
        if state.fallback:
            self.append_event("fallback", state.report or "LLM fallback", agent_id=agent_id)
        self.refresh_frames({agent_id: loop})

    def set_task_state(self, agent_id: int, task: str | None) -> None:
        with self._lock:
            state = self._states[int(agent_id)]
            state.task = task
            state.mode = "task" if task else "autonomous"
            state.updated_at = self._now()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "server_time": self._now(),
                "refresh_ms": self.refresh_ms,
                "agents": [asdict(self._states[i]) for i in sorted(self._states)],
                "events": copy.deepcopy(self._events),
            }

    def _make_handler(self):
        from http.server import BaseHTTPRequestHandler
        import json

        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):  # noqa: N802 - stdlib handler API
                path = self.path.split("?", 1)[0]
                if path == "/":
                    html = _DASHBOARD_HTML.replace("__REFRESH_MS__", str(dashboard.refresh_ms)).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", html)
                    return
                if path == "/api/state":
                    payload = json.dumps(dashboard.snapshot(), ensure_ascii=False).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                if path == "/health":
                    payload = json.dumps({"ok": True, "agents": len(dashboard.labels)}).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", payload)
                    return
                self._send(404, "text/plain; charset=utf-8", b"Not found")

            def do_POST(self):  # noqa: N802 - stdlib handler API
                self._send(405, "text/plain; charset=utf-8", b"Read-only observer")

            def log_message(self, format, *args):  # noqa: A003 - stdlib signature
                return

        return Handler

    def start(self) -> None:
        if self._server is not None:
            return
        from http.server import ThreadingHTTPServer

        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        # port=0 is useful for tests and supported by the stdlib server.
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="observer-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

    @property
    def url(self) -> str:
        host = "localhost" if self.host in {"127.0.0.1", "0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}"


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI2-THOR Social Observer</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
body { margin:0; background:#0d1117; color:#e6edf3; }
header { display:flex; gap:16px; align-items:center; justify-content:space-between; padding:14px 18px; border-bottom:1px solid #30363d; position:sticky; top:0; background:#0d1117ee; backdrop-filter:blur(8px); z-index:5; }
header h1 { margin:0; font-size:18px; }
.sub { color:#8b949e; font-size:12px; }
.badge { display:inline-block; padding:3px 8px; border:1px solid #30363d; border-radius:999px; font-size:11px; margin-right:5px; }
.badge.warn { border-color:#d29922; color:#e3b341; }
.badge.ok { border-color:#238636; color:#3fb950; }
main { padding:16px; }
.grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.card { border:1px solid #30363d; border-radius:12px; overflow:hidden; background:#161b22; min-width:0; }
.cardhead { display:flex; justify-content:space-between; gap:12px; padding:10px 12px; border-bottom:1px solid #30363d; }
.cardhead strong { font-size:15px; }
.viewport { position:relative; aspect-ratio:4/3; background:#010409; display:flex; align-items:center; justify-content:center; }
.viewport img { width:100%; height:100%; object-fit:contain; }
.placeholder { color:#6e7681; font-size:13px; }
.detail { padding:12px; display:grid; gap:9px; }
.row { display:grid; grid-template-columns:112px minmax(0,1fr); gap:8px; align-items:start; font-size:12px; }
.key { color:#8b949e; }
.val { white-space:pre-wrap; overflow-wrap:anywhere; }
.report { min-height:42px; }
.action { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }
section.log { margin-top:14px; border:1px solid #30363d; border-radius:12px; background:#161b22; overflow:hidden; }
.log h2 { font-size:14px; margin:0; padding:10px 12px; border-bottom:1px solid #30363d; }
#events { max-height:300px; overflow:auto; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
.event { padding:6px 10px; border-bottom:1px solid #21262d; }
.event .time { color:#6e7681; }
.event.fallback { color:#e3b341; }
.event.speech { color:#a5d6ff; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div><h1>AI2-THOR Social Observer</h1><div class="sub">Read-only observer. Reports are explicit model output, not hidden chain-of-thought.</div></div>
  <div id="status" class="sub">connecting…</div>
</header>
<main>
  <div id="agents" class="grid"></div>
  <section class="log"><h2>Shared observer log</h2><div id="events"></div></section>
</main>
<script>
const REFRESH_MS = __REFRESH_MS__;
const esc = (x) => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function actionText(actions) {
  if (!actions || !actions.length) return 'Observe / no physical action';
  return actions.map(a => `${a.kind}${a.target ? '("'+a.target+'")' : ''}`).join(' → ');
}
function bodyText(b) {
  if (!b) return '';
  const out=[];
  if ('decision_success' in b) out.push(`success=${b.decision_success}`);
  if ('distance_moved' in b) out.push(`moved=${b.distance_moved}m`);
  if ('yaw_change_degrees' in b) out.push(`yawΔ=${b.yaw_change_degrees}°`);
  if ('decision_primitive_steps' in b) out.push(`primitive_steps=${b.decision_primitive_steps}`);
  return out.join(' · ');
}
function renderAgent(a) {
  const badges = [
    `<span class="badge">${esc(a.mode)}</span>`,
    a.used_vision ? '<span class="badge ok">vision</span>' : '<span class="badge">no image this turn</span>',
    a.fallback ? '<span class="badge warn">fallback</span>' : ''
  ].join('');
  const img = a.image_data_uri ? `<img src="${a.image_data_uri}" alt="Agent ${esc(a.label)} camera">` : '<div class="placeholder">No camera frame yet</div>';
  const heard = (a.heard||[]).map(h => `${h.speaker||'?'}: ${h.text||''}`).join('\n');
  const spoken = (a.spoken||[]).join('\n');
  return `<article class="card">
    <div class="cardhead"><div><strong>Agent ${esc(a.label)}</strong><div>${badges}</div></div><div class="sub">decision ${a.decision_index} · step ${a.primitive_steps}</div></div>
    <div class="viewport">${img}</div>
    <div class="detail">
      <div class="row"><div class="key">Vision/status</div><div class="val report">${esc(a.report || '—')}</div></div>
      <div class="row"><div class="key">Focus</div><div class="val">${esc(a.focus || '—')}</div></div>
      <div class="row"><div class="key">Chosen action</div><div class="val action">${esc(actionText(a.actions))}</div></div>
      <div class="row"><div class="key">Body feedback</div><div class="val">${esc(bodyText(a.body_feedback) || '—')}</div></div>
      <div class="row"><div class="key">Task</div><div class="val">${esc(a.task || 'autonomous')}</div></div>
      <div class="row"><div class="key">Said aloud</div><div class="val">${esc(spoken || '—')}</div></div>
      <div class="row"><div class="key">Heard</div><div class="val">${esc(heard || '—')}</div></div>
    </div>
  </article>`;
}
function renderEvents(events) {
  return (events||[]).slice().reverse().map(e => {
    const t = new Date(e.timestamp).toLocaleTimeString();
    const who = e.agent_label ? `[${e.agent_label}] ` : '';
    return `<div class="event ${esc(e.kind)}"><span class="time">${esc(t)}</span> ${esc(who+e.text)}</div>`;
  }).join('') || '<div class="event">No observer events yet.</div>';
}
async function tick() {
  try {
    const r = await fetch('/api/state', {cache:'no-store'});
    const s = await r.json();
    document.getElementById('agents').innerHTML = s.agents.map(renderAgent).join('');
    document.getElementById('events').innerHTML = renderEvents(s.events);
    document.getElementById('status').textContent = `live · ${new Date(s.server_time).toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById('status').textContent = 'dashboard disconnected';
  }
}
tick(); setInterval(tick, REFRESH_MS);
</script>
</body>
</html>'''
