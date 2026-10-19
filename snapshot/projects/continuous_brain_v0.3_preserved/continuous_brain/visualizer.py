from __future__ import annotations

from collections import deque
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous Brain v0.2</title>
<style>
:root{color-scheme:dark;--bg:#081018;--panel:#101a24;--panel2:#0b151f;--text:#e7f0f8;--muted:#8ca0b3;--line:#243546;--agent:#62d9ff;--target:#ffd166;--pred:#ff7aa8;--good:#74e0a1;--warn:#ffb454;--accent:#9e8cff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}.wrap{max-width:1180px;margin:auto;padding:18px}.top{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:14px}.title{font-size:22px;font-weight:750}.subtitle{color:var(--muted)}.controls{display:flex;gap:8px;flex-wrap:wrap}button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}button:hover{border-color:#48617a}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:14px}@media(max-width:820px){.grid{grid-template-columns:1fr}}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}.worldbox{position:relative;aspect-ratio:1/1;max-height:660px}canvas{width:100%;height:100%;display:block;border-radius:8px;background:var(--panel2)}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);margin-top:9px}.dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:9px}.label{color:var(--muted);font-size:12px}.value{font-size:18px;font-weight:700;margin-top:2px;overflow-wrap:anywhere}.wide{grid-column:1/-1}.section{margin-top:14px}.section h3{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:0 0 8px}.bars{display:grid;grid-template-columns:repeat(8,1fr);gap:3px;height:82px;align-items:end}.bar{background:var(--accent);min-height:2px;border-radius:3px 3px 0 0;opacity:.9}.qrow{display:grid;grid-template-columns:62px 1fr 56px;gap:8px;align-items:center;margin:6px 0}.qtrack{height:8px;background:#182635;border-radius:99px;overflow:hidden}.qfill{height:100%;background:var(--agent)}.statusline{display:flex;gap:10px;align-items:center;color:var(--muted)}.statusdot{width:9px;height:9px;border-radius:50%;background:var(--warn)}.statusdot.live{background:var(--good)}.foot{margin-top:12px;color:var(--muted);font-size:12px}.spark{height:74px}.badge{padding:2px 7px;border:1px solid var(--line);border-radius:99px;color:var(--muted)}
</style>
</head>
<body><div class="wrap">
<div class="top"><div><div class="title">Continuous Brain v0.2</div><div class="subtitle">Live world + neural learning monitor</div></div><div class="controls"><button id="pause">Pause</button><button id="slower">Slower</button><button id="faster">Faster</button><button id="normal">1×</button></div></div>
<div class="grid">
<section class="panel"><div class="worldbox"><canvas id="world" width="720" height="720"></canvas></div><div class="legend"><span><i class="dot" style="background:var(--agent)"></i>agent</span><span><i class="dot" style="background:var(--target)"></i>target</span><span><i class="dot" style="background:var(--pred)"></i>world-model prediction</span><span>faint line = trajectory</span></div></section>
<aside class="panel"><div class="statusline"><i id="statusdot" class="statusdot"></i><span id="connection">connecting…</span><span id="speed" class="badge">1×</span></div>
<div class="metrics section">
<div class="metric"><div class="label">Tick</div><div class="value" id="step">–</div></div><div class="metric"><div class="label">Action</div><div class="value" id="action">–</div></div>
<div class="metric"><div class="label">Goals</div><div class="value" id="goals">–</div></div><div class="metric"><div class="label">Energy</div><div class="value" id="energy">–</div></div>
<div class="metric"><div class="label">Total reward</div><div class="value" id="reward">–</div></div><div class="metric"><div class="label">Episode reward</div><div class="value" id="episode">–</div></div>
<div class="metric"><div class="label">Learning loss</div><div class="value" id="loss">–</div></div><div class="metric"><div class="label">Surprise</div><div class="value" id="surprise">–</div></div>
<div class="metric"><div class="label">Exploration ε</div><div class="value" id="epsilon">–</div></div><div class="metric"><div class="label">Distance</div><div class="value" id="distance">–</div></div>
<div class="metric"><div class="label">Replay</div><div class="value" id="replay">–</div></div><div class="metric"><div class="label">Episodic memories</div><div class="value" id="memories">–</div></div>
</div>
<div class="section"><h3>Action values (Q)</h3><div id="qvalues"></div></div>
<div class="section"><h3>Recurrent-state activity</h3><div id="hidden" class="bars"></div></div>
<div class="section"><h3>Loss history</h3><canvas id="losschart" class="spark" width="480" height="74"></canvas></div>
<div class="foot">The pink marker is where the learned world model expected the agent to be after the chosen action. Its separation from the blue agent is a visible component of prediction error.</div>
</aside></div></div>
<script>
(()=>{const $=id=>document.getElementById(id), world=$('world'),ctx=world.getContext('2d'),lc=$('losschart'),lctx=lc.getContext('2d');let state=null,losses=[],paused=false;const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();function xy(p){return [(p[0]+1)*.5*world.width,(1-(p[1]+1)*.5)*world.height]};function linePath(points,color,width=2,alpha=1){if(!points||points.length<2)return;ctx.save();ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.restore()};function drawWorld(s){ctx.clearRect(0,0,world.width,world.height);ctx.fillStyle=css('--panel2');ctx.fillRect(0,0,world.width,world.height);ctx.strokeStyle=css('--line');ctx.lineWidth=1;for(let i=1;i<10;i++){const v=i*world.width/10;ctx.beginPath();ctx.moveTo(v,0);ctx.lineTo(v,world.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,v);ctx.lineTo(world.width,v);ctx.stroke()}linePath(s.trail,css('--agent'),2,.28);if(s.predicted_position){const [ax,ay]=xy(s.agent),[px,py]=xy(s.predicted_position);ctx.setLineDash([7,7]);ctx.strokeStyle=css('--pred');ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(px,py);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=css('--pred');ctx.beginPath();ctx.arc(px,py,8,0,Math.PI*2);ctx.fill()}const [tx,ty]=xy(s.target);ctx.strokeStyle=css('--target');ctx.lineWidth=4;ctx.beginPath();ctx.arc(tx,ty,15,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.moveTo(tx-22,ty);ctx.lineTo(tx+22,ty);ctx.moveTo(tx,ty-22);ctx.lineTo(tx,ty+22);ctx.stroke();const [ax,ay]=xy(s.agent);ctx.fillStyle=css('--agent');ctx.beginPath();ctx.arc(ax,ay,12,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()};function fmt(v,n=3){return Number.isFinite(v)?v.toFixed(n):'–'}function updateQ(s){const box=$('qvalues');box.innerHTML='';const vals=s.q_values||[],names=s.action_names||[];if(!vals.length)return;const mn=Math.min(...vals),mx=Math.max(...vals),span=Math.max(1e-6,mx-mn);vals.forEach((v,i)=>{const r=document.createElement('div');r.className='qrow';const a=document.createElement('div');a.textContent=names[i]||String(i);const t=document.createElement('div');t.className='qtrack';const f=document.createElement('div');f.className='qfill';f.style.width=(12+88*(v-mn)/span)+'%';t.appendChild(f);const val=document.createElement('div');val.textContent=fmt(v,2);r.append(a,t,val);box.appendChild(r)})}function updateHidden(s){const box=$('hidden');box.innerHTML='';const vals=s.hidden_sample||[];const m=Math.max(.001,...vals.map(v=>Math.abs(v)));vals.forEach(v=>{const b=document.createElement('div');b.className='bar';b.style.height=(4+78*Math.abs(v)/m)+'px';b.style.opacity=String(.35+.65*Math.min(1,Math.abs(v)/m));box.appendChild(b)})}function drawLoss(){lctx.clearRect(0,0,lc.width,lc.height);lctx.fillStyle=css('--panel2');lctx.fillRect(0,0,lc.width,lc.height);const good=losses.filter(Number.isFinite);if(good.length<2)return;const max=Math.max(...good,.001),min=Math.min(...good,0);lctx.strokeStyle=css('--good');lctx.lineWidth=2;lctx.beginPath();losses.forEach((v,i)=>{const x=i/(losses.length-1)*lc.width,y=lc.height-(v-min)/(max-min||1)*(lc.height-6)-3;i?lctx.lineTo(x,y):lctx.moveTo(x,y)});lctx.stroke()}function render(s){state=s;drawWorld(s);$('step').textContent=s.step.toLocaleString();$('action').textContent=s.action_name;$('goals').textContent=s.goals;$('energy').textContent=fmt(s.energy,2);$('reward').textContent=(s.total_reward>=0?'+':'')+fmt(s.total_reward,2);$('episode').textContent=(s.episode_reward>=0?'+':'')+fmt(s.episode_reward,2);$('loss').textContent=fmt(s.loss,4);$('surprise').textContent=fmt(s.surprise,5);$('epsilon').textContent=fmt(s.epsilon,3);$('distance').textContent=fmt(s.distance,3);$('replay').textContent=s.replay.toLocaleString();$('memories').textContent=s.memories.toLocaleString();$('speed').textContent=fmt(s.speed,2)+'×';paused=!!s.paused;$('pause').textContent=paused?'Resume':'Pause';if(Number.isFinite(s.loss)){losses.push(s.loss);if(losses.length>160)losses.shift()}updateQ(s);updateHidden(s);drawLoss()}async function poll(){try{const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);const s=await r.json();render(s);$('statusdot').classList.add('live');$('connection').textContent=s.paused?'paused':'live'}catch(e){$('statusdot').classList.remove('live');$('connection').textContent='disconnected'}setTimeout(poll,100)}async function control(action){try{await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})}catch(e){}}$('pause').addEventListener('click',()=>control(paused?'resume':'pause'));$('slower').addEventListener('click',()=>control('slower'));$('faster').addEventListener('click',()=>control('faster'));$('normal').addEventListener('click',()=>control('normal'));poll()})();
</script></body></html>'''


class LiveVisualizer:
    """Tiny dependency-free browser dashboard for the continuously running agent."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, trail_length: int = 240):
        self.host = host
        self.port = int(port)
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {}
        self._trail: deque[list[float]] = deque(maxlen=max(20, int(trail_length)))
        self._paused = False
        self._speed = 1.0
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def set_paused(self, value: bool) -> None:
        with self._lock:
            self._paused = bool(value)
            self._state["paused"] = self._paused

    def set_speed(self, value: float) -> None:
        with self._lock:
            self._speed = max(0.1, min(20.0, float(value)))
            self._state["speed"] = self._speed

    def update(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            agent = snapshot.get("agent")
            if isinstance(agent, (list, tuple)) and len(agent) >= 2:
                self._trail.append([float(agent[0]), float(agent[1])])
            clean = dict(snapshot)
            clean["trail"] = list(self._trail)
            clean["paused"] = self._paused
            clean["speed"] = self._speed
            clean["server_time"] = time.time()
            self._state = clean

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _apply_control(self, action: str) -> None:
        action = action.strip().lower()
        if action == "pause":
            self.set_paused(True)
        elif action == "resume":
            self.set_paused(False)
        elif action == "slower":
            self.set_speed(self.speed / 2.0)
        elif action == "faster":
            self.set_speed(self.speed * 2.0)
        elif action == "normal":
            self.set_speed(1.0)

    def start(self) -> str:
        visualizer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt: str, *_args) -> None:
                return

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/":
                    self._send(HTTPStatus.OK, "text/html; charset=utf-8", _DASHBOARD_HTML.encode())
                elif path == "/state":
                    body = json.dumps(visualizer.snapshot(), separators=(",", ":")).encode()
                    self._send(HTTPStatus.OK, "application/json", body)
                elif path == "/health":
                    self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
                else:
                    self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

            def do_POST(self) -> None:
                if urlparse(self.path).path != "/control":
                    self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")
                    return
                try:
                    n = min(4096, int(self.headers.get("Content-Length", "0") or 0))
                    payload = json.loads(self.rfile.read(n) or b"{}")
                    visualizer._apply_control(str(payload.get("action", "")))
                    body = json.dumps({"ok": True, "paused": visualizer.paused, "speed": visualizer.speed}).encode()
                    self._send(HTTPStatus.OK, "application/json", body)
                except Exception as exc:
                    self._send(HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"ok": False, "error": str(exc)}).encode())

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="continuous-brain-visualizer", daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
