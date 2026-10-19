from __future__ import annotations

from collections import deque
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous Brain v0.5.1</title><style>
:root{color-scheme:dark;--bg:#071019;--panel:#101a24;--panel2:#0a141d;--text:#e7f0f8;--muted:#8fa4b8;--line:#26394b;--agent:#62d9ff;--target:#ffd166;--pred:#ff7aa8;--good:#74e0a1;--warn:#ffb454;--accent:#9e8cff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}.wrap{max-width:1320px;margin:auto;padding:16px}.top{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}.title{font-size:22px;font-weight:750}.subtitle,.muted{color:var(--muted)}.controls{display:flex;gap:8px;flex-wrap:wrap}button{border:1px solid var(--line);background:var(--panel);color:var(--text);padding:8px 12px;border-radius:8px;cursor:pointer}.layout{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(330px,.8fr);gap:14px}@media(max-width:900px){.layout{grid-template-columns:1fr}}.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}.views{display:grid;grid-template-columns:1fr 1fr;gap:10px}@media(max-width:680px){.views{grid-template-columns:1fr}}.view h3,.section h3{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 7px}.canvasbox{aspect-ratio:1/1;min-height:240px}.camerabox{aspect-ratio:4/3;min-height:240px}canvas{width:100%;height:100%;display:block;background:var(--panel2);border-radius:8px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}.metric{background:var(--panel2);border:1px solid var(--line);border-radius:9px;padding:8px}.label{font-size:11px;color:var(--muted)}.value{font-size:17px;font-weight:700;overflow-wrap:anywhere}.wide{grid-column:1/-1}.section{margin-top:12px}.bars{display:grid;grid-template-columns:repeat(16,1fr);gap:2px;height:70px;align-items:end}.bar{background:var(--accent);min-height:2px;border-radius:3px 3px 0 0}.qrow{display:grid;grid-template-columns:82px 1fr 52px;gap:7px;align-items:center;margin:5px 0}.qtrack{height:8px;background:#182635;border-radius:99px;overflow:hidden}.qfill{height:100%;background:var(--agent)}.statusline{display:flex;gap:9px;align-items:center;color:var(--muted);margin-bottom:10px}.dot{width:9px;height:9px;border-radius:50%;background:var(--warn)}.dot.live{background:var(--good)}.badge{padding:2px 7px;border:1px solid var(--line);border-radius:99px}.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:12px}.key{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.foot{margin-top:10px;color:var(--muted);font-size:12px}.spark{height:70px}
</style></head><body><div class="wrap">
<div class="top"><div><div class="title">Continuous Brain v0.5.1</div><div class="subtitle">persistent spatial cognition + self-generated waypoints + hierarchical skill retention</div></div><div class="controls"><button id="pause">Pause</button><button id="slower">Slower</button><button id="faster">Faster</button><button id="normal">1×</button></div></div>
<div class="layout"><section class="panel"><div class="views"><div class="view"><h3>Top-down spatial trace</h3><div class="canvasbox"><canvas id="world" width="640" height="640"></canvas></div><div class="legend"><span><i class="key" style="background:var(--agent)"></i>agent</span><span><i class="key" style="background:var(--target)"></i>goal</span><span><i class="key" style="background:var(--pred)"></i>world-model prediction (2-D world only)</span></div></div><div class="view"><h3>First-person RGB sensor</h3><div class="camerabox"><canvas id="camera" width="640" height="480"></canvas></div><div class="legend"><span>yaw <b id="yaw">–</b>°</span><span>horizon <b id="horizon">–</b>°</span><span>depth center <b id="depth">–</b> m</span></div></div></div><div class="foot">For AI2-THOR, the top-down plot is diagnostic only; the neural policy receives the configured sensor/goal slots plus its visual embedding, not this dashboard map.</div></section>
<aside class="panel"><div class="statusline"><i id="statusdot" class="dot"></i><span id="connection">connecting…</span><span id="speed" class="badge">1×</span></div><div class="metrics">
<div class="metric"><div class="label">Tick</div><div id="step" class="value">–</div></div><div class="metric"><div class="label">Action / primitive</div><div id="action" class="value">–</div></div>
<div class="metric"><div class="label">Curriculum</div><div id="curriculum" class="value">–</div></div><div class="metric"><div class="label">Task target</div><div id="targetlabel" class="value">–</div></div>
<div class="metric"><div class="label">Lifetime goals</div><div id="goals" class="value">–</div></div><div class="metric"><div class="label">Current-skill goals</div><div id="skillgoals" class="value">–</div></div><div class="metric wide"><div class="label">Distance / bearing</div><div id="distance" class="value">–</div></div>
<div class="metric"><div class="label">Total reward</div><div id="reward" class="value">–</div></div><div class="metric"><div class="label">Episode reward</div><div id="episode" class="value">–</div></div>
<div class="metric"><div class="label">Learning loss</div><div id="loss" class="value">–</div></div><div class="metric"><div class="label">Surprise</div><div id="surprise" class="value">–</div></div>
<div class="metric"><div class="label">Exploration ε</div><div id="epsilon" class="value">–</div></div><div class="metric"><div class="label">Goal signal</div><div id="goalsignal" class="value">–</div></div>
<div class="metric"><div class="label">Vision backbone</div><div id="vision" class="value">–</div></div><div class="metric"><div class="label">Visual adapter gate</div><div id="vgate" class="value">–</div></div>
<div class="metric"><div class="label">Curiosity</div><div id="curiosity" class="value">–</div></div><div class="metric"><div class="label">Visual novelty</div><div id="novelty" class="value">–</div></div>
<div class="metric"><div class="label">Replay / memories</div><div id="memory" class="value">–</div></div><div class="metric"><div class="label">Obstacle near fraction</div><div id="near" class="value">–</div></div>
<div class="metric"><div class="label">Allowed actions</div><div id="allowed" class="value">–</div></div><div class="metric"><div class="label">Stagnation</div><div id="stagnation" class="value">–</div></div>
<div class="metric"><div class="label">Navigation diagnostics</div><div id="navdiag" class="value">–</div></div><div class="metric"><div class="label">Spatial memory</div><div id="spatialdiag" class="value">–</div></div><div class="metric wide"><div class="label">Sensor integrity</div><div id="sensorhealth" class="value">–</div></div><div class="metric"><div class="label">Mastered skills</div><div id="mastered" class="value">–</div></div>
<div class="metric wide"><div class="label">Curriculum window / graduation</div><div id="gradstats" class="value">–</div></div>
</div><div class="section"><h3>Action values (Q)</h3><div id="qvalues"></div></div><div class="section"><h3>Recurrent-state activity</h3><div id="hidden" class="bars"></div></div><div class="section"><h3>Loss history</h3><canvas id="losschart" class="spark" width="480" height="70"></canvas></div></aside></div></div>
<script>(()=>{const $=id=>document.getElementById(id),wc=$('world'),wctx=wc.getContext('2d'),cc=$('camera'),cctx=cc.getContext('2d'),lc=$('losschart'),lctx=lc.getContext('2d');let losses=[],paused=false,lastCamSeq=-1,camOff=document.createElement('canvas'),camCtx=camOff.getContext('2d');const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();function fmt(v,n=3){return Number.isFinite(v)?v.toFixed(n):'–'}function bounds(s){const b=s.world_bounds;if(Array.isArray(b)&&b.length===4&&b[1]>b[0]&&b[3]>b[2])return b;return[-1,1,-1,1]}function xy(p,s){const b=bounds(s),pad=28;return[pad+(p[0]-b[0])/(b[1]-b[0])*(wc.width-2*pad),wc.height-pad-(p[1]-b[2])/(b[3]-b[2])*(wc.height-2*pad)]}function drawWorld(s){wctx.fillStyle=css('--panel2');wctx.fillRect(0,0,wc.width,wc.height);wctx.strokeStyle=css('--line');for(let i=1;i<10;i++){const v=i*wc.width/10;wctx.beginPath();wctx.moveTo(v,0);wctx.lineTo(v,wc.height);wctx.stroke();wctx.beginPath();wctx.moveTo(0,v);wctx.lineTo(wc.width,v);wctx.stroke()}const sm=s.spatial;if(sm&&Array.isArray(sm.values)&&Array.isArray(sm.shape)&&sm.shape.length===2){const rows=sm.shape[0],cols=sm.shape[1],b=sm.bounds||bounds(s);for(let rz=0;rz<rows;rz++){for(let rx=0;rx<cols;rx++){const v=sm.values[rz*cols+rx]||0;if(Math.abs(v)<4)continue;const x0=b[0]+rx/(Math.max(1,cols-1))*(b[1]-b[0]),z0=b[2]+rz/(Math.max(1,rows-1))*(b[3]-b[2]);const x1=b[0]+(rx+1)/(Math.max(1,cols-1))*(b[1]-b[0]),z1=b[2]+(rz+1)/(Math.max(1,rows-1))*(b[3]-b[2]);const p0=xy([x0,z0],s),p1=xy([x1,z1],s);wctx.globalAlpha=Math.min(.5,.10+Math.abs(v)/180);wctx.fillStyle=v>0?css('--warn'):css('--good');wctx.fillRect(Math.min(p0[0],p1[0]),Math.min(p0[1],p1[1]),Math.max(1,Math.abs(p1[0]-p0[0])+1),Math.max(1,Math.abs(p1[1]-p0[1])+1));}}wctx.globalAlpha=1}const tr=s.trail||[];if(tr.length>1){wctx.strokeStyle=css('--agent');wctx.globalAlpha=.28;wctx.lineWidth=2;wctx.beginPath();tr.forEach((p,i)=>{const [x,y]=xy(p,s);i?wctx.lineTo(x,y):wctx.moveTo(x,y)});wctx.stroke();wctx.globalAlpha=1}if(Array.isArray(s.target)){const[t1,t2]=xy(s.target,s);wctx.strokeStyle=css('--target');wctx.lineWidth=4;wctx.beginPath();wctx.arc(t1,t2,13,0,Math.PI*2);wctx.stroke()}if(Array.isArray(s.waypoint)){const[w1,w2]=xy(s.waypoint,s);wctx.fillStyle=css('--good');wctx.beginPath();wctx.arc(w1,w2,7,0,Math.PI*2);wctx.fill();if(Array.isArray(s.agent)){const[a1,a2]=xy(s.agent,s);wctx.setLineDash([4,5]);wctx.strokeStyle=css('--good');wctx.lineWidth=2;wctx.beginPath();wctx.moveTo(a1,a2);wctx.lineTo(w1,w2);wctx.stroke();wctx.setLineDash([])}}if(Array.isArray(s.predicted_position)&&Array.isArray(s.agent)){const[a1,a2]=xy(s.agent,s),[p1,p2]=xy(s.predicted_position,s);wctx.setLineDash([6,6]);wctx.strokeStyle=css('--pred');wctx.beginPath();wctx.moveTo(a1,a2);wctx.lineTo(p1,p2);wctx.stroke();wctx.setLineDash([])}if(Array.isArray(s.agent)){const[x,y]=xy(s.agent,s);wctx.fillStyle=css('--agent');wctx.beginPath();wctx.arc(x,y,10,0,Math.PI*2);wctx.fill();if(Number.isFinite(s.yaw)){const a=(s.yaw-90)*Math.PI/180;wctx.strokeStyle='#fff';wctx.lineWidth=2;wctx.beginPath();wctx.moveTo(x,y);wctx.lineTo(x+22*Math.cos(a),y+22*Math.sin(a));wctx.stroke()}}}function drawCamera(s){const cam=s.camera,sh=s.sensor_health||{};if(!cam||!cam.rgb_b64){cctx.fillStyle=css('--panel2');cctx.fillRect(0,0,cc.width,cc.height);cctx.fillStyle=css('--warn');cctx.font='18px system-ui';cctx.fillText('camera frame unavailable',18,32);return}try{const seq=Number.isFinite(cam.seq)?cam.seq:null;if(seq===null||seq!==lastCamSeq){const bin=atob(cam.rgb_b64),arr=new Uint8ClampedArray(bin.length/3*4);for(let i=0,j=0;i<bin.length;i+=3,j+=4){arr[j]=bin.charCodeAt(i);arr[j+1]=bin.charCodeAt(i+1);arr[j+2]=bin.charCodeAt(i+2);arr[j+3]=255}if(camOff.width!==cam.width||camOff.height!==cam.height){camOff.width=cam.width;camOff.height=cam.height}camCtx.putImageData(new ImageData(arr,cam.width,cam.height),0,0);lastCamSeq=seq===null?lastCamSeq+1:seq}cctx.fillStyle=css('--panel2');cctx.fillRect(0,0,cc.width,cc.height);cctx.imageSmoothingEnabled=false;cctx.drawImage(camOff,0,0,cc.width,cc.height);if(sh.rgb_stale){cctx.fillStyle='rgba(170,35,35,.75)';cctx.fillRect(0,0,cc.width,44);cctx.fillStyle='#fff';cctx.font='bold 20px system-ui';cctx.fillText('RGB SENSOR STALE',16,29)}}catch(e){cctx.fillStyle=css('--warn');cctx.font='16px system-ui';cctx.fillText('camera render error: '+String(e).slice(0,80),12,28)}}function updateQ(s){const box=$('qvalues');box.innerHTML='';const vals=s.q_values||[],names=s.action_names||[],allowed=new Set(s.allowed_actions||names.map((_,i)=>i));if(!vals.length)return;const validVals=vals.filter((_,i)=>allowed.has(i)),mn=Math.min(...(validVals.length?validVals:vals)),mx=Math.max(...(validVals.length?validVals:vals)),span=Math.max(1e-6,mx-mn);vals.forEach((v,i)=>{const r=document.createElement('div');r.className='qrow';if(!allowed.has(i))r.style.opacity='.28';const a=document.createElement('div');a.textContent=(names[i]||i)+(allowed.has(i)?'':' [masked]');const t=document.createElement('div');t.className='qtrack';const f=document.createElement('div');f.className='qfill';f.style.width=(12+88*(v-mn)/span)+'%';t.appendChild(f);const val=document.createElement('div');val.textContent=fmt(v,2);r.append(a,t,val);box.appendChild(r)})}function updateHidden(s){const box=$('hidden');box.innerHTML='';const vals=s.hidden_sample||[],m=Math.max(.001,...vals.map(v=>Math.abs(v)));vals.forEach(v=>{const b=document.createElement('div');b.className='bar';b.style.height=(3+65*Math.abs(v)/m)+'px';b.style.opacity=.35+.65*Math.min(1,Math.abs(v)/m);box.appendChild(b)})}function drawLoss(){lctx.fillStyle=css('--panel2');lctx.fillRect(0,0,lc.width,lc.height);const g=losses.filter(Number.isFinite);if(g.length<2)return;const max=Math.max(...g,.001),min=Math.min(...g,0);lctx.strokeStyle=css('--good');lctx.lineWidth=2;lctx.beginPath();losses.forEach((v,i)=>{const x=i/(losses.length-1)*lc.width,y=lc.height-(v-min)/(max-min||1)*(lc.height-6)-3;i?lctx.lineTo(x,y):lctx.moveTo(x,y)});lctx.stroke()}function render(s){drawWorld(s);drawCamera(s);$('step').textContent=(s.step||0).toLocaleString();$('action').textContent=(s.action_name||'–')+(s.primitive_action?' / '+s.primitive_action:'');$('curriculum').textContent=Number.isFinite(s.configured_curriculum_level)?'L'+s.configured_curriculum_level+(s.is_rehearsal?' · rehearse L'+s.curriculum_level:'')+' '+(s.task_kind||''):(s.task_kind||'–');$('targetlabel').textContent=s.target_label||'–';$('goals').textContent=s.goals??'–';const csg=s.curriculum_stats||{};$('skillgoals').textContent=csg.goals_total??0;$('distance').textContent='wp '+fmt(s.distance,2)+' m / '+fmt(s.bearing,1)+'° · final '+fmt(s.final_distance,2)+' m'+((s.waypoint_count||0)?' · '+(s.waypoint_index+1)+'/'+s.waypoint_count:'');$('reward').textContent=fmt(s.total_reward,2);$('episode').textContent=fmt(s.episode_reward,2);$('loss').textContent=fmt(s.loss,4);$('surprise').textContent=fmt(s.surprise,5);$('epsilon').textContent=fmt(s.epsilon,3)+' (floor '+fmt(s.level_epsilon_floor,3)+')';$('goalsignal').textContent=(s.goal_signal||'–')+(s.waypoint_source?' · '+s.waypoint_source:'');$('vision').textContent=(s.vision_backbone||'none')+(s.vision_pretrained?' ★':'');$('vgate').textContent=fmt(s.visual_gate,3);$('curiosity').textContent=fmt(s.intrinsic_reward,5);$('novelty').textContent=fmt(s.visual_novelty,5);$('memory').textContent=(s.replay||0)+' / '+(s.memories||0)+' + '+(s.visual_memories||0)+'v';$('near').textContent=fmt(s.near_obstacle_fraction,3);const names=s.action_names||[],aa=s.allowed_actions||[];$('allowed').textContent=aa.map(i=>names[i]||i).join(', ')||'all';$('stagnation').textContent=(s.stagnation_steps??0)+' / '+(s.stagnation_limit??0);$('navdiag').textContent='move '+(s.successful_translations_episode??0)+' · blocked '+(s.blocked_advances_episode??0)+' · replans '+(s.replans_episode??0)+' · edges '+(s.known_blocked_edges??0);const sp=s.spatial_stats||(s.spatial&&s.spatial.counts)||{};$('spatialdiag').textContent='known '+(sp.known??0)+'/'+(sp.cells??0)+' · free '+(sp.free??0)+' · occ '+(sp.occupied??0)+' · landmarks '+(sp.landmarks??0)+' · plans '+(sp.plans??0)+'/'+(sp.plan_failures??0);const sh=s.sensor_health||{};$('sensorhealth').textContent='frame '+(sh.frame_seq??'–')+' · hash '+(sh.frame_hash??'–')+' · diff '+fmt(sh.rgb_diff,2)+' · stale '+(sh.rgb_stale?'YES':'no')+' ('+(sh.rgb_stale_motion_events??0)+') · depth valid '+(100*(sh.depth_valid_fraction??0)).toFixed(0)+'% · faults '+(sh.faults_total??0);$('mastered').textContent=(s.mastered_levels||[]).map(x=>'L'+x).join(', ')||'none';const cs=s.curriculum_stats||{},sr=Number.isFinite(cs.recent_success_rate)?(100*cs.recent_success_rate).toFixed(1)+'%':'–',med=Number.isFinite(cs.median_success_steps)?cs.median_success_steps.toFixed(1):'–',to=Number.isFinite(cs.recent_timeout_rate)?(100*cs.recent_timeout_rate).toFixed(1)+'%':'–';$('gradstats').textContent='L'+(cs.level??s.configured_curriculum_level??'–')+' · '+(cs.recent_attempts??0)+'/'+(cs.window??'–')+' episodes · success '+sr+' · median '+med+' steps · goals '+(cs.goals_total??0)+' · timeout '+to+(s.graduation_ready?' · READY':'');$('yaw').textContent=fmt(s.yaw,1);$('horizon').textContent=fmt(s.camera_horizon,1);$('depth').textContent=fmt(s.depth_center,2);$('speed').textContent=fmt(s.speed,2)+'×';paused=!!s.paused;$('pause').textContent=paused?'Resume':'Pause';if(Number.isFinite(s.loss)){losses.push(s.loss);if(losses.length>180)losses.shift()}updateQ(s);updateHidden(s);drawLoss()}async function poll(){try{const r=await fetch('/state',{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json());$('statusdot').classList.add('live');$('connection').textContent=paused?'paused':'live'}catch(e){$('statusdot').classList.remove('live');$('connection').textContent='disconnected'}setTimeout(poll,100)}async function ctl(action){try{await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})}catch(e){}}$('pause').onclick=()=>ctl(paused?'resume':'pause');$('slower').onclick=()=>ctl('slower');$('faster').onclick=()=>ctl('faster');$('normal').onclick=()=>ctl('normal');poll()})();</script></body></html>'''


class LiveVisualizer:
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
        if action == "pause": self.set_paused(True)
        elif action == "resume": self.set_paused(False)
        elif action == "slower": self.set_speed(self.speed / 2.0)
        elif action == "faster": self.set_speed(self.speed * 2.0)
        elif action == "normal": self.set_speed(1.0)

    def start(self) -> str:
        visualizer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _fmt: str, *_args) -> None: return
            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/": self._send(HTTPStatus.OK, "text/html; charset=utf-8", _DASHBOARD_HTML.encode())
                elif path == "/state": self._send(HTTPStatus.OK, "application/json", json.dumps(visualizer.snapshot(), separators=(",", ":")).encode())
                elif path == "/health": self._send(HTTPStatus.OK, "text/plain; charset=utf-8", b"ok\n")
                else: self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")
            def do_POST(self) -> None:
                if urlparse(self.path).path != "/control": self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n"); return
                try:
                    n=min(4096,int(self.headers.get("Content-Length","0") or 0)); payload=json.loads(self.rfile.read(n) or b"{}"); visualizer._apply_control(str(payload.get("action",""))); self._send(HTTPStatus.OK,"application/json",json.dumps({"ok":True,"paused":visualizer.paused,"speed":visualizer.speed}).encode())
                except Exception as exc: self._send(HTTPStatus.BAD_REQUEST,"application/json",json.dumps({"ok":False,"error":str(exc)}).encode())
        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, name="continuous-brain-visualizer", daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown(); self._server.server_close(); self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0); self._thread = None
