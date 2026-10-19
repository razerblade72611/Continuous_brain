"""Local, dependency-free dashboard. All text uses textContent, never user HTML."""
from .visualizer import LiveVisualizer

HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous Brain · Persistent Individual Core</title>
<style>
body{background:#08131d;color:#dfeefa;font:16px system-ui;margin:0;padding:28px;max-width:1200px;margin:auto}
h1{font-size:25px;font-weight:500}small,.muted{color:#93aabd}section{background:#101f2c;border:1px solid #263d50;border-radius:12px;padding:18px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}.value{font-size:24px;color:#76d7cf;margin-top:8px}
button{background:#203b50;color:white;border:1px solid #406078;border-radius:8px;padding:9px 18px;cursor:pointer}
canvas{width:100%;max-width:490px;background:#0a1520;border-radius:8px;image-rendering:pixelated}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}label{font-size:13px;color:#93aabd}
</style><h1>Continuous Brain <small>v0.6 · Persistent Individual Core</small></h1>
<p class="muted">An experimental learning system. These indicators are not evidence of consciousness.</p>
<p id="identity">Connecting…</p><button onclick="ctl('pause')">Pause</button> <button onclick="ctl('resume')">Resume</button>
<button onclick="ctl('slower')">Slower</button> <button onclick="ctl('faster')">Faster</button>
<div class="grid"><section><label>Lifetime decisions / motor ticks</label><div class="value" id="ticks">—</div></section>
<section><label>Active intention / skill</label><div class="value" id="goal">—</div></section>
<section><label>Learning updates / memories</label><div class="value" id="learning">—</div></section>
<section><label>Task successes / attempts</label><div class="value" id="tasks">—</div></section></div>
<div class="grid"><section><label>Egocentric sensory image</label><p><canvas id="camera" width="96" height="96"></canvas></p></section>
<section><label>Evaluator-only world view · never a cognitive input</label><p><canvas id="world" width="400" height="400"></canvas></p></section></div>
<section><label>Skill composition and provenance</label><pre id="plan"></pre></section>
<section><label>Empirical self-state and learning telemetry</label><pre id="state"></pre></section>
<script>
const el=id=>document.getElementById(id);
async function ctl(action){await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})})}
function camera(c){if(!c||!c.rgb_b64)return;const cv=el('camera');cv.width=c.width;cv.height=c.height;const raw=atob(c.rgb_b64),rgba=new Uint8ClampedArray(c.width*c.height*4);for(let i=0,j=0;i<raw.length;i+=3,j+=4){rgba[j]=raw.charCodeAt(i);rgba[j+1]=raw.charCodeAt(i+1);rgba[j+2]=raw.charCodeAt(i+2);rgba[j+3]=255}cv.getContext('2d').putImageData(new ImageData(rgba,c.width,c.height),0,0)}
function world(s){const ctx=el('world').getContext('2d');ctx.fillStyle='#0a1520';ctx.fillRect(0,0,400,400);const b=s.world_bounds||[-3,3,-3,3],xy=p=>[(p[0]-b[0])/(b[1]-b[0])*380+10,390-(p[1]-b[2])/(b[3]-b[2])*380];ctx.strokeStyle='#294155';ctx.strokeRect(10,10,380,380);for(const [name,color] of [['target','#f7c65a'],['agent','#68d5cd']]){if(!s[name])continue;const p=xy(s[name]);ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill()}}
async function poll(){try{const s=await(await fetch('/state',{cache:'no-store'})).json(),i=s.individual;if(!i)return;const m=i.individual;el('identity').textContent=m.name+' · '+m.individual_id+(s.paused?' · paused':'');el('ticks').textContent=m.decisions+' / '+m.ticks;el('goal').textContent=i.intention.goal+' / '+i.last_skill;el('learning').textContent=(i.training.updates||0)+' / '+i.memory_count;el('tasks').textContent=m.successes+' / '+m.tasks;el('plan').textContent=JSON.stringify({plan:i.plan,selection:i.selection_source,motor:i.motor_source,teacher_probability:i.teacher_probability,retrieved_events:i.memory_ids,evaluation:i.evaluation},null,2);el('state').textContent=JSON.stringify({energy:m.energy,surprise:m.surprise,uncertainty:m.uncertainty,workspace_norm:i.workspace_norm,skill_statistics:m.skill_stats,training:i.training},null,2);camera(s.camera);world(s)}catch(e){el('identity').textContent='Disconnected · '+e.message}finally{setTimeout(poll,500)}}poll();
</script></html>'''


def individual_visualizer(host="127.0.0.1", port=8765):
    return LiveVisualizer(host, port, html=HTML)
