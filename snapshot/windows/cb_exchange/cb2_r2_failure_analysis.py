import json, glob, sys
import numpy as np, pyarrow.parquet as pq
ROOT = sys.argv[1]
INV = ("health","food","drink","energy","sapling","wood","stone","coal","iron","diamond","wood_pickaxe","stone_pickaxe","iron_pickaxe","wood_sword","stone_sword","iron_sword")
ACT = ["noop","move_left","move_right","move_up","move_down","do","sleep","place_stone","place_table","place_furnace","place_plant","make_wood_pickaxe","make_stone_pickaxe","make_iron_pickaxe","make_wood_sword","make_stone_sword","make_iron_sword"]
OPT = ("routine","explore","sleeping","eat")
WATER, TREE = 1, 6; COW = 2; ZOMBIE = 3
def load(stream):
    lives = []
    for sj in sorted(glob.glob(f"{ROOT}/{stream}/shard-*.json")):
        b = sj[:-5]
        s = pq.read_table(b+".student.parquet", columns=["tick","life","step","seed","inventory","action","messages","done","dying"]).to_pydict()
        t = pq.read_table(b+".teacher.parquet", columns=["tick","teacher_action","teacher_option","view_material","view_object"]).to_pydict()
        assert s["tick"] == t["tick"]
        lives.append({k: np.array(v) if k not in ("messages",) else v for k, v in {**s, **t}.items()})
    return lives
def since(mask):
    out = np.full(len(mask), 10**6); last = None
    for i, m in enumerate(mask):
        if m: last = i
        out[i] = 0 if m else (i - last if last is not None else 10**6)
    return out
def bucket(x, edges):
    labs=[]; 
    for v in x:
        for lo, hi, name in edges:
            if lo <= v <= hi: labs.append(name); break
        else: labs.append("never seen")
    return np.array(labs)
cand, rout = load("eval-r2-PA"), load("eval-r2-PA-routine")
R = {}
# 1. overall + drift over the life
A = np.concatenate([l["action"] == l["teacher_action"] for l in cand]); ST = np.concatenate([l["step"] for l in cand])
R["closed_loop_agreement"] = round(float(A.mean()), 3); R["candidate_steps"] = int(len(A))
R["agreement_by_step_in_life"] = {f"{lo}-{hi}": [round(float(A[(ST>=lo)&(ST<=hi)].mean()),3), int(((ST>=lo)&(ST<=hi)).sum())] for lo,hi in [(0,49),(50,99),(100,199),(200,399),(400,10**6)] if ((ST>=lo)&(ST<=hi)).sum()}
# 2. by what the routine wanted to do
O = np.concatenate([l["teacher_option"] for l in cand])
R["agreement_by_routine_option"] = {OPT[o]: [round(float(A[O==o].mean()),3), int((O==o).sum())] for o in range(4) if (O==o).sum()}
TA = np.concatenate([l["teacher_action"] for l in cand]); CA = np.concatenate([l["action"] for l in cand])
R["agreement_by_routine_action"] = {ACT[a]: [round(float(A[TA==a].mean()),3), int((TA==a).sum())] for a in range(17) if (TA==a).sum() >= 30}
R["candidate_action_share"] = {ACT[a]: round(float((CA==a).mean()),3) for a in range(17) if (CA==a).mean() > 0.01}
R["routine_cf_action_share"] = {ACT[a]: round(float((TA==a).mean()),3) for a in range(17) if (TA==a).mean() > 0.01}
# 3. memory test: thirst/hunger x time since the resource was last in view (difference-in-differences vs not-needy steps)
E = [(0,0,"in view"),(1,10,"1-10 steps ago"),(11,40,"11-40"),(41,150,"41-150"),(151,10**6-1,">150")]
def mem_table(lives, inv_idx, mat=None, obj=None, low=4):
    rows = {"needy": {}, "not needy": {}}
    acc = {"needy": [], "not needy": []}
    for l in lives:
        vis = (l["view_material"] == mat).any(1) if mat is not None else (l["view_object"] == obj).any(1)
        sb = bucket(since(vis), E); need = l["inventory"][:, inv_idx] <= low; ag = l["action"] == l["teacher_action"]
        for k, m in (("needy", need), ("not needy", ~need)):
            acc[k].append((sb[m], ag[m]))
    for k in acc:
        sb = np.concatenate([a for a, _ in acc[k]]); ag = np.concatenate([b for _, b in acc[k]])
        for _,_,name in E + [(0,0,"never seen")]:
            m = sb == name
            if m.sum() >= 30: rows[k][name] = [round(float(ag[m].mean()),3), int(m.sum())]
    return rows
R["memory_drink_vs_water_last_seen"] = mem_table(cand, INV.index("drink"), mat=WATER)
R["memory_food_vs_cow_last_seen"] = mem_table(cand, INV.index("food"), obj=COW)
# 4. requests: time since the last caretaker message
acc = []
for l in cand:
    heard = np.array([m not in ("", "[]") for m in l["messages"]]); acc.append((bucket(since(heard), E), l["action"] == l["teacher_action"]))
sb = np.concatenate([a for a,_ in acc]); ag = np.concatenate([b for _,b in acc])
R["agreement_vs_last_message_heard"] = {n: [round(float(ag[sb==n].mean()),3), int((sb==n).sum())] for _,_,n in E+[(0,0,"never seen")] if (sb==n).sum()>=30}
# 5. how lives end: vitals at the last step
def ends(lives):
    c = {"food 0":0,"drink 0":0,"energy 0":0,"health<=2 with food,drink,energy>0":0}; n=len(lives); L=[]
    for l in lives:
        v = l["inventory"][-1]; L.append(len(l["step"]))
        if v[1]==0: c["food 0"]+=1
        if v[2]==0: c["drink 0"]+=1
        if v[3]==0: c["energy 0"]+=1
        if v[0]<=2 and v[1]>0 and v[2]>0 and v[3]>0: c["health<=2 with food,drink,energy>0"]+=1
    return {"lives": n, "mean_steps": round(float(np.mean(L)),1), **c}
R["life_endings"] = {"candidate": ends(cand), "routine": ends(rout)}
# 6. drift: how often the routine (counterfactual) wants survival options in the candidate's lives vs its own
Orout = np.concatenate([l["teacher_option"] for l in rout])
R["routine_option_share"] = {"in candidate lives (counterfactual)": {OPT[o]: round(float((O==o).mean()),3) for o in range(4)},
                             "in routine's own lives": {OPT[o]: round(float((Orout==o).mean()),3) for o in range(4)}}
print(json.dumps(R, indent=1))
