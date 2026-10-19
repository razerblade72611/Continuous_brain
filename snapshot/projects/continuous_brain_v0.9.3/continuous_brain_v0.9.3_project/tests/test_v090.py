from pathlib import Path
import json
import numpy as np
import torch

from continuous_brain.continuity_v090 import ContinuityStore
from continuous_brain.infant import InfantBrain, InfantConfig, byte_encode, BYTE_PAD
from continuous_brain.parented_data import NurseryDataset
from continuous_brain.parented_train import save_checkpoint, load_checkpoint
from continuous_brain.parenting import ParentRegistry, ParentSpec


def pad(rows):
    width=max(map(len,rows)); out=torch.full((len(rows),width),BYTE_PAD,dtype=torch.long)
    for i,row in enumerate(rows): out[i,:len(row)]=torch.tensor(row)
    return out


def test_raw_child_forward_and_detachable_parent_head():
    cfg=InfantConfig.for_profile('proof')
    m=InfantBrain(cfg, vision_parent_dims={'vision.dinov3.base':768})
    images=torch.rand(2,3,96,96)
    visual=m.vision(images)
    assert visual.shape==(2,cfg.bus_dim)
    assert m.audio(torch.rand(2,16000)).shape==(2,cfg.bus_dim)
    assert m.parent_visual_prediction('vision.dinov3.base', visual).shape==(2,768)
    state=m.nucleus(visual, torch.zeros(2,48), torch.zeros(2,64), torch.zeros(2,dtype=torch.long), m.nucleus.initial_state(2,'cpu'))
    assert state['workspace'].shape==(2,cfg.bus_dim)
    q=pad([byte_encode('Who are you?',64), byte_encode('What do you see?',64)])
    a=pad([byte_encode('I am learning.',64), byte_encode('A room.',64)])
    logits=m.language(state['workspace'],q,a[:,:-1])
    assert logits.shape[:2]==a[:,:-1].shape
    assert m.parameter_count > 1_000_000


def test_parent_registry_weans_and_upgrades(tmp_path):
    r=ParentRegistry(tmp_path/'parents.json')
    assert r.active_parent('vision') is not None
    before=r.guidance_probability('motor')
    for _ in range(3): r.record_score('motor',0.95,threshold=0.8,required_passes=3,step=0.25)
    assert r.guidance_probability('motor') < before
    spec=ParentSpec('language.future','language','Future/Model','openai-chat')
    r.upgrade(spec)
    assert r.active_parent('language').parent_id=='language.future'
    r2=ParentRegistry(tmp_path/'parents.json')
    assert r2.active_parent('language').model=='Future/Model'


def test_checkpoint_roundtrip(tmp_path):
    m=InfantBrain(InfantConfig.for_profile('proof'),vision_parent_dims={'vision.dinov3.legacy':384})
    path=tmp_path/'brain.pt'
    save_checkpoint(path,m,updates=7,parent_id='vision.dinov3.legacy',dataset_digest='abc')
    restored,state=load_checkpoint(path)
    assert state['updates']==7
    assert restored.parameter_count==m.parameter_count


def test_continuity_transaction_and_provenance(tmp_path):
    c=ContinuityStore(tmp_path/'life.sqlite3')
    h=torch.randn(1,5); w=torch.randn(1,3)
    c.commit_episode_and_state(kind='episodic',provenance='lived',action=1,reward=.2,success=True,
                               content={'x':1},step=1,hidden=h,workspace=w,model_ref='brain.pt')
    state=c.runtime_state()
    assert state['step']==1
    assert torch.allclose(state['hidden'],h)
    assert c.counts()['lived']==1
    vec=c.recent_memory_vector(64)
    assert vec.shape==(64,) and np.isfinite(vec).all()
    c.close()


def test_import_semantic_preserves_lessons_not_weights(tmp_path):
    src=tmp_path/'semantic'; (src/'images').mkdir(parents=True); (src/'dino_cache').mkdir()
    from PIL import Image
    Image.fromarray(np.zeros((16,16,3),dtype=np.uint8)).save(src/'images'/'x.png')
    np.save(src/'dino_cache'/'sample-1.npy',np.ones(4,dtype=np.float32))
    row={'record_id':'sample-1','image':'images/x.png','context':[0.0]*48,'source':'test',
         'teacher':{'qas':[{'kind':'vision','question':'What?','answer':'Something.'}],
                    'care':{'respect_request':1.0}},'temporal':{'previous_skill':'navigate','previous_success':True}}
    (src/'records.jsonl').write_text(json.dumps(row)+'\n')
    ds=NurseryDataset(tmp_path/'nursery')
    manifest=ds.import_semantic(src,parent_id='vision.old')
    assert manifest['records']==1 and manifest['vision_parent_dim']==4
    lesson=ds.records()[0]
    assert lesson['qas'][0]['answer']=='Something.'
    assert not any((tmp_path/'nursery').glob('*.pt'))
