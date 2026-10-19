from pathlib import Path
from agent_core.models import Action, ActionKind, AgentPose, ObjectState, WorldState
from agent_core.grounding import SemanticGrounder, GroundingError
from agent_core.executor import Verifier
from agent_core.social import TextBus


def obj(oid, kind, **kw):
    return ObjectState(oid,kind,kind,True,{"x":0,"y":1,"z":1},1.0,**kw)

def world(objects, held=None):
    return WorldState('FloorPlan1',AgentPose({"x":0,"y":0.9,"z":0},{"x":0,"y":0,"z":0}),{o.object_id:o for o in objects},held,None,True,None,True)

def test_toggle_grounding_requires_toggleable():
    lamp=obj('Lamp|1','DeskLamp',toggleable=True,is_toggled=False)
    g=SemanticGrounder(); a=g.ground('ToggleObjectOn','lamp',{},world([lamp])).action
    assert a.kind=='ToggleObjectOn' and a.params['objectId']=='Lamp|1'

def test_focus_exposes_toggle_affordance():
    lamp=obj('Lamp|1','DeskLamp',toggleable=True,is_toggled=False)
    w=world([lamp]); g=SemanticGrounder(); assert g.try_focus('lamp',w)
    assert 'ToggleObjectOn' in g.focus_snapshot(w)['currently_possible_actions']

def test_drop_requires_held_object_and_never_needs_object_id():
    apple=obj('Apple|1','Apple',pickupable=True,is_picked_up=True)
    g=SemanticGrounder(); action=g.ground('DropObject',None,{},world([apple],held='Apple|1')).action
    assert action.kind==ActionKind.DROP.value and action.params=={}
    try:
        g.ground('DropObject',None,{},world([apple],held=None))
        assert False
    except GroundingError: pass

def test_verifier_toggle_and_drop():
    lamp0=obj('Lamp|1','DeskLamp',toggleable=True,is_toggled=False); lamp1=obj('Lamp|1','DeskLamp',toggleable=True,is_toggled=True)
    assert Verifier().verify(world([lamp0]),world([lamp1]),Action('ToggleObjectOn',{'objectId':'Lamp|1'}))[0]
    apple0=obj('Apple|1','Apple',pickupable=True,is_picked_up=True); apple1=obj('Apple|1','Apple',pickupable=True,is_picked_up=False)
    assert Verifier().verify(world([apple0],held='Apple|1'),world([apple1],held=None),Action('DropHandObject',{}))[0]

def test_text_bus_is_distance_independent_and_persistent(tmp_path):
    log=tmp_path/'messages.jsonl'; bus=TextBus({0:'A',1:'B'},log_path=log)
    event=bus.send(0,'B','Found something interesting.')
    assert event is not None
    inbox=bus.consume(1); assert inbox[0].for_llm()=={'from':'A','text':'Found something interesting.','channel':'text'}
    assert 'Found something interesting.' in log.read_text()

def test_text_bus_rejects_self_and_unknown_peer():
    bus=TextBus({0:'A',1:'B'})
    assert bus.send(0,'A','self') is None
    assert bus.send(0,'C','unknown') is None


class FakeClient:
    def __init__(self,response): self.response=response
    def chat(self,system,user,**kwargs): return self.response
    def chat_with_image(self,system,user,image_bytes,**kwargs): return self.response

def test_planner_textpeer_has_no_unity_action():
    from agent_core.roam import EmbodiedPlanner
    client=FakeClient('{"observation":"","intention":null,"say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"TextPeer","target":"B","params":{"text":"Meet me later."}}]}')
    planner=EmbodiedPlanner(client,SemanticGrounder(),social_enabled=True,agent_label='A')
    d=planner.decide(world([]),'autonomous',None,[],None,social_context={"peer_identity":[{"label":"B"}]})
    assert d.actions==[] and d.texted==[{"recipient":"B","text":"Meet me later."}]
    assert d.semantic_actions[0].kind=='TextPeer'

def test_planner_dropobject_becomes_native_drop_action():
    from agent_core.roam import EmbodiedPlanner
    apple=obj('Apple|1','Apple',pickupable=True,is_picked_up=True)
    client=FakeClient('{"observation":"","intention":null,"say":"","focus":null,"pace":"normal","task_complete":false,"actions":[{"kind":"DropObject","target":null,"params":{}}]}')
    planner=EmbodiedPlanner(client,SemanticGrounder(),allow_interactions=True)
    d=planner.decide(world([apple],held='Apple|1'),'autonomous',None,[],None)
    assert d.actions[0].kind=='DropHandObject'
    assert d.semantic_actions[0].kind=='DropObject'


def test_interaction_aliases_are_mechanical_only():
    from agent_core.roam import EmbodiedPlanner
    assert EmbodiedPlanner._normalize_kind('Grab') == 'PickupObject'
    assert EmbodiedPlanner._normalize_kind('Drop') == 'DropObject'
    assert EmbodiedPlanner._normalize_kind('TurnOn') == 'ToggleObjectOn'
    assert EmbodiedPlanner._normalize_kind('Text') == 'TextPeer'
