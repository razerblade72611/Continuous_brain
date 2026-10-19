import json
from continuous_brain_unified.runtime import UnifiedRuntime
from continuous_brain_unified.checkpoint import save_shell_checkpoint,verify_shell_checkpoint

def test_shell_checkpoint_is_explicitly_shell_only(tmp_path):
    m={'schema_version':1,'source_snapshot':{},'components':[]}; mp=tmp_path/'m.json'; mp.write_text(json.dumps(m)); r=UnifiedRuntime(mp); r.boot(); p=save_shell_checkpoint(r,tmp_path/'s.json'); d=verify_shell_checkpoint(p); assert d['kind']=='v038_shell_checkpoint'; assert d['status']['scientific_integration_claimed'] is False
