import importlib.abc, importlib.util, pathlib, sys
sys.dont_write_bytecode=True
OVERRIDES={'continuous_brain.teacher': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/teacher.py', 'continuous_brain.growth_v110': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/growth_v110.py', 'continuous_brain.cognition': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/cognition.py', 'continuous_brain.model': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/model.py', 'continuous_brain.parented_train': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/parented_train.py', 'continuous_brain.whole_brain_v110': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/whole_brain_v110.py', 'continuous_brain.infant': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/infant.py', 'continuous_brain.lifelong.model': '/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/historical_core/continuous_brain/lifelong/model.py'}
class _HistoricalCoreFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        p=OVERRIDES.get(fullname)
        if p is None: return None
        return importlib.util.spec_from_file_location(fullname,p)
sys.meta_path.insert(0,_HistoricalCoreFinder())
root=pathlib.Path('/home/davei/projects/continuous_brain_v0.12.2').resolve()
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location("cb_g0_frozen_evaluator", '/home/davei/projects/continuous_brain_v0.12.2/evaluate_recurrent_self_trajectory_test_v1.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
vals=range(302000,303024)
setattr(m,'TEST_IDS', list(vals) if 'range'=="list" else range(302000,303024))
sys.argv=['/home/davei/projects/continuous_brain_v0.12.2/evaluate_recurrent_self_trajectory_test_v1.py',"--state-dir",'/home/davei/projects/continuous_brain_v0.38.1.dev3_g0_historical_core/Continuous_brain/runs/g0_scientific_v1/recurrent_self_trajectory_v031_g0-v031-dev/raw_evaluator']
m.main()
