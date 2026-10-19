"""Small execution-contract tests; no neural checkpoint or third-party imports."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('continuation',Path(__file__).resolve().parents[1]/'diagnose_option_continuation_prediction_v1.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Env:
    def __init__(self,start=0):
        self.steps=start
        self.collisions=0
    def step(self,action):
        self.steps+=1
        reached=action==1 and self.steps>=6
        return None,-.004+2*reached,reached or self.steps==120,{'collision':False,'reached':reached}


class Runtime:
    def root(self,env):return env
    def stop(self,root,option,age):return option==0 and age>=3
    def select(self,root):return 1
    def action(self,root,option):return int(option==1)


class ContractTests(unittest.TestCase):
    def test_termination_then_shared_direct_control(self):
        b=m.continue_branch(Runtime(),Env(),0)
        self.assertEqual([e['option'] for e in b['events']],[0,0,0,1,1,1])
        self.assertEqual(b['initial_option_steps'],3)
        self.assertTrue(b['completed'])
    def test_no_silent_twelve_step_execution_cap(self):
        b=m.continue_branch(Runtime(),Env(80),5)
        self.assertEqual(b['initial_option_steps'],40)
        self.assertEqual(b['initial_option_end'],'episode_end')
        self.assertEqual(b['terminal_step'],120)
    def test_no_fresh_budget_from_boundary(self):
        b=m.continue_branch(Runtime(),Env(118),5)
        self.assertEqual(b['steps'],2)
        self.assertEqual(b['terminal_step'],120)
    def test_branch_does_not_mutate_start(self):
        env=Env(10)
        m.continue_branch(Runtime(),env,5)
        self.assertEqual(env.steps,10)
    def test_missing_evidence_rejected(self):
        with self.assertRaises(RuntimeError):m.summarize([])


if __name__=='__main__':unittest.main()
