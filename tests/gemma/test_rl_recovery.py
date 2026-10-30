import json
from pathlib import Path
import re
import tempfile
import unittest

from verl.extensions.gemma.rl_recovery import context_budget_error, grade_run_id, output_cap, recover_initial, sha


class RecoveryTest(unittest.TestCase):
    def test_official_output_cap_alias(self):
        self.assertEqual(output_cap({'max_completion_tokens':16384}),16384)
        self.assertEqual(output_cap({'max_tokens':16384}),16384)
        self.assertIsNone(output_cap({}))
        self.assertTrue(context_budget_error('maximum context length 32768 tokens, requested 32769'))
        self.assertFalse(context_budget_error('CUDA out of memory'))

    def test_actual_grade_id_schema(self):
        run = grade_run_id(Path('/episode/abc'), 'repo__issue-1')
        self.assertRegex(run, r'^v4-[a-f0-9]{12}-[a-f0-9]{12}-g1$')
        self.assertNotEqual(run, grade_run_id(Path('/episode/xyz'), 'repo__issue-1'))
        with self.assertRaises(ValueError):
            grade_run_id(Path('/episode/abc'), 'repo__issue-1', 4)

    def test_recovery_scoped_and_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'old'; source.mkdir(); out=root/'new'; out.mkdir()
            task={'instance_id':'a'};config=root/'config';config.write_text('fixed')
            adapter=root/'adapter';adapter.write_text('weights')
            files={'episode-request.json':{'task':task,'launch_owner':'owner'},
                   'request-000.json':{'max_completion_tokens':16384},
                   'response-000.json':{'ordinary':False,'response_ids':[1],'log_probs':[-.1],
                                        'extra_fields':{'min_global_steps':0,'max_global_steps':0}}}
            for name,value in files.items():(source/name).write_text(json.dumps(value))
            manifest={'config_sha256':sha(config),'inputs':{'adapter':sha(adapter)},'source_owner':'owner',
                      'tasks':{'a':{'path':'old','files':{name:sha(source/name)for name in files}}}}
            (root/'initial-validation-recovery.json').write_text(json.dumps(manifest))
            for validate,step in ((False,0),(True,1)):
                self.assertIsNone(recover_initial(root,task,out,validate=validate,step=step,config_path=config))
            turns=recover_initial(root,task,out,validate=True,step=0,config_path=config)
            self.assertTrue(turns[0]['ordinary'])
            self.assertFalse(json.loads((source/'response-000.json').read_text())['ordinary'])
            adapter.write_text('changed')
            with self.assertRaisesRegex(ValueError,'input changed'):
                recover_initial(root,task,out,validate=True,step=0,config_path=config)
            adapter.write_text('weights');(source/'request-000.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'generation changed'):
                recover_initial(root,task,out,validate=True,step=0,config_path=config)
