import json
from pathlib import Path
import re
import tempfile
import unittest

from verl.extensions.gemma.rl_recovery import (context_budget_error, grade_run_id, output_cap,
                                             recover_initial, recover_completed, CompletedClaims, sha)


class RecoveryTest(unittest.TestCase):
    def test_whole_group_claims_preserve_sessions_and_versions(self):
        groups = [{str(i): f'{g}-{i}' for i in range(8)} for g in range(2)]
        manifest = {'dev':{'devtask':'devrecord'}, 'train':{'task':groups}}
        claims = CompletedClaims()
        for group in range(2):
            for session in reversed(range(8)):
                self.assertEqual(claims.claim(manifest, task='task', uid=f'uid{group}', session=session,
                    validate=False, step=group), f'{group}-{session}')
        with self.assertRaisesRegex(ValueError, 'already recovered'):
            claims.claim(manifest, task='task', uid='uid0', session=0, validate=False, step=0)
        with self.assertRaisesRegex(ValueError, 'exhausted'):
            claims.claim(manifest, task='task', uid='uid2', session=0, validate=False, step=1)
        with self.assertRaisesRegex(ValueError, 'version lag'):
            claims.claim(manifest, task='task', uid='uid2', session=0, validate=False, step=2)
        self.assertEqual(claims.claim(manifest, task='devtask', uid='d', session=0, validate=True, step=0), 'devrecord')
        self.assertIsNone(claims.claim(manifest, task='devtask', uid='d', session=0, validate=True, step=1))

    def test_completed_recovery_retains_reward_and_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'episodes/train/old'; source.mkdir(parents=True)
            out=root/'new';out.mkdir();config=root/'config';config.write_text('fixed')
            task={'instance_id':'task'}
            files={'episode-request.json':{'task':task}, 'grade/reward.json':{'resolved':True},
                   'completed.json':{'launch_owner':'owner','task':'task','reward':1.,'seconds':12.,
                                     'uid':'oldgroup','session':3,'context_errors':[]},
                   'request-000.json':{'max_completion_tokens':16384},
                   'response-000.json':{'prompt_ids':[2,3],'response_ids':[4],'log_probs':[-.1],
                                        'extra_fields':{'min_global_steps':0,'max_global_steps':0}}}
            for name,value in files.items():
                path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            record={'path':'episodes/train/old','files':{name:sha(source/name) for name in files},'evidence':{}}
            manifest={'config_sha256':sha(config),'inputs':{},'source_owner':'owner'}
            (root/'completed-episode-recovery.json').write_text(json.dumps(manifest))
            turns,seconds,errors=recover_completed(root,task,out,record,config_path=config)
            self.assertEqual(seconds,12.)
            self.assertEqual(turns[0]['extra_fields']['max_global_steps'],0)
            self.assertTrue(json.loads((out/'grade/reward.json').read_text())['resolved'])
            self.assertFalse((out/'completed.json').exists())
            (source/'grade/reward.json').write_text('{"resolved":false}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                recover_completed(root,task,out,record,config_path=config)

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
