"""Pass@k denominators must exclude incomplete groups without hiding coverage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys
import concurrent.futures
from unittest.mock import Mock, patch
import time

PATH = Path(__file__).resolve().parents[3]/"verl/extensions/gemma/passk.py"
spec = importlib.util.spec_from_file_location("passk_under_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HistogramTests(unittest.TestCase):
    def test_zero_mixed_and_all_pass(self):
        groups = [[{"resolved": j < count} for j in range(8)] for count in (0,1,2,8)]
        result = module.histogram(groups)
        self.assertEqual(result["histogram"], [1,1,1,0,0,0,0,0,1])
        self.assertEqual(result["pass_at_8"], .75)
        self.assertEqual(result["mean_attempt_success"], 11/32)

    def test_infrastructure_is_not_zero(self):
        result = module.histogram([[{"resolved":False}]*7, [{"error":"missing tests"}]*8])
        self.assertEqual(result["incomplete_issues"],2)
        self.assertEqual(result["complete_issues"],0)
        self.assertIsNone(result["pass_at_8"])

    def test_integer_is_not_a_grading_boolean(self):
        self.assertEqual(module.histogram([[{"resolved":1}]*8])["incomplete_issues"],1)

    def test_atomic_result_is_readable(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"result.json"
            module.write(path,{"resolved":True})
            self.assertEqual(json.loads(path.read_text()),{"resolved":True})
            self.assertEqual(list(path.parent.glob("*.tmp")),[])


class ProgressTests(unittest.TestCase):
    def test_dead_server_prevents_dispatch_even_with_live_proxy(self):
        e = object.__new__(module.Experiment)
        e.services = [(Mock(pid=101, poll=Mock(return_value=1)), None),
                      (Mock(pid=102, poll=Mock(return_value=None)), None)]
        with self.assertRaisesRegex(RuntimeError, "PID 101 exited"):
            e.check_services()

    def test_healthy_services_and_initial_qualification_allowed(self):
        e = object.__new__(module.Experiment)
        for services in ([], [(Mock(poll=Mock(return_value=None)), None)]):
            e.services = services
            e.check_services()

    def test_dead_service_stops_wait_without_consuming_pending_issue(self):
        pending = [("task", {}, [concurrent.futures.Future()])]
        with self.assertRaisesRegex(RuntimeError, "service died"):
            module.next_completed_issue(pending, Mock(side_effect=RuntimeError("service died")))
        self.assertEqual(len(pending), 1)
        self.assertFalse(pending[0][2][0].cancelled())

    def test_observes_real_child_exit(self):
        e = object.__new__(module.Experiment)
        p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
        p.wait(timeout=5)
        e.services = [(p, None)]
        with self.assertRaisesRegex(RuntimeError, "exited \\(7\\)"):
            e.check_services()

    def test_slow_first_issue_does_not_block_completed_later_issue(self):
        slow, fast = concurrent.futures.Future(), concurrent.futures.Future()
        fast.set_result(None)
        pending = [("slow", {}, [slow]), ("fast", {}, [fast])]
        self.assertEqual(module.next_completed_issue(pending)[0], "fast")
        self.assertEqual(pending[0][0], "slow")
        self.assertFalse(slow.cancelled())

    def test_waits_for_all_attempts_of_any_issue(self):
        a, b = concurrent.futures.Future(), concurrent.futures.Future()
        a.set_result(None)
        pending = [("task", {}, [a, b])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(module.next_completed_issue, pending)
            self.assertFalse(waiting.done())
            b.set_result(None)
            self.assertEqual(waiting.result(timeout=2)[0], "task")
        self.assertEqual(pending, [])

    def test_process_disappears_during_proc_read(self):
        with patch.dict(module._process_cache, {"time": time.monotonic(), "rows": []}):
            with patch.object(Path, "read_text", side_effect=ProcessLookupError(3, "No such process")):
                self.assertEqual(module.private_snapshot(12345), {})
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                with self.assertRaises(PermissionError):
                    module.private_snapshot(12345)


class ImageReleaseTests(unittest.TestCase):
    def experiment(self, tmp):
        e = object.__new__(module.Experiment)
        e.out = Path(tmp)
        e.c = {"environment":{"release_owned_images":True,"pull_seconds":10}}
        e.client = Mock()
        e.client.images.get.return_value = Mock(id="owned-id")
        e.client.containers.list.return_value = []
        e.acquired_images = {}
        e.initial_images = set()
        e.capacity = Mock()
        e.bounded = Mock()
        e.run_grade = Mock(return_value={"resolved":False,"tests_executed":True})
        return e

    def test_preexisting_image_never_becomes_owned(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"docker":Mock()}):
            e = self.experiment(tmp)
            with self.assertRaises(module.QualificationRejected):
                e.qualify({"instance_id":"task","image_name":"image:latest","patch":"synthetic"})
            self.assertEqual(e.acquired_images, {})
            e.bounded.assert_not_called()

    def test_failed_control_retains_acquisition_for_cleanup(self):
        class NotFound(Exception): pass
        fake = Mock()
        fake.errors.ImageNotFound = NotFound
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"docker":fake}):
            e = self.experiment(tmp)
            e.client.images.get.side_effect = [NotFound(),Mock(id="owned-id")]
            with self.assertRaises(module.QualificationRejected):
                e.qualify({"instance_id":"task","image_name":"image:latest","patch":"synthetic"})
            self.assertEqual(e.acquired_images['task']['image_id'], 'owned-id')
            self.assertTrue((e.out/'qualification/task/acquired-image.json').exists())

    def test_release_refuses_active_containers_and_changed_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = self.experiment(tmp)
            e.acquired_images['task'] = {'image':'image:latest','image_id':'owned-id'}
            e.client.containers.list.return_value = [Mock()]
            with self.assertRaises(RuntimeError): e.release_image('task')
            e.client.images.remove.assert_not_called()
            e.client.containers.list.return_value = []
            e.client.images.get.return_value = Mock(id='changed')
            with self.assertRaises(RuntimeError): e.release_image('task')
            e.client.images.remove.assert_not_called()
            e.client.images.get.return_value = Mock(id='owned-id')
            e.release_image('task')
            e.client.images.remove.assert_called_once_with('image:latest',force=False,noprune=True)
            self.assertEqual(e.acquired_images, {})


class PublishedRequirementsTests(unittest.TestCase):
    def test_null_path_uses_exact_snapshot_and_restores_upstream(self):
        upstream = Mock()
        original = upstream.get_requirements
        row = {"install_config":{"packages":"requirements.txt","reqs_path":None},
               "requirements":"pytest==8.3.5\nexample==1.0\n"}
        def build(value, namespace):
            self.assertIs(value, row)
            self.assertEqual(namespace, "swerebench")
            self.assertEqual(upstream.get_requirements(value), row['requirements'])
            return 'spec'
        spec, receipt = module.published_test_spec(row, build, upstream)
        self.assertEqual(spec, 'spec')
        self.assertIs(upstream.get_requirements, original)
        self.assertTrue(receipt['uses_existing_image'])
        with self.assertRaises(ValueError):
            module.published_test_spec(row, Mock(side_effect=ValueError('build failed')), upstream)
        self.assertIs(upstream.get_requirements, original)

    def test_explicit_paths_keep_upstream_behavior(self):
        upstream, build = Mock(), Mock(return_value='spec')
        original = upstream.get_requirements
        row = {'install_config':{'packages':'requirements.txt','reqs_path':['requirements.txt']}}
        self.assertEqual(module.published_test_spec(row, build, upstream), ('spec',None))
        self.assertIs(upstream.get_requirements, original)

    def test_missing_snapshot_is_not_empty_dependencies(self):
        row = {'install_config':{'packages':'requirements.txt','reqs_path':None}}
        for value in [None, '', '  ', []]:
            with self.assertRaises(RuntimeError):
                module.published_test_spec(row | {'requirements':value}, Mock(), Mock())


@unittest.skipUnless(sys.platform == "linux", "Linux process identity qualification")
class RetentionTests(unittest.TestCase):
    def helper(self):
        sys.path.insert(0,"/pritish/workspace/gemma-post-training-kaggle/scripts")
        from owned_process_group_v4 import RetainedGroup
        return module.live_retention(RetainedGroup)

    def test_parallel_real_cleanup(self):
        cls=self.helper()
        def one(index):
            with tempfile.TemporaryDirectory() as tmp:
                p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True)
                holder={}
                def snapshot(deadline=None):
                    keys=holder["g"].record["known_processes"] if "g" in holder else ()
                    return module.private_snapshot(p.pid,keys,deadline=deadline)
                g=cls(p,Path(tmp)/"receipt.json",snapshot=snapshot)
                holder["g"]=g
                try:
                    self.assertEqual(snapshot()[p.pid]["sid"],p.pid)
                finally:
                    g.cleanup(max_seconds=15)
                self.assertTrue(g.record["owned_processes_stopped"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(one,range(8)))

    def test_retire_dead_history_but_reject_reused_identity(self):
        cls=self.helper()
        with tempfile.TemporaryDirectory() as tmp:
            p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],start_new_session=True)
            g=cls(p,Path(tmp)/"receipt.json",monitor=False,snapshot=lambda deadline=None:module.private_snapshot(p.pid,deadline=deadline))
            try:
                original=dict(g.record["known_processes"][str(p.pid)])
                # These identities cannot be current Linux PIDs; observed ESRCH retires them.
                for i in range(130):
                    dead={**original,"pid":2**31+i}
                    g.record["known_processes"][str(dead["pid"])]=dead
                g._remember(module.private_snapshot(p.pid))
                self.assertEqual(len(g.record["retired_processes"]),130)
                self.assertEqual(len(g.record["known_processes"]),1)
                changed={p.pid:{**original,"start_ticks":"different","state":"Z"}}
                with self.assertRaises(RuntimeError):g._remember(changed)
            finally:
                g.cleanup(max_seconds=15)


if __name__ == "__main__":
    unittest.main()
