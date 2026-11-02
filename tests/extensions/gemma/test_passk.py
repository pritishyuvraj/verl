"""Pass@k denominators must exclude incomplete groups without hiding coverage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
import sys
import concurrent.futures
from unittest.mock import patch
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
