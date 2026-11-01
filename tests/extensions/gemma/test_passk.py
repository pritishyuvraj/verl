"""Pass@k denominators must exclude incomplete groups without hiding coverage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
