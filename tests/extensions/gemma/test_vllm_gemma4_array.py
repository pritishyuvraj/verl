"""CPU-only checks for the deployed Gemma4 parser progress guard."""
import ast
import importlib.metadata
import os
from pathlib import Path
import sys
import unittest


class GemmaArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get("GEMMA_PARSER_TEST_SOURCE")
        if source is None:
            try:
                source = importlib.metadata.distribution("vllm").locate_file(
                    "vllm/tool_parsers/gemma4_tool_parser.py")
            except importlib.metadata.PackageNotFoundError:
                raise unittest.SkipTest("vLLM parser source unavailable")
        tree = ast.parse(Path(source).read_text())
        names = {"_parse_gemma4_value", "_parse_gemma4_args", "_parse_gemma4_array"}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == 3
        cls.ns = {"STRING_DELIM": '<|"|>'}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), cls.ns)

    def test_valid_array_values(self):
        self.assertEqual(self.ns["_parse_gemma4_array"]("1,true,null,[2,3],{x:4}"),
                         [1, True, None, [2, 3], {"x": 4}])

    def test_quoted_bracket_is_preserved(self):
        self.assertEqual(self.ns["_parse_gemma4_array"]('<|"|>]<|"|>'), ["]"])

    def test_partial_bare_value_is_withheld(self):
        self.assertEqual(self.ns["_parse_gemma4_array"]("1,tru", partial=True), [1])

    def test_unexpected_closer_is_rejected(self):
        def bounded_parse(text, partial):
            events = 0
            def trace(frame, event, arg):
                nonlocal events
                if event == "line":
                    events += 1
                    if events > 2000:
                        raise RuntimeError("Parser did not terminate within the test budget")
                return trace
            previous = sys.gettrace()
            sys.settrace(trace)
            try:
                return self.ns["_parse_gemma4_array"](text, partial=partial)
            finally:
                sys.settrace(previous)
        for text in ("]", "1,]", "[1],]"):
            for partial in (False, True):
                with self.subTest(text=text, partial=partial):
                    with self.assertRaises(ValueError):
                        bounded_parse(text, partial)

    def test_full_tool_arguments_preserve_types(self):
        self.assertEqual(self.ns["_parse_gemma4_args"]('path:<|"|>x.py<|"|>,lines:[1,2]'),
                         {"path": "x.py", "lines": [1, 2]})


if __name__ == "__main__":
    unittest.main()
