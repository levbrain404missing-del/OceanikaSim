import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class HomeReportTests(unittest.TestCase):
    def run_fake(self, scenario, dep_ok=True):
        import check_home
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            dest = Path(folder) / "report.md"
            with patch.object(check_home,"required_cases",return_value=[("№ 1", (scenario.id(),))]):
                ok = check_home.check(dest,unittest.TestSuite([scenario]),dep_ok)
            return ok,dest.read_text(encoding="utf-8")

    def test_success_is_written(self):
        case = unittest.FunctionTestCase(lambda: None)
        ok, report = self.run_fake(case)
        self.assertTrue(ok)
        self.assertIn("1 из 1",report)

    def test_failure_and_dependency_problem_change_final_status(self):
        def bad():
            raise AssertionError("synthetic failure")
        for case,dep_ok in ((unittest.FunctionTestCase(bad),True),(unittest.FunctionTestCase(lambda:None),False)):
            ok, report = self.run_fake(case,dep_ok)
            self.assertFalse(ok)
            self.assertIn("Требует проверки",report)

    def test_skipped_case_is_not_a_pass(self):
        def skipped():
            raise unittest.SkipTest("fixture unavailable")
        ok, report = self.run_fake(unittest.FunctionTestCase(skipped))
        self.assertFalse(ok)
        self.assertIn("0 из 1",report)
