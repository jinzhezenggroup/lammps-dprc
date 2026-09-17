"""QUICK success exit status must not admit an unconverged label."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('check_quick_scf', Path(__file__).resolve().parents[1] / 'tools/check_quick_scf_convergence.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class QuickSCFConvergenceTest(unittest.TestCase):
    def test_success_requires_exact_count_and_no_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'quick.out'
            path.write_text('REACH CONVERGENCE AFTER 19 CYCLES\n' * 2)
            self.assertEqual(check.check_log(path, 2)['converged_frames'], 2)
            with self.assertRaisesRegex(ValueError, 'SCF count'):
                check.check_log(path, 3)
            path.write_text('REACH CONVERGENCE AFTER 19 CYCLES\nRAN OUT OF CYCLES. NO CONVERGENCE.\nNormal Termination\n')
            with self.assertRaisesRegex(ValueError, 'SCF failure'):
                check.check_log(path, 1)


if __name__ == '__main__':
    unittest.main()
