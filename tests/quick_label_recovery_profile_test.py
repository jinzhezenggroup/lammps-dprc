"""Keep the published recovery profile bound to its reviewed source patch."""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QuickLabelRecoveryProfileTest(unittest.TestCase):
    def test_retained_artifacts_match_qualified_profile(self):
        manifest = json.loads((ROOT / 'config/quick_pbe0_label_recovery.json').read_text())
        for name in ('patch', 'input'):
            record = manifest[name]
            self.assertEqual(hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest(),
                             record['sha256'], msg=f'Stale recovery {name} hash')
        text = (ROOT / manifest['input']['path']).read_text()
        profile = manifest['profile']
        for name in ('scf_cyc', 'ncyc', 'denserms', 'intcutoff', 'xccutoff', 'gradcutoff', 'basiscutoff'):
            match = re.search(rf'^\s*{name}\s*=\s*([^,]+),', text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            self.assertEqual(float(match[1].lower().replace('d', 'e')), profile[name])
        self.assertGreater(profile['ncyc'], profile['scf_cyc'])
        self.assertEqual(profile['denserms'], 1e-8)
        self.assertEqual(manifest['qualification']['force_tolerance_kcal_mol_A'], .003)
        self.assertTrue((ROOT / manifest['source']['license_file']).is_file())


if __name__ == '__main__':
    unittest.main()
