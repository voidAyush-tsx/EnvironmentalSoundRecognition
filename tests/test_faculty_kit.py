"""Cheap isolated tests for submission/readme tooling; never touch real projects."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import faculty_submission as fs
from tools.update_readme import update_readme, START, END


class FacultyPackageTests(unittest.TestCase):
    def test_readme_sync_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'README.md').write_text('# Existing student notes\n', encoding='utf-8')
            update_readme(root)
            first = (root / 'README.md').read_text(encoding='utf-8')
            update_readme(root)
            second = (root / 'README.md').read_text(encoding='utf-8')
            self.assertEqual(first, second)
            self.assertIn('# Existing student notes', first)
            self.assertEqual(first.count(START), 1)
            self.assertEqual(first.count(END), 1)

    def test_fresh_project_cannot_pass_faculty_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors = fs.audit(Path(tmp))
            self.assertTrue(errors)
            self.assertTrue(any('README.md' in x for x in errors))

    def test_no_cache_or_dataset_wavs_in_archive_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'src' / '__pycache__').mkdir(parents=True)
            (root / 'src' / '__pycache__' / 'a.pyc').write_bytes(b'bytecode')
            (root / 'src' / 'dataset.py').write_text('print(1)')
            (root / 'data' / 'UrbanSound8K' / 'audio').mkdir(parents=True)
            (root / 'data' / 'UrbanSound8K' / 'audio' / 'secret.wav').write_bytes(b'wav')
            files = [p.relative_to(root).as_posix() for p in fs.collect_files(root)]
            self.assertIn('src/dataset.py', files)
            self.assertFalse(any('pycache' in x or x.endswith('.wav') for x in files))

    def test_student_metadata_present_in_generated_readme(self):
        result = fs.student_section('Test Student', '202612345', 'B')
        self.assertIn('Test Student', result)
        self.assertIn('202612345', result)
        self.assertIn('Section:** B', result)


if __name__ == '__main__':
    unittest.main()
