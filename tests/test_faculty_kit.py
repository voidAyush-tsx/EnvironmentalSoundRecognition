"""Cheap isolated tests for submission/readme tooling; never touch real projects."""
import sys
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import submission as fs
from src.submission import update_readme, START, END


class FacultyPackageTests(unittest.TestCase):
    REGISTRATION = '123456'
    NAME = 'Test Student'
    SECTION = 'B'
    CHECKPOINT = b'tiny synthetic selected checkpoint'
    CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT).hexdigest()

    def write_test_archive(self, path: Path, *, omit=(), replacements=None, extra=None):
        """Use only tiny placeholder members; never read a real project checkpoint."""
        entries = {name: b'placeholder\n' for name in fs.STANDARD}
        entries['README.md'] = fs.student_section(self.NAME, self.REGISTRATION, self.SECTION).encode('utf-8')
        entries['models/selected_block6_best.pt'] = self.CHECKPOINT
        entries['external/PANNs_PINNED_SOURCE.md'] = fs.COMMIT.encode('utf-8')
        evaluation = {'test_evaluated': True, 'config': 'block6', 'test_fold': 1,
                      'validation_fold': 2, 'test_samples': 873, 'checkpoint_epoch': 4,
                      'model_source_commit': fs.COMMIT, 'stage4_best_checkpoint_sha256': self.CHECKPOINT_SHA}
        entries['reports/faculty_evidence/heldout/evaluation_summary.json'] = json.dumps(evaluation).encode('utf-8')
        for method in ('head', 'fc1', 'block6'):
            base = f'reports/faculty_evidence/training_runs/{method}/'
            config = {'config': method, 'diagnostic_only': False, 'test_fold': 1,
                      'validation_fold': 2, 'model_source_commit': fs.COMMIT}
            entries[base + 'config.json'] = json.dumps(config).encode('utf-8')
            entries[base + 'summary.json'] = json.dumps({'best_epoch': 4, 'test_evaluated': False}).encode('utf-8')
            entries[base + 'history.csv'] = b'epoch\n4\n'
        entries.update(replacements or {})
        entries.update(extra or {})
        for name in omit:
            entries.pop(name)
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED) as archive:
            for name, contents in entries.items():
                archive.writestr(f'{self.REGISTRATION}/{name}', contents)
        return entries

    def inspect_test_archive(self, path: Path, *, name=None):
        with patch.object(fs, 'SELECTED_SHA256', self.CHECKPOINT_SHA):
            return fs.inspect_archive(path, self.REGISTRATION, name or self.NAME, self.SECTION)

    def test_inspect_archive_checks_tiny_archive_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f'{self.REGISTRATION}.zip'
            entries = self.write_test_archive(path)
            before = path.read_bytes()
            report = self.inspect_test_archive(path)
            self.assertEqual(report['integrity'], 'PASS')
            self.assertTrue(report['identity_verified'])
            self.assertEqual(report['count'], len(entries))
            self.assertEqual(report['checkpoint_sha256'], self.CHECKPOINT_SHA)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_make_archive_reuses_inspection_with_tiny_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = self.write_test_archive(Path(tmp) / 'fixture.zip')
            root = Path(tmp) / 'project'
            for relative, contents in entries.items():
                if relative in {'models/selected_block6_best.pt', 'external/PANNs_PINNED_SOURCE.md'}:
                    continue
                file = root / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(contents)
            checkpoint = root / 'tiny-selected.pt'
            checkpoint.write_bytes(self.CHECKPOINT)
            output = root / f'{self.REGISTRATION}.zip'
            with patch.object(fs, 'SELECTED_SHA256', self.CHECKPOINT_SHA), \
                 patch.object(fs, 'inspect_archive', wraps=fs.inspect_archive) as inspect:
                report = fs.make_archive(root, output, self.REGISTRATION, self.NAME, self.SECTION, checkpoint)
            inspect.assert_called_once_with(output, self.REGISTRATION, self.NAME, self.SECTION)
            self.assertEqual(report['integrity'], 'PASS')

    def test_inspect_archive_requires_cpu_and_common_dependency_files(self):
        for missing in ('requirements-cpu.txt', 'requirements-common.txt'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f'{self.REGISTRATION}.zip'
                self.write_test_archive(path, omit=(missing,))
                with self.assertRaisesRegex(ValueError, missing):
                    self.inspect_test_archive(path)

    def test_inspect_archive_rejects_wrong_identity_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f'{self.REGISTRATION}.zip'
            entries = self.write_test_archive(path)
            with self.assertRaisesRegex(ValueError, 'identity'):
                self.inspect_test_archive(path, name='Different Student')
            summary_name = 'reports/faculty_evidence/heldout/evaluation_summary.json'
            summary = json.loads(entries[summary_name])
            summary['model_source_commit'] = 'wrong-source'
            self.write_test_archive(path, replacements={summary_name: json.dumps(summary).encode('utf-8')})
            with self.assertRaisesRegex(ValueError, 'provenance'):
                self.inspect_test_archive(path)

    def test_inspect_archive_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f'{self.REGISTRATION}.zip'
            self.write_test_archive(path, replacements={'models/selected_block6_best.pt': b'other tiny weights'})
            with self.assertRaisesRegex(ValueError, 'CHECKPOINT SHA256 MISMATCH'):
                self.inspect_test_archive(path)

    def test_inspect_archive_rejects_raw_audio_and_caches(self):
        for unwanted in ('data/example.WAV', 'src/__pycache__/cached.pyc'):
            with self.subTest(unwanted=unwanted), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f'{self.REGISTRATION}.zip'
                self.write_test_archive(path, extra={unwanted: b'tiny unwanted placeholder'})
                with self.assertRaisesRegex(ValueError, 'raw audio or caches'):
                    self.inspect_test_archive(path)

    def test_inspect_archive_detects_corrupt_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f'{self.REGISTRATION}.zip'
            self.write_test_archive(path)
            contents = path.read_bytes()
            self.assertEqual(contents.count(self.CHECKPOINT), 1)
            path.write_bytes(contents.replace(self.CHECKPOINT, b'X' + self.CHECKPOINT[1:], 1))
            with self.assertRaisesRegex(ValueError, 'Invalid submission archive'):
                self.inspect_test_archive(path)

    def test_inspect_archive_reports_non_zip_as_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f'{self.REGISTRATION}.zip'
            path.write_bytes(b'not a ZIP')
            with self.assertRaisesRegex(ValueError, 'Invalid submission archive'):
                self.inspect_test_archive(path)

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
