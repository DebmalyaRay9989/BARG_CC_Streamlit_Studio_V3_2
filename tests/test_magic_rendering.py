"""Offline regression checks for accidental Streamlit expression rendering."""
import ast
import hashlib
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
from engine import cluster_documents
from document_pipeline import _extract_zip
from advanced_audit import cluster_keywords

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'streamlit_app.py').read_text(encoding='utf-8'))


def get_renderer(name, mock):
    fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {'st': mock, 'pd': pd, 'hashlib': hashlib}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(ROOT/'streamlit_app.py'), 'exec'), ns)
    return ns[name]


class MagicRenderingTests(unittest.TestCase):
    def test_magic_disabled(self):
        with (ROOT/'.streamlit/config.toml').open('rb') as f:
            self.assertIs(tomllib.load(f)['runner']['magicEnabled'], False)

    def test_no_bare_conditional_rendering(self):
        bad = [n.lineno for n in ast.walk(TREE) if isinstance(n, ast.Expr) and isinstance(n.value, ast.IfExp)]
        self.assertEqual(bad, [], f'Bare conditional expressions at {bad}')

    def test_keywords_present_real_clustering(self):
        records, _ = _extract_zip(ROOT/'data/demo_mixed_documents.zip', ocr=False)
        res = cluster_documents(records, 2, 4, 17, 0, profile='Quick')
        res['keywords'] = cluster_keywords(res['records'], res['labels'])
        self.assertTrue(res['keywords'])
        mock = MagicMock()
        self.assertIsNone(get_renderer('render_cluster_summary', mock)(res['details'], res))
        self.assertEqual(mock.dataframe.call_count, 2)
        summary = mock.dataframe.call_args_list[0].args[0]
        self.assertEqual(summary.documents.sum(), len(res['details']))
        pd.testing.assert_frame_equal(mock.dataframe.call_args_list[1].args[0], pd.DataFrame(res['keywords']))
        mock.caption.assert_not_called()

    def test_keywords_empty_or_missing(self):
        for res in ({'keywords': []}, {}, {'keywords': None}):
            with self.subTest(res=res):
                mock = MagicMock()
                self.assertIsNone(get_renderer('render_cluster_summary', mock)(pd.DataFrame({'cluster':[0,1,0]}), res))
                mock.dataframe.assert_called_once()
                mock.caption.assert_called_once_with('Insufficient text for keyword extraction.')
                mock.write.assert_not_called()

    def test_integrity_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sample.txt').write_bytes(b'evidence')
            manifest = {'sample.txt':hashlib.sha256(b'evidence').hexdigest()}
            mock = MagicMock()
            self.assertIsNone(get_renderer('render_evidence_status', mock)(manifest, root))
            mock.success.assert_called_once_with('1 files verified.')
            mock.error.assert_not_called()
            mock.reset_mock()
            (root/'sample.txt').write_bytes(b'changed')
            get_renderer('render_evidence_status', mock)(manifest, root)
            mock.error.assert_called_once_with("Hash mismatch: ['sample.txt']")
            mock.success.assert_not_called()
