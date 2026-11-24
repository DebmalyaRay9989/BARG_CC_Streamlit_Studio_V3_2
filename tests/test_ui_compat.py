"""Offline regression checks for the Streamlit inspector (no Streamlit import)."""
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
from engine import NAMES, cluster_documents
from document_pipeline import _extract_zip

ROOT = Path(__file__).resolve().parents[1]

class UICompatibilityTests(unittest.TestCase):
    def test_no_multi_argument_container_write(self):
        tree = ast.parse((ROOT/'streamlit_app.py').read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write' and len(node.args)>1:
                self.assertIn(ast.unparse(node.func.value), ('st','st.sidebar'), f'Unsafe write at line {node.lineno}')

    def test_document_inspector_with_real_clustering_result(self):
        tree = ast.parse((ROOT/'streamlit_app.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='render_document_inspector')
        mock = MagicMock(); mock.columns.return_value = (MagicMock(), MagicMock())
        records, _ = _extract_zip(ROOT/'data/demo_mixed_documents.zip', ocr=False)
        res = cluster_documents(records, 2, 4, 17, 0, profile='Quick')
        detail = res['details']; filtered = detail.iloc[1:3]; idx = filtered.index[-1]
        mock.selectbox.return_value = idx
        ns = {'st':mock,'pd':pd,'NAMES':NAMES}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'streamlit_app.py'),'exec'),ns)
        ns['render_document_inspector'](filtered,detail,res)
        mock.code.assert_called_once_with(str(res['records'][idx]['sha256']),language=None)
        mock.text.assert_called_once_with(str(res['records'][idx]['ocr_status']))
        mock.metric.assert_called_once_with('Cluster', int(detail.loc[idx,'cluster']))
        mock.text_area.assert_called_once_with('Extracted text',res['records'][idx]['text'][:10000],disabled=True)
        self.assertEqual(len(mock.bar_chart.call_args.args[0]),3)
        mock.write.assert_not_called()
