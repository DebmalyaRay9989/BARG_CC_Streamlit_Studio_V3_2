"""Offline API migration regressions; real Streamlit exercised by smoke_ui.py."""
import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from packaging.requirements import Requirement
from packaging.version import Version
import pandas as pd
import plotly.express as px
from studio_theme import PALETTES, style_figure

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT/'streamlit_app.py').read_text(encoding='utf-8'))

class ModernWidthTests(unittest.TestCase):
    def test_no_legacy_width_keywords_in_application(self):
        legacy = {'use_container_width', 'use_column_width'}
        for path in ROOT.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            offenders = [n.lineno for n in ast.walk(tree) if
                (isinstance(n, ast.keyword) and n.arg in legacy) or
                (isinstance(n, ast.arg) and n.arg in legacy)]
            self.assertEqual(offenders, [], str(path))

    def test_chart_helper_modern_width_and_config(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'chart')
        mock = MagicMock(); mock.session_state = {'studio_palette': next(iter(PALETTES))}
        ns = dict(st=mock, style_figure=style_figure, PALETTES=PALETTES)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'streamlit_app.py'),'exec'),ns)
        for kwargs, expected in [({},'stretch'),({'width':'content'},'content'),({'width':480},480),({'compact':True},'stretch')]:
            with self.subTest(kwargs=kwargs):
                fig = px.line(x=[0,1],y=[1,2]); mock.reset_mock()
                ns['chart'](fig, **kwargs)
                mock.plotly_chart.assert_called_once()
                args = mock.plotly_chart.call_args
                self.assertIs(args.args[0],fig)
                self.assertEqual(args.kwargs['width'],expected)
                self.assertIsNone(args.kwargs['theme'])
                self.assertFalse(args.kwargs['config']['displaylogo'])
                self.assertEqual(args.kwargs['config']['toImageButtonOptions']['scale'],2)

    def test_all_explicit_widths_valid(self):
        found = [n.value for n in ast.walk(TREE) if isinstance(n,ast.keyword) and n.arg=='width']
        self.assertEqual(len(found),21)
        for v in found:
            if isinstance(v,ast.Constant):
                self.assertIn(v.value, ('stretch','content'))
            else:
                self.assertIsInstance(v,ast.Name); self.assertEqual(v.id,'width')

    def test_keyword_tables_stretch(self):
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'render_cluster_summary')
        mock = MagicMock(); ns = dict(st=mock,pd=pd)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'streamlit_app.py'),'exec'),ns)
        ns['render_cluster_summary'](pd.DataFrame({'cluster':[0,0,1]}),{'keywords':[{'cluster':0,'keyword':'example'}]})
        self.assertEqual(mock.dataframe.call_count,2)
        for call in mock.dataframe.call_args_list:
            self.assertEqual(call.kwargs['width'],'stretch')
            self.assertTrue(call.kwargs['hide_index'])

    def test_dependency_version_floor(self):
        req = next(Requirement(line) for line in (ROOT/'requirements.txt').read_text().splitlines() if line.startswith('streamlit'))
        self.assertNotIn(Version('1.39.0'),req.specifier)
        self.assertNotIn(Version('1.50.0'),req.specifier)
        self.assertIn(Version('1.51.0'),req.specifier)
        self.assertNotIn(Version('2.0.0'),req.specifier)

if __name__=='__main__': unittest.main()
