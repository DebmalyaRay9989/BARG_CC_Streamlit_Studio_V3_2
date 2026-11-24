import ast
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import plotly.express as px
from studio_theme import PALETTES, OUTCOME_COLORS, css_variables, style_figure, workflow_markup, loss_curve

class StudioTests(unittest.TestCase):
    def test_palettes_and_charts(self):
        for name in PALETTES:
            self.assertIn('--accent:', css_variables(name))
            df=pd.DataFrame({'x':[1,2,3],'y':[2,1,3],'group':['a','a','b']})
            for fig in (px.line(df,x='x',y='y'),px.bar(df,x='x',y='y'),px.scatter(df,x='x',y='y',color='group'),px.imshow(np.eye(3)),px.histogram(df,x='x',color='group')):
                style_figure(fig,name).to_json()
                self.assertEqual(fig.layout.paper_bgcolor, PALETTES[name]['surface'])
    def test_outcome_colors(self):
        df=pd.DataFrame({'x':[0,1],'certified_pct':[100,20],'unchanged_pct':[100,98]})
        for name in PALETTES:
            fig=style_figure(px.line(df,x='x',y=['certified_pct','unchanged_pct']),name)
            self.assertEqual(fig.data[0].line.color,OUTCOME_COLORS['certified_pct'])
            self.assertEqual(fig.data[1].line.dash,'dash')
    def test_loss_curve(self):
        q,d=loss_curve(); self.assertEqual(d[0],0); self.assertTrue(np.all(np.diff(d)>0)); np.testing.assert_allclose(d*d,2*(1-np.sqrt(1-q)))
    def test_workflow(self):
        self.assertEqual(workflow_markup(True,True,False).count('step complete'),2)
        self.assertEqual(workflow_markup(False,False,False).count('step pending'),3)
    def test_chart_name_not_shadowed(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'streamlit_app.py').read_text())
        self.assertFalse(any(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store) and n.id=='chart' for n in ast.walk(tree)))
