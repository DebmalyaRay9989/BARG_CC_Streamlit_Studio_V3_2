"""Presentation-only theme helpers. No Streamlit or network dependency."""
from html import escape
import numpy as np

PALETTES = {
    'Aurora · indigo / teal': dict(accent='#5145CD', accent2='#087F82', ink='#182440', muted='#52617B', bg='#F4F6FC', surface='#FFFFFF', border='#DCE2F0', soft='#EEEDFF', sidebar='#141B39', sidebar2='#263052', chart=['#5145CD','#087F82','#C46B13','#B43C78','#286CB1','#64748B','#7655AA','#477A42','#AB482E','#247D9B']),
    'Ocean · blue / cyan': dict(accent='#1765B5', accent2='#007E83', ink='#122D43', muted='#4C6479', bg='#F1F7FA', surface='#FFFFFF', border='#D5E4EC', soft='#E5F3FC', sidebar='#10263D', sidebar2='#183E56', chart=['#1765B5','#007E83','#B86415','#9D437B','#6555AC','#64748B','#2E8060','#A34F33','#2488A5','#8748A1']),
    'Graphite · plum / copper': dict(accent='#7045A0', accent2='#A55326', ink='#27243A', muted='#605B71', bg='#F7F5F9', surface='#FFFFFF', border='#E4DEEB', soft='#F1EAF8', sidebar='#242036', sidebar2='#41334F', chart=['#7045A0','#A55326','#087F82','#2A6BB0','#B73D76','#64748B','#55833A','#966025','#5368AA','#367F8A']),
}

# Stable outcome meanings across palette changes; do not encode outcomes solely by color.
OUTCOME_COLORS = {'certified_pct':'#5145CD','unchanged_pct':'#087F82','certification_all':'#5145CD','certification_dropped':'#B45309','certification_retained':'#286CB1','unchanged_all':'#087F82','silhouette':'#5145CD','baseline_silhouette':'#087F82'}

def css_variables(name):
    p=PALETTES[name]
    return ':root{' + ';'.join('--'+k+':'+v for k,v in p.items() if isinstance(v,str)) + '}'

def style_figure(fig, name, compact=False):
    p=PALETTES[name]
    fig.update_layout(template='plotly_white', paper_bgcolor=p['surface'], plot_bgcolor=p['surface'],
        font=dict(family='Inter, Segoe UI, Arial, sans-serif',size=13,color=p['ink']),
        title=dict(font=dict(size=18,color=p['ink']),x=.035,xanchor='left'),
        colorway=p['chart'], margin=dict(l=36,r=28,t=76 if fig.layout.title.text else 40,b=52),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='left',x=0,font=dict(size=11)),
        hoverlabel=dict(bgcolor=p['sidebar'],font=dict(color='#FFFFFF',size=13)),
        height=340 if compact else 420, autosize=True)
    fig.update_xaxes(gridcolor=p['border'],linecolor=p['border'],zerolinecolor=p['border'],automargin=True)
    fig.update_yaxes(gridcolor=p['border'],linecolor=p['border'],zerolinecolor=p['border'],automargin=True)
    # Plotly Express sets explicit trace colors before layout colorway is applied.
    series_colors = {}
    for i,t in enumerate(fig.data):
        if t.type in ('bar','scatter','scattergl','histogram','box','violin'):
            series_key=t.name or f'__trace_{i}'
            c=series_colors.setdefault(series_key, OUTCOME_COLORS.get(t.name,p['chart'][len(series_colors)%len(p['chart'])]))
            if hasattr(t,'marker'): t.marker.color=c
            if hasattr(t,'line'): t.line.color=c
            if t.type in ('scatter','scattergl'):
                if t.marker.size is None: t.marker.size=8
                if 'lines' in (t.mode or ''):
                    t.line.width=3
                    if t.name in ('unchanged_all','unchanged_pct','baseline_silhouette'):t.line.dash='dash'
            if t.type=='bar': t.marker.line.width=0
    if any(t.type=='heatmap' for t in fig.data):
        fig.update_layout(coloraxis=dict(colorscale=[[0,'#B45309'],[.5,'#F8FAFC'],[1,p['accent']]],cmid=0))
    return fig

def workflow_markup(imported, clustered, audited):
    rows=[('01','Import data',imported),('02','Fit clusters',clustered),('03','Audit stability',audited)]
    return '<div class="workflow-strip">'+''.join(f'<div class="workflow-step {"complete" if done else "pending"}"><span class="step-number">{num}</span><div><strong>{escape(label)}</strong><small>{"Complete" if done else "Pending"}</small></div></div>' for num,label,done in rows)+'</div>'

def loss_curve():
    q=np.linspace(0, .999, 250)
    return q,np.sqrt(2*(1-np.sqrt(1-q)))
