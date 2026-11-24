"""BARG-CC / IJSS research dashboard, adapted from HAMVCC Premium UI V3."""
from pathlib import Path
import io, json, hashlib, platform, sys, zipfile, traceback
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from engine import (fuse, coverage, digits_demo, exact_demo, coordinate_demo, cluster_documents, view_loss_audit, fixed_mask_demo, prepare_documents, NAMES)
from hamv_gate import masked_gate, strength_for_floor
from document_pipeline import _extract_zip
from advanced_audit import cluster_keywords, stability_audit
from threadpoolctl import threadpool_limits
from studio_theme import PALETTES, css_variables, style_figure, workflow_markup, loss_curve

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'reference'/'results'
st.set_page_config(page_title='BARG-CC | Reliability Studio',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
# Native controls remain light for consistent accessibility; Studio palettes customize the shell and figures.
def chart(fig, width="stretch", compact=False):
    style_figure(fig, st.session_state.get('studio_palette', next(iter(PALETTES))), compact=compact)
    st.plotly_chart(fig, width=width, theme=None,
                    config={'displaylogo':False,'responsive':True,'scrollZoom':False,
                            'toImageButtonOptions':{'format':'png','filename':'bargcc_chart','scale':2}})


def header(kicker,title,desc):
    st.markdown(f'<div class="workspace-intro"><div class="section-kicker">{kicker}</div><div class="page-heading">{title}</div><p class="page-description">{desc}</p></div>',unsafe_allow_html=True)
def note(title,body):
    st.markdown(f'<div class="research-note"><strong>{title}</strong><span>{body}</span></div>',unsafe_allow_html=True)
def download(label,df,name):
    # Prevent spreadsheet-formula injection from user-controlled text columns.
    clean=df.copy()
    for c in clean.select_dtypes(include=['object','string']).columns:
        clean[c]=clean[c].map(lambda x: "'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
    st.download_button(label,clean.to_csv(index=False).encode('utf-8-sig'),name,'text/csv')
def json_dl(label,obj,name):
    st.download_button(label,json.dumps(obj,indent=2,default=str).encode(),name,'application/json')
def safe(fn):
    try: return fn()
    except Exception as e:
        st.error(f'{type(e).__name__}: {e}')
        with st.expander('Technical details'): st.code(traceback.format_exc())
        return None

def render_cluster_summary(detail, res):
    """Render explicit elements; never expose a widget return value to magic."""
    st.subheader("Cluster summary and keywords")
    summary = detail.groupby("cluster").size().rename("documents").reset_index()
    st.dataframe(summary, width="stretch", hide_index=True)
    keywords = res.get("keywords", [])
    if keywords:
        st.dataframe(pd.DataFrame(keywords), width="stretch", hide_index=True)
    else:
        st.caption("Insufficient text for keyword extraction.")


def render_evidence_status(manifest, root):
    failures = [path for path, digest in manifest.items()
                if not (root / path).exists()
                or hashlib.sha256((root / path).read_bytes()).hexdigest() != digest]
    if not failures:
        st.success(f"{len(manifest)} files verified.")
    else:
        st.error(f"Hash mismatch: {failures}")


def render_document_inspector(filtered, detail, res):
    """Use individual UI elements; never multi-argument DeltaGenerator.write."""
    idx = st.selectbox('Individual document inspector', filtered.index.tolist(),
                       format_func=lambda i: detail.loc[i, 'filename'])
    rec = res['records'][idx]
    a, b = st.columns([1, 2])
    with a:
        st.metric('Cluster', int(detail.loc[idx, 'cluster']))
        st.markdown('**SHA-256**')
        st.code(str(rec['sha256']), language=None)
        st.markdown('**OCR status**')
        st.text(str(rec['ocr_status']))
    with b:
        st.text_area('Extracted text', rec['text'][:10000], disabled=True)
        st.bar_chart(pd.DataFrame({'weight': [detail.loc[idx, n.lower()+'_weight']
                                              for n in NAMES]}, index=NAMES))

def show_coverage(df,prefix):
    cov=coverage(df); a,b,c=st.columns(3)
    a.metric('All-record coverage',f'{100*df.certified.mean():.1f}%')
    affected=df[df.dropped]; b.metric('Affected-row coverage',f'{100*affected.certified.mean():.1f}%' if len(affected) else 'N/A')
    c.metric('Observed unchanged',f'{100*df.unchanged.mean():.1f}%')
    chart(px.bar(cov.melt(id_vars=['group'],value_vars=['certified_pct','unchanged_pct']),x='group',y='value',color='variable',barmode='group',labels={'value':'Share (%)','variable':'Outcome'}),width="stretch")
    st.dataframe(cov,width="stretch",hide_index=True)
    st.caption('A non-certificate is inconclusive, not necessarily an assignment change. The one-refit result is not global optimality or semantic accuracy.')
    with st.expander('Per-record audit'): st.dataframe(df,width="stretch")
    download('Export pointwise audit',df,prefix+'_pointwise.csv')

def gate_controls(key):
    rho=st.slider('Minimum participation floor ρ',.01,.33,.10,.01,key=key+'_floor')
    cap=strength_for_floor(rho,3)
    mode=st.radio('Fusion policy',['Availability-only (recommended)','Bounded reliability gate'],key=key+'_policy',horizontal=True)
    kappa=0. if mode.startswith('Availability') else cap*st.slider('Fraction of permissible gate strength',0.,1.,1.,.05,key=key+'_fraction')
    st.caption(f'κ = {kappa:.6f} · floor-constrained maximum = {cap:.6f} for three views')
    return rho,kappa

st.sidebar.markdown('<div class="sidebar-brand"><span class="brand-mark">B</span><div><strong>BARG-CC</strong><small>RELIABILITY STUDIO</small></div></div>',unsafe_allow_html=True)
st.sidebar.caption('Studio V3.2 · IJSS research edition · 2026')
page=st.sidebar.radio('WORKSPACES',['01 · Results','02 · Bounded gate lab','03 · Document features','04 · Diagnostics','05 · Archive upload & clustering','06 · View-loss certification','07 · Research & provenance'])
palette=st.sidebar.selectbox('Appearance', list(PALETTES), key='studio_palette', help='Changes the interface and chart palette, not the computation.')
st.markdown('<style>'+css_variables(palette)+(ROOT/'assets'/'streamlit_theme.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
st.sidebar.caption('Local-first analysis · change appearance without refitting clusters')
st.sidebar.divider(); st.sidebar.markdown('**RESEARCH SCOPE**'); st.sidebar.caption('Bounded availability and reliability gating · separated feature blocks · conditional assurance')
st.sidebar.info('Availability-only is the operational default. Certification does not establish semantic accuracy or deployment safety.')

if page.startswith('01'):
    header('01 / BENCHMARKS','Manuscript benchmark results','Inspect archived IJSS experimental records without mixing them with new application runs.')
    cov=pd.read_csv(RESULTS/'conditional_coverage_audit'/'conditional_coverage.csv'); row=cov[np.isclose(cov.dropout_probability,.1)].iloc[0]
    a,b,c,d=st.columns(4);a.metric('Digits records',int(row.n));b.metric('10% loss: certified',f'{100*row.certification_all:.1f}%');c.metric('Affected: certified',f'{100*row.certification_dropped:.1f}%');d.metric('Affected records',int(row.n_dropped))
    note('Evidence boundary','Archived transductive mechanism audit on derived Digits views; no general neural-superiority claim. The affected-row result is intentionally shown alongside overall coverage.')
    families={ 'Corrected family-balanced Digits':'inductive_digit_balanced/summary.csv','Competitive Digits':'competitive_digit/summary.csv','Observational first-run':'observational_external_first_run/summary.csv','Matched classical comparators':'matched_library_comparators/summary.csv','Neural ablations':'neural_probe/summary.csv','Denoising neural':'denoising_neural/summary.csv','Factorial mechanism':'factorial_mechanism/summary.csv','Strong local baselines':'strong_local_baselines/summary.csv','View-loss / all records':'mask_switch_audit/summary.csv','Event-conditioned coverage':'conditional_coverage_audit/conditional_coverage.csv','Exact global view-loss':'mask_switch_audit/global_exact_summary.csv','Global score/embedding drift':'global_kmeans_certificate/summary.csv'}
    sel=st.selectbox('Evidence family',list(families));path=RESULTS/families[sel]; df=pd.read_csv(path); st.caption('Archived source: reference/results/'+families[sel])
    if 'dataset' in df: df=df[df.dataset.isin(st.multiselect('Datasets',list(df.dataset.unique()),default=list(df.dataset.unique())))]
    if 'condition' in df: df=df[df.condition.isin(st.multiselect('Conditions',list(df.condition.unique()),default=list(df.condition.unique())))]
    if 'method' in df: df=df[df.method.isin(st.multiselect('Methods',list(df.method.unique()),default=list(df.method.unique()))) ]
    numeric=[c for c in df.select_dtypes(include='number').columns]; pref=next((c for c in ['ari','ARI_mean','mean_ARI','certified_mean','certification_all'] if c in numeric),numeric[0] if numeric else None)
    if numeric and len(df):
        metric=st.selectbox('Recorded metric',numeric,index=numeric.index(pref)); x=next((c for c in ['condition','dropout_probability','scenario','method'] if c in df),None)
        if x:
            if x=='dropout_probability': fig=px.line(df,x=x,y=metric,markers=True)
            else: fig=px.bar(df,x=x,y=metric,color='method' if 'method' in df else None,facet_col='dataset' if 'dataset' in df and df.dataset.nunique()<=4 else None,barmode='group')
            chart(fig,width="stretch")
    st.dataframe(df,width="stretch",hide_index=True);download('Export filtered archive table',df,'bargcc_archived_summary.csv')
    with st.expander('All archived CSV and JSON artifacts'):
        paths=sorted(p.relative_to(RESULTS).as_posix() for p in RESULTS.rglob('*') if p.suffix in ('.csv','.json'))
        rel=st.selectbox('Artifact',paths); p=RESULTS/rel
        if p.suffix=='.csv': st.dataframe(pd.read_csv(p).head(100),width="stretch")
        else: st.code(p.read_text(encoding='utf-8')[:12000],language='json')
        st.download_button('Download original artifact',p.read_bytes(),p.name)

elif page.startswith('02'):
    header('02 / INTERACTIVE METHODS','Bounded reliability gate','Inspect exact masking, minimum participation, score sensitivity and view-removal geometry.')
    cols=st.columns(3);r=[];m=[]
    for col,name in zip(cols,NAMES):
        with col:
            m.append(st.checkbox(name+' available',True));r.append(st.slider(name+' reliability',0.,1.,.7,.01))
    rho,kappa=gate_controls('lab'); n=sum(m)
    if not n: st.error('All views missing: unassignable. No silent repair.')
    else:
        w=masked_gate(np.array([r]),np.array([m]),kappa)[0]; floor=1/(1+(n-1)*np.exp(2*kappa));jac=2*kappa*(np.diag(w)-np.outer(w,w))
        a,b,c=st.columns(3);a.metric('κ',f'{kappa:.4f}');b.metric('Guaranteed lower share',f'{floor:.4f}');c.metric('Actual minimum share',f'{w[np.array(m)].min():.4f}')
        chart(px.bar(x=NAMES,y=w,labels={'x':'View','y':'Weight'},title='Masked softmax weights · Eq. (2)'),width="stretch")
        chart(px.imshow(jac,x=NAMES,y=NAMES,aspect='auto',title='Gate Jacobian · Eq. (4)'),width="stretch")
        st.latex(r'\alpha_m=\frac{a_m\exp(\kappa(2r_m-1)-c)}{\sum_j a_j\exp(\kappa(2r_j-1)-c)},\quad z=[\sqrt{\alpha_m}e_m]_m')
        delta=st.slider('Fixed-mask score-error budget δ',0.,1.,.1,.01); st.write(f'Unit-norm row-normalized drift bound: {min(kappa*delta,np.sqrt(2)):.6f} (Proposition 4).')
        available=[j for j,ok in enumerate(m) if ok];drop=st.multiselect('Remove available views (retain at least one)',available,format_func=lambda j:NAMES[j]);
        if len(drop)==n: st.warning('Removing every view is not permitted.')
        elif drop:
            q=w[drop].sum(); st.metric('Exact unit-block drift · Eq. (22)',f'{np.sqrt(2*(1-np.sqrt(1-q))):.6f}');st.caption(f'Removed mass q = {q:.6f}; participation-floor lower bound = {np.sqrt(2*(1-np.sqrt(1-len(drop)*rho))):.6f}. Retained scores and unit embeddings must remain frozen.')
        with st.expander('Interactive participation–view-loss curve',expanded=True):
            qgrid,drift=loss_curve()
            import plotly.graph_objects as go
            fig=go.Figure(go.Scatter(x=qgrid,y=drift,mode='lines',name='Exact drift'))
            if drop and len(drop)<n:
                fig.add_scatter(x=[float(w[drop].sum())],y=[float(np.sqrt(2*(1-np.sqrt(1-w[drop].sum()))))],mode='markers',name='Selected view loss',marker={'size':13})
            fig.update_layout(title='Removed reliability mass vs. representation drift',xaxis_title='Removed mass q',yaxis_title='Unit-block drift')
            chart(fig,compact=True)
            st.caption('Equation (22) assumes frozen retained logits and unit embeddings; it is not an accuracy curve.')
        download('Export gate weights',pd.DataFrame({'view':NAMES,'available':m,'score':r,'weight':w}),'bargcc_gate.csv')

elif page.startswith('03'):
    header('03 / DOCUMENT ANALYSIS','Document-feature laboratory','Explore public Digits-derived feature groups or previously imported document features.')
    source=st.radio('Feature source',['Public Digits (derived views)','Imported archive'],horizontal=True)
    if source.startswith('Public'):
        from sklearn.datasets import load_digits
        from mask_switch_certificate import views_from_digits
        x,y=load_digits(return_X_y=True); views=views_from_digits(x);names=['Row/column sums','Pixels','Gradients'];df=pd.DataFrame({'record':np.arange(len(x)),'label_evaluation_only':y})
        for name,v in zip(names,views): df[name+' dimension']=v.shape[1];df[name+' norm']=np.linalg.norm(v,axis=1)
        st.dataframe(df.head(100),width="stretch");chart(px.histogram(df,x='Gradients norm',color=df.label_evaluation_only.astype(str),title='Gradient feature distribution'),width="stretch")
        st.caption('All three are derived from the same image, not independently acquired modalities. Labels shown for inspection only.')
        download('Export Digits feature summary',df,'bargcc_digits_features.csv')
    else:
        records=st.session_state.get('records',[])
        if not records: st.info('Import documents in workspace 05 first.')
        else:
            rows=[{'filename':r['filename'],'image_extracted':r['image_features'] is not None,'text_characters':len(r['text']),'ocr_status':r['ocr_status'],'size_bytes':r['size_bytes']} for r in records];df=pd.DataFrame(rows);st.dataframe(df,width="stretch");download('Export feature inventory',df,'bargcc_feature_inventory.csv')
            rec=st.selectbox('Inspect raw feature record',range(len(records)),format_func=lambda i:records[i]['filename']);st.json({k:v for k,v in records[rec].items() if k!='text'});st.text_area('Extracted text',records[rec]['text'][:10000],disabled=True)

elif page.startswith('04'):
    header('04 / ROBUSTNESS','Geometric and global diagnostics','Reproduce coordinate invariance and exact small-sample partition assurance.')
    seed=st.number_input('Simulation seed',0,1000000,42);rho,kappa=gate_controls('diag')
    if st.button('Run coordinate robustness simulation'):
        st.session_state['geometry']=safe(lambda:coordinate_demo(int(seed),kappa))
    if st.session_state.get('geometry') is not None: st.json(st.session_state['geometry']);st.caption('Proposition 6: independent orthogonal rotations preserve separated-block distances with fixed weights.')
    st.divider();st.subheader('Exact global K-means certificate · Proposition 11')
    scenario=st.radio('Synthetic separation',['Separated','Overlapping'],horizontal=True);p=st.slider('Probability of single-view loss',0.,1.,.1,.01)
    if st.button('Enumerate all 127 partitions',type='primary'):
        st.session_state['global']=safe(lambda:exact_demo(int(seed),scenario,p,kappa))
    result=st.session_state.get('global')
    if result is not None:
        st.json(result);json_dl('Export exact global audit',result,'bargcc_global_audit.json')
    st.warning('Eight records, two clusters only. Exhaustive global certification is exponential; this does not certify arbitrary Lloyd iterations.')
    st.divider();st.subheader('Fixed-mask compositional certificate · Proposition 10')
    st.caption('Separate synthetic experiment using unnormalized blocks and externally specified perturbation budgets; not learned or validated field budgets.')
    a,b,c=st.columns(3);delta=a.slider('Score error δ',0.,.5,.05,.01);eta=b.slider('Embedding drift ε',0.,1.,.05,.01);zeta=c.slider('Centre drift ζ',0.,1.,.05,.01)
    if st.button('Run fixed-mask perturbation audit'):
        st.session_state['fixed_mask']=safe(lambda:fixed_mask_demo(int(seed),kappa,delta,eta,zeta))
    if st.session_state.get('fixed_mask') is not None:
        fd=st.session_state['fixed_mask'];a,b=st.columns(2);a.metric('Certified',f'{fd.certified.mean():.1%}');b.metric('Observed unchanged',f'{fd.unchanged.mean():.1%}');st.dataframe(fd,width="stretch");download('Export fixed-mask audit',fd,'bargcc_fixed_mask.csv')


elif page.startswith('05'):
    header('05 / DOCUMENT WORKSPACE','Upload, select K, and cluster documents','Import mixed document archives, inspect feature extraction, compare policies and review every assignment.')
    note('Exploratory document processing','Uses lightweight structural/image features and TF-IDF, not a trained semantic encoder. PDF extraction reads text only; optional local Tesseract OCR handles images.')
    workflow=st.empty()
    def refresh_workflow():
        workflow.markdown(workflow_markup(bool(st.session_state.get('records')),st.session_state.get('cluster_result') is not None,st.session_state.get('doc_audit') is not None),unsafe_allow_html=True)
    refresh_workflow()
    z=st.file_uploader('Upload ZIP archive',type=['zip']);individual=st.file_uploader('Or select individual documents',type=['png','jpg','jpeg','tif','tiff','bmp','webp','txt','md','pdf'],accept_multiple_files=True)
    st.caption('Limits: 500 MB compressed, 180 MB expanded, 1,500 files, 12 MB per document, 32 megapixels per image. No files extracted to disk.')
    ocr=st.checkbox('Enable image OCR (slower; requires local Tesseract)',value=False,help='Without OCR, images still contribute structural and visual views; text files and searchable PDFs still supply text. This changes available views, not just runtime.')
    demo=st.button('Load bundled mixed-document demo archive')
    import_clicked=st.button('Import and inspect archive',type='primary',disabled=not(z or individual))
    if demo or import_clicked:
        def ingest():
            if demo: data=(ROOT/'data'/'demo_mixed_documents.zip').read_bytes()
            elif z: data=z.getvalue()
            else:
                if len(individual)>1500 or sum(f.size for f in individual)>180_000_000 or any(f.size>12_000_000 for f in individual): raise ValueError('Individual upload limits exceeded.')
                buf=io.BytesIO()
                with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as out:
                    for i,f in enumerate(individual):out.writestr(f'{i:04d}/{Path(f.name).name}',f.getvalue())
                data=buf.getvalue()
            if len(data)>500_000_000: raise ValueError('ZIP exceeds 500 MB.')
            digest=hashlib.sha256(data).hexdigest()
            if st.session_state.get('import_key')==(digest,ocr):
                st.info('Identical archive and OCR setting already loaded. Reusing session records.');return
            bar=st.progress(0.,text='Reading archive')
            try: rec,skipped=_extract_zip(io.BytesIO(data),ocr=ocr,progress=lambda i,n,t:bar.progress(i/max(n,1),text=t))
            finally: bar.empty()
            st.session_state.update(records=rec,skipped=skipped,archive_sha256=digest,import_key=(digest,ocr),ocr_enabled=ocr)
            for key in ('cluster_result','doc_audit','seed_audit','prepared_doc'):st.session_state.pop(key,None)
        safe(ingest)
        refresh_workflow()
    if 'records' in st.session_state and st.button('Clear imported data and cached features'):
        for key in ('records','skipped','archive_sha256','import_key','ocr_enabled','cluster_result','doc_audit','seed_audit','prepared_doc'):st.session_state.pop(key,None)
        st.rerun()
    if 'records' in st.session_state:
        records=st.session_state['records'];skipped=st.session_state['skipped'];a,b,c=st.columns(3);a.metric('Readable unique files',len(records));b.metric('Image features',sum(r['image_features'] is not None for r in records));c.metric('Text extracted',sum(bool(r['text'].strip()) for r in records))
        if skipped:
            with st.expander('Skipped files / duplicate audit'): st.dataframe(pd.DataFrame(skipped));download('Export skipped files',pd.DataFrame(skipped),'bargcc_skipped.csv')
        if len(records)>=3:
            st.subheader('Clustering protocol');rho,kappa=gate_controls('doc'); mode=st.radio('Block representation',['Row-normalized · Eq. (22) compatible','Unnormalized · Eq. (7)'],horizontal=True);norm=mode.startswith('Row')
            kmax=min(12,len(records)-1); kmin=st.number_input('Minimum K',2,kmax,2); kmax=st.number_input('Maximum K',int(kmin),kmax,min(kmax,max(4,int(kmin))));seed=st.number_input('Clustering seed',0,1000000,17)
            profile=st.radio('Computation profile',['Research','Quick'],horizontal=True,help='Research: 10 restarts and full silhouette. Quick: 3 restarts and up to 600 sampled rows. Quick can change selected K and assignments.')
            st.caption('Research preserves the previous clustering budget. Both profiles keep exact masking and certificate calculations; no GPU needed.')
            if st.button('Run clustering and matched baseline',type='primary'):
                st.session_state.pop('doc_audit',None);st.session_state.pop('seed_audit',None)
                def run_doc():
                    from time import perf_counter
                    from sklearn.decomposition import PCA
                    begin=perf_counter(); key=(st.session_state['import_key'],int(seed)); cached=st.session_state.get('prepared_doc')
                    hit=cached is not None and cached[0]==key
                    if not hit:
                        with st.spinner('Preparing feature blocks…'):
                            cached=(key,prepare_documents(records,int(seed)));st.session_state['prepared_doc']=cached
                    bar=st.progress(0.,text='Fitting candidate clusters')
                    try: result=cluster_documents(records,int(kmin),int(kmax),int(seed),kappa,norm,profile=profile,prepared=cached[1],progress=lambda i,n,t:bar.progress(i/max(n,1),text=t))
                    finally: bar.empty()
                    x=result['embedding']
                    with threadpool_limits(limits=1):result['projection']=PCA(n_components=min(2,x.shape[1]),random_state=17).fit_transform(x)
                    result['keywords']=cluster_keywords(result['records'],result['labels']);result['prepared_cache_hit']=hit;result['wall_seconds']=perf_counter()-begin
                    return result
                st.session_state['cluster_result']=safe(run_doc)
                refresh_workflow()
            res=st.session_state.get('cluster_result')
            if res is not None:
                detail=res['details']; diag=res['diagnostics'];x=res['embedding']; st.caption(f'Last completed run: κ={res["kappa"]:.6f}, seed={res["seed"]}, row-normalized={res["row_normalized"]}, K range {res["k_min"]}–{res["k_max"]}. Rerun after changing controls.')
                st.caption(f"Profile: {res['profile']} · {res['fit_count']} K-means fits · baseline reused: {res['baseline_reused']} · feature cache hit: {res['prepared_cache_hit']} · OCR: {st.session_state['ocr_enabled']} · run time: {res['wall_seconds']:.2f} s")
                if res['profile']=='Quick':st.info('Quick mode uses fewer restarts and may use sampled silhouette. Use Research for full-budget comparisons.')
                a,b,c=st.columns(3);a.metric('Selected K',res['best_k']);b.metric('Assignable documents',len(detail));c.metric('Unassignable',len(res['excluded']))
                if res['excluded']: st.dataframe(pd.DataFrame(res['excluded']));download('Export unassignable documents',pd.DataFrame(res['excluded']),'bargcc_unassignable.csv')
                left,right=st.columns(2)
                with left: chart(px.line(diag,x='K',y=['silhouette','baseline_silhouette'],markers=True,title='K selection · matched baseline'),compact=True)
                with right: chart(px.line(diag,x='K',y='inertia',markers=True,title='Elbow diagnostic'),compact=True)
                with st.expander('Detailed K selection table'): st.dataframe(diag,width="stretch")
                projection=res['projection']
                projection_df=detail[['filename','cluster']].copy();projection_df['cluster']=projection_df.cluster.astype(str);projection_df['PC1']=projection[:,0];projection_df['PC2']=projection[:,1] if projection.shape[1]>1 else 0
                chart(px.scatter(projection_df,x='PC1',y='PC2',color='cluster',hover_name='filename',title='Exploratory PCA projection'),width="stretch")
                st.caption('Hover to inspect records; use the chart toolbar to zoom, pan, reset or export a PNG. PCA is exploratory, not a certificate.')
                render_cluster_summary(detail, res)
                st.subheader('Document directory');f=st.selectbox('Cluster filter',['All']+sorted(detail.cluster.unique().tolist()));query=st.text_input('Search filename or text');filtered=detail if f=='All' else detail[detail.cluster==f];filtered=filtered[filtered.filename.str.contains(query,case=False,regex=False)|filtered.text_preview.str.contains(query,case=False,regex=False)] if query else filtered;st.dataframe(filtered,width="stretch")
                if len(filtered):
                    render_document_inspector(filtered, detail, res)
                st.subheader('Independent diagnostics');p=st.slider('Simulated single-view loss probability',0.,1.,.1,.01,key='doc_p')
                if st.button('Audit one-refit stability',disabled=not res['row_normalized']):
                    def audit():
                        d=view_loss_audit(res['views'],res['mask'],res['scores'],res['kappa'],res['best_k'],res['seed'],p,res['labels']);d['filename']=detail.filename;return d
                    st.session_state['doc_audit']=safe(audit);st.session_state['doc_audit_probability']=p
                    refresh_workflow()
                if st.session_state.get('doc_audit') is not None:
                    st.caption(f'Last audit dropout probability: {st.session_state.get("doc_audit_probability",0):.0%}');show_coverage(st.session_state['doc_audit'],'bargcc_document')
                if not res['row_normalized']:st.caption('View-loss Eq. (22) requires row-normalized unit blocks. Select that mode and rerun.')
                if st.button('Run seed-sensitivity audit'):
                    with threadpool_limits(limits=1): st.session_state['seed_audit']=safe(lambda:stability_audit(x,res['best_k']))
                if st.session_state.get('seed_audit') is not None:st.json(st.session_state['seed_audit'])
                download('Download assignments (CSV)',detail,'bargcc_document_assignments.csv');download('Download K diagnostics (CSV)',diag,'bargcc_k_diagnostics.csv');json_dl('Download reproducibility report',{'archive_sha256':st.session_state['archive_sha256'],'seed':res['seed'],'kappa':res['kappa'],'row_normalized':res['row_normalized'],'selected_k':res['best_k'],'K_range':[res['k_min'],res['k_max']],'diagnostics':diag.to_dict('records'),'excluded':res['excluded'],'skipped':skipped,'semantic_validation':'not performed','profile':res['profile'],'n_init':res['n_init'],'fit_count':res['fit_count'],'baseline_reused':res['baseline_reused'],'feature_cache_hit':res['prepared_cache_hit'],'wall_seconds':res['wall_seconds'],'ocr_enabled':st.session_state['ocr_enabled'],'python':sys.version,'platform':platform.platform()},'bargcc_archive_manifest.json')
                st.warning('K selection and comparison use the uploaded corpus; internal indices and partition ARI do not establish semantic superiority.')
        else:st.warning('At least three assignable records are required to select K below the record count.')

elif page.startswith('06'):
    header('06 / CONDITIONAL ASSURANCE','View-loss certification laboratory','Compare the manuscript’s event-conditioned results with fresh controlled simulations.')
    df=pd.read_csv(RESULTS/'conditional_coverage_audit'/'conditional_coverage.csv'); show=df[['dropout_probability','certification_all','certification_dropped','certification_retained','unchanged_all']].copy();show.iloc[:,1:]*=100; show['dropout_probability']*=100
    chart(px.line(show,x='dropout_probability',y=list(show.columns[1:]),markers=True,labels={'value':'Share (%)','dropout_probability':'Dropout probability (%)'},title='Archived event-conditioned audit'),width="stretch");st.dataframe(df,width="stretch");download('Download archived coverage',df,'bargcc_archived_coverage.csv')
    st.subheader('New Digits simulation · one held-out fold');note('Protocol','Train-fold-only scaling/PCA12; clustering within the test fold is transductive. Deterministic scores are uncalibrated. This is not the archived five-fold replay.')
    rho,kappa=gate_controls('loss');p=st.slider('Single-view dropout probability',0.,1.,.1,.01);seed=st.number_input('Seed',0,1000000,42);k=st.slider('Clusters',2,15,10)
    if st.button('Execute new view-loss experiment',type='primary'):
        st.session_state['digits']=safe(lambda:digits_demo(kappa,p,int(seed),k));st.session_state['digits_config']={'kappa':kappa,'p':p,'seed':seed,'K':k}
    if st.session_state.get('digits') is not None:
        st.caption('Last run: '+str(st.session_state['digits_config']));show_coverage(st.session_state['digits'],'bargcc_live_digits');json_dl('Export live run configuration',st.session_state['digits_config'],'bargcc_live_config.json')
    st.latex(r'\|z-z^\prime\|_2=\sqrt{2(1-\sqrt{1-q})}')
    st.latex(r'\|z_i-\mu_b\|_2-\|z_i-\mu_a\|_2>2E_i+\overline E_a+\overline E_b\quad\forall b\ne a')

else:
    header('07 / RESEARCH RECORD','Architecture and provenance','Trace application methods to the corrected 44-page IJSS manuscript and its archived evidence.')
    st.image(str(ROOT/'assets'/'architecture.png'),caption='Figure 1 · supplied manuscript architecture',width="stretch")
    st.markdown('**BARG-CC: reliability-constrained information fusion with certified view-loss stability and failure diagnostics**')
    st.caption('Debmalya Ray and Prodipto Das · supplied corrected IJSS manuscript (2026); not represented as an accepted publication.')
    st.download_button('Download supplied manuscript PDF',(ROOT/'reference'/'manuscript.pdf').read_bytes(),'BARG_CC_IJSS_2026_Corrected_Main_44p.pdf','application/pdf')
    st.markdown((ROOT/'reference'/'ASSURANCE_EVIDENCE_MATRIX.md').read_text(encoding='utf-8'))
    st.caption('The matrix is copied from the supplied source; historical test counts are not app test results.')
    st.subheader('Evidence integrity');manifest=json.loads((ROOT/'reference'/'archive_sha256.json').read_text());
    if st.button('Verify bundled evidence hashes'):
        render_evidence_status(manifest, ROOT)
