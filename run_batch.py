"""Local, browser-free document processing with reproducible CSV/JSON outputs."""
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
from document_pipeline import _extract_zip, MAX_ARCHIVE
from engine import cluster_documents, view_loss_audit, coverage
from hamv_gate import strength_for_floor
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,default=Path('batch_output'))
    p.add_argument('--profile',choices=['Research','Quick'],default='Research');p.add_argument('--ocr',action='store_true')
    p.add_argument('--k-min',type=int,default=2);p.add_argument('--k-max',type=int,default=4);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--gated',action='store_true');p.add_argument('--floor',type=float,default=.1);p.add_argument('--dropout',type=float,default=None)
    a=p.parse_args();start=time.perf_counter()
    if a.archive.suffix.lower()!='.zip' or a.archive.stat().st_size>MAX_ARCHIVE:p.error('Expected ZIP under 500 MB')
    if not 0<a.floor<=1/3 or (a.dropout is not None and not 0<=a.dropout<=1):p.error('Invalid floor or dropout')
    if a.output.exists() and any(a.output.iterdir()):p.error('Output directory must be empty; choose a new directory')
    def progress(i,n,t):
        if i==n or i%max(1,n//10)==0:print(f'[{i}/{n}] {t}',flush=True)
    rec,skipped=_extract_zip(a.archive,ocr=a.ocr,progress=progress);kappa=strength_for_floor(a.floor,3) if a.gated else 0.
    res=cluster_documents(rec,a.k_min,a.k_max,a.seed,kappa,profile=a.profile,progress=progress)
    a.output.mkdir(parents=True,exist_ok=True)
    def save(df,name):
        df=df.copy()
        for c in df.select_dtypes(include=['object','string']).columns:df[c]=df[c].map(lambda x:"'"+x if isinstance(x,str) and x.lstrip().startswith(('=','+','-','@')) else x)
        df.to_csv(a.output/name,index=False,encoding='utf-8-sig')
    save(res['details'],'assignments.csv');save(res['diagnostics'],'k_diagnostics.csv');save(pd.DataFrame(skipped,columns=['file','reason']),'skipped.csv');save(pd.DataFrame(res['excluded'],columns=['filename','document_id','reason','assignment_status']),'unassignable.csv')
    if a.dropout is not None:
        df=view_loss_audit(res['views'],res['mask'],res['scores'],kappa,res['best_k'],a.seed,a.dropout,res['labels']);df['filename']=res['details'].filename;save(df,'view_loss.csv');save(coverage(df),'coverage.csv')
    h=hashlib.sha256()
    with a.archive.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    report={k:res[k] for k in ['profile','seed','kappa','best_k','k_min','k_max','fit_count','n_init','baseline_reused','compute_seconds']};report.update(archive_sha256=h.hexdigest(),ocr=a.ocr,dropout=a.dropout,semantic_validation='not performed',wall_seconds=time.perf_counter()-start,python=sys.version,platform=platform.platform())
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
