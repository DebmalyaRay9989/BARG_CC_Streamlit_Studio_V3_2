"""Compare original V1 engine with V2 on deterministic synthetic documents.
Use --original-engine pointing to engine.py extracted from the original ZIP.
"""
import argparse, importlib.util, json, platform, statistics, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import engine
from sklearn.metrics import adjusted_rand_score

def fixture(n):
    rng=np.random.default_rng(2026); vocab=[f'term{i:04d}' for i in range(600)]
    rows=[]
    for i in range(n):
        g=i%8; tokens=rng.choice(vocab[g*70:g*70+70],size=36); text=' '.join(tokens)
        raw=np.abs(rng.normal(.2+g*.1,.08,41));raw[33:35]=rng.random(2)*.4
        rows.append(dict(filename=f'doc_{i}.txt',document_id=str(i),text=text,image_features=raw.tolist() if i%5 else None,ocr_status='synthetic'))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--original-engine',required=True,type=Path);p.add_argument('--n',default=1000,type=int);p.add_argument('--repeats',default=3,type=int);p.add_argument('--output',type=Path,default=Path('performance_results.json'));a=p.parse_args()
    spec=importlib.util.spec_from_file_location('original',a.original_engine);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    docs=fixture(a.n);results=[]
    with threadpool_limits(limits=1):
        for strength in (0.,.75):
            timings={};outs={}
            for name,call in [('Original',lambda:original.cluster_documents(docs,2,8,17,strength)),('Research',lambda:engine.cluster_documents(docs,2,8,17,strength)),('Quick',lambda:engine.cluster_documents(docs,2,8,17,strength,profile='Quick'))]:
                ts=[]
                for _ in range(a.repeats):
                    t=time.perf_counter();out=call();ts.append(time.perf_counter()-t)
                timings[name]=statistics.median(ts);outs[name]=out
            old=outs['Original'];new=outs['Research'];quick=outs['Quick'];old_diag=old['diagnostics'];new_diag=new['diagnostics'];err=float(np.max(np.abs(old_diag.silhouette-new_diag.silhouette)))
            assert old['best_k']==new['best_k'] and np.array_equal(old['labels'],new['labels']) and err<1e-10
            results.append(dict(kappa=strength,median_seconds=timings,research_speedup=timings['Original']/timings['Research'],quick_speedup=timings['Original']/timings['Quick'],research_label_parity=True,research_max_silhouette_difference=err,selected_k={name:o['best_k'] for name,o in outs.items()},quick_partition_ari_vs_original=adjusted_rand_score(old['labels'],quick['labels']),research_fit_count=new['fit_count'],quick_fit_count=quick['fit_count'],original_fit_count=14))
    out=dict(scope='Synthetic document records; excludes archive decoding and OCR, includes feature preparation and K search. Single-threaded numerical libraries, median of repeated runs. Quick uses a different budget; speedups are not scientific equivalence claims.',n=a.n,k_range=[2,8],repeats=a.repeats,platform=platform.platform(),results=results);a.output.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
