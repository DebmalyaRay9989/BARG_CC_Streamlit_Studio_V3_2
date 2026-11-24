#!/usr/bin/env python3
"""Exact small-n global k-means partition certificate for BARG-CC.

This is a mathematical/numerical *mechanism* experiment on simulated views.
It is NOT an independent benchmark or an accuracy improvement claim.
The certificate includes REFITTED centroids by representing the full k-means
objective through within-cluster pairwise squared distances and enumerating
all unlabeled partitions (restricted growth strings). Complexity is exponential.
"""
from __future__ import annotations
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd

KAPPA=0.7520386983881371

def partitions(n,k):
    """Yield canonical partition labels with exactly k nonempty clusters."""
    z=np.zeros(n,dtype=np.int16)
    def rec(t,mx):
        if t==n:
            if mx+1==k: yield z.copy()
            return
        for a in range(min(mx+1,k-1)+1):
            z[t]=a
            if mx+1+(1 if a==mx+1 else 0)+(n-t-1)<k: continue
            yield from rec(t+1,max(mx,a))
    z[0]=0
    yield from rec(1,0)

def fuse(blocks,scores):
    # Full-availability, separated block rule, with sqrt(M alpha) scaling.
    m=len(blocks)
    q=KAPPA*(2*scores-1)
    p=np.exp(q-q.max(axis=1,keepdims=True))
    p=p/p.sum(axis=1,keepdims=True)
    return np.concatenate([np.sqrt(m*p[:,j:j+1])*b for j,b in enumerate(blocks)],axis=1)

def objective(x,labels):
    s=0.
    for c in np.unique(labels):
        u=x[labels==c]
        s+=float(np.sum((u-u.mean(0))**2))
    return s

def objective_pairwise(x,labels):
    s=0.
    for c in np.unique(labels):
        u=x[labels==c]
        dd=((u[:,None,:]-u[None,:,:])**2).sum(2)
        s+=float(dd.sum()/(2*len(u)))
    return s

def partition_budget(x,eps,labels):
    s=0.
    for c in np.unique(labels):
        ii=np.flatnonzero(labels==c)
        for ti,i in enumerate(ii):
            for j in ii[ti+1:]:
                distance=np.linalg.norm(x[i]-x[j]); sdiff=eps[i]+eps[j]
                s+=(2*distance*sdiff+sdiff**2)/len(ii)
    return float(s)

def certificate(x,x_new,eps,plist):
    assert np.all(np.linalg.norm(x-x_new,axis=1)<=eps+1e-9)
    js=np.array([objective(x,p) for p in plist]); jnew=np.array([objective(x_new,p) for p in plist])
    best=int(np.argmin(js)); bestnew=int(np.argmin(jnew))
    assert np.isclose(js[best],objective_pairwise(x,plist[best]),atol=1e-10)
    budgets=np.array([partition_budget(x,eps,p) for p in plist])
    assert np.all(np.abs(jnew-js)<=budgets+1e-8), 'Objective perturbation violated'
    competitor=np.min(np.delete(js-budgets,best))
    certified=bool(js[best]+budgets[best] < competitor-1e-12)
    if certified: assert best==bestnew, 'Certified but optimum changed'
    second=np.partition(js,1)[1]
    return dict(certified=certified,optimal_unchanged=bool(best==bestnew),
                n_partitions=len(plist),best_objective=float(js[best]),
                second_best_gap=float(second-js[best]),
                certified_gap_lower=float(competitor-(js[best]+budgets[best])),
                largest_objective_change=float(np.max(np.abs(js-jnew))),
                max_allowed_objective_change=float(budgets.max()))

def trial(scenario,seed,delta,eta):
    rng=np.random.default_rng(seed)
    n=8; m=3; k=2; dim=3
    spread=0.055 if scenario=='separated' else 2.25
    y=np.repeat([-1.,1.],4)
    base=np.zeros((n,dim));base[:,0]=y*(2.7 if scenario=='separated' else 0.8)
    blocks=[base+rng.normal(0,spread,(n,dim)) for _ in range(m)]
    scores=rng.uniform(.35,.75,(n,m))
    x=fuse(blocks,scores)
    score_shift=rng.uniform(-delta,delta,(n,m))
    scores2=np.clip(scores+score_shift,0,1)
    errors=[rng.uniform(-eta,eta,size=(n,dim)) for _ in range(m)]
    newblocks=[b+e for b,e in zip(blocks,errors)]
    xnew=fuse(newblocks,scores2)
    radius=max(np.linalg.norm(b,axis=1).max() for b in blocks)
    max_e=max(np.linalg.norm(e,axis=1).max() for e in errors)
    # Proposition 10 supplies the local uncertainty budget for every point.
    eps=np.full(n,radius*np.sqrt(m)*min(KAPPA*delta,2)+np.sqrt(m)*max_e)
    plist=list(partitions(n,k))
    r=certificate(x,xnew,eps,plist)
    r.update(scenario=scenario,seed=seed,delta_score=delta,max_embedding_drift=max_e,
             actual_max_representation_drift=float(np.max(np.linalg.norm(x-xnew,axis=1))),
             theoretical_max_representation_drift=float(np.max(eps)))
    return r

def main(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Design stages are numerical tests, NOT external datasets.
    for scenario in ('separated','overlapping'):
        for delta,eta in ((0.0,0.0),(.0005,.001),(.002,.004),(.01,.015),(.03,.05),(.08,.18),(.20,.55)):
            for seed in range(60,75):
                rows.append(trial(scenario,seed,delta,eta))
    df=pd.DataFrame(rows);df.to_csv(out/'raw.csv',index=False)
    summary=df.groupby(['scenario','delta_score'],as_index=False).agg(
        cases=('seed','count'), certified=('certified','sum'),
        unchanged=('optimal_unchanged','sum'),
        mean_margin=('second_best_gap','mean'),
        mean_bound=('theoretical_max_representation_drift','mean'))
    summary.to_csv(out/'summary.csv',index=False)
    h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(code_sha256=h,experiment='small-n global kmeans reoptimization certificate',
        source='synthetic normalized 3-view data; NO natural-view or semantic benchmark',
        n_trials=len(df),global_partitions=int(df.n_partitions.iloc[0]),
        environment={'python':platform.python_version(),'numpy':np.__version__},
        certificate_checks='exact objectives for all unlabeled K-partitions plus exact perturbation bound',
        statistical_limitation='seed repetitions from the same specified generative simulation, not independent domain inference'),indent=2))
    print(summary.to_string(index=False))
    print('All',len(df),'individually bounded cases passed; certified-and-failed',len(df[(df.certified)&(~df.optimal_unchanged)]))
    return df

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args()
    main(args.output)
