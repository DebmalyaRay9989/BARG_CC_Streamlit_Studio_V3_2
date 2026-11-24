#!/usr/bin/env python3
"""Mask-switch reliability-fusion theorem and empirical audit.

All results are MECHANISM audits, not demonstrations of clustering accuracy.
A disappearing view is renormalised by masked softmax. Each available
projected block is unit-norm, and the concatenated representation uses
sqrt(alpha_m) times that block (rather than sqrt(M*alpha_m)).

Real-data audit: 5 non-overlapping test folds of sklearn Digits, 3 derived
pseudo-views from one image, train-only preprocessing, label-blind KMeans
fitted within held-out fold to examine self-consistent centroid refitting.
This is TRANSDUCTIVE and is not a held-out inductive clustering benchmark.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize

KAPPA=0.7520386983881371


def weights(scores: np.ndarray, mask: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    mask=np.asarray(mask, dtype=bool)
    scores=np.asarray(scores, dtype=float)
    if scores.shape != mask.shape or scores.ndim != 2:
        raise ValueError('scores and mask must be matching [records,views] matrices')
    if not np.all(mask.any(axis=1)):
        raise ValueError('at least one observed view is mandatory for every record')
    if np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('scores must be in [0,1]')
    logits=np.where(mask,kappa*(2*scores-1),-np.inf)
    p=np.exp(logits-np.max(logits,axis=1,keepdims=True))
    return p/p.sum(axis=1,keepdims=True)


def fuse(view_embeddings: list[np.ndarray], alpha: np.ndarray) -> np.ndarray:
    if len(view_embeddings) != alpha.shape[1]:
        raise ValueError('block-view mismatch')
    for x in view_embeddings:
        if x.shape[0] != alpha.shape[0]:
            raise ValueError('record mismatch')
    return np.concatenate([np.sqrt(alpha[:,j:j+1])*v for j,v in enumerate(view_embeddings)],axis=1)


def dropout_eps(alpha: np.ndarray, old_mask: np.ndarray, new_mask: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    old_mask=np.asarray(old_mask,dtype=bool); new_mask=np.asarray(new_mask,dtype=bool)
    if not np.all(new_mask <= old_mask) or not new_mask.any(axis=1).all():
        raise ValueError('single-event dropout only; must retain an observed view')
    q=np.sum(alpha*(old_mask & ~new_mask),axis=1)
    return np.sqrt(np.maximum(0.,2*(1-np.sqrt(np.maximum(0.,1-q))))),q


def pairwise_distances(x:np.ndarray, centers:np.ndarray) -> np.ndarray:
    return np.linalg.norm(x[:,None,:]-centers[None,:,:],axis=2)


def lloyd_refit_certificate(x,x_new,eps,ref_labels,k):
    """Pointwise sufficient guarantee for *one refit of fixed memberships*.

    NOT a global KMeans guarantee or a certificate for arbitrary iterates.
    """
    n=len(x); ref_labels=np.asarray(ref_labels,dtype=int)
    assert np.all(np.bincount(ref_labels,minlength=k)>0)
    c=np.array([x[ref_labels==j].mean(axis=0) for j in range(k)])
    c_new=np.array([x_new[ref_labels==j].mean(axis=0) for j in range(k)])
    ebar=np.array([eps[ref_labels==j].mean() for j in range(k)])
    assert np.all(np.linalg.norm(c-c_new,axis=1)<=ebar+1e-6)
    d=pairwise_distances(x,c)
    dnew=pairwise_distances(x_new,c_new)
    # Margin for each point against each competitor.
    threshold = 2*eps[:,None] + ebar[ref_labels,None]+ebar[None,:]
    margin=d-d[np.arange(n),ref_labels,None]
    other=np.arange(k)[None,:]!=ref_labels[:,None]
    # d(comp)-d(own) > threshold(comp).
    certified=np.all((margin>threshold)|~other,axis=1)
    nearest_new=dnew.argmin(axis=1)
    nearest_old=d.argmin(axis=1)
    if not np.array_equal(nearest_old,ref_labels):
        raise AssertionError('reference KMeans partition is not a Lloyd fixed point')
    if np.any(certified & (nearest_new!=ref_labels)):
        raise AssertionError('false self-consistent-refit certificate')
    return certified,nearest_new,ebar


def views_from_digits(x:np.ndarray) -> list[np.ndarray]:
    # Source is 8x8 pixel image; all views are derived from SAME record.
    im=x.reshape(-1,8,8)
    sums=np.concatenate([im.sum(1),im.sum(2)],axis=1)
    grads=np.concatenate([(im[:,:,1:]-im[:,:,:-1]).reshape(-1,56),
                          (im[:,1:,:]-im[:,:-1,:]).reshape(-1,56)],axis=1)
    return [sums,x.copy(),grads]


def fit_pseudoviews(train,test):
    tr=views_from_digits(train); te=views_from_digits(test)
    return _fit_views(tr,te)


def _fit_views(train,test):
    outputs=[]
    for a,b in zip(train,test):
        scale=StandardScaler().fit(a)
        pca=PCA(n_components=12, svd_solver='full').fit(scale.transform(a))
        v=normalize(pca.transform(scale.transform(b)))
        if not np.allclose(np.linalg.norm(v,axis=1),1.,atol=1e-9):
            raise AssertionError('zero or nonunit test embedding found')
        outputs.append(v)
    return outputs


def run_digits(out, repetitions=5):
    x,y=load_digits(return_X_y=True)
    sf=StratifiedKFold(n_splits=repetitions,shuffle=True,random_state=20261123)
    rows=[]; point_rows=[]
    for fold,(idxtr,idxts) in enumerate(sf.split(x,y)):
        view=fit_pseudoviews(x[idxtr],x[idxts]); n=len(idxts); m=len(view)
        # Deterministic, uncalibrated geometric test scores; NOT a quality regressor.
        # Avoid presenting these as learned probabilities of semantic usefulness.
        score=np.column_stack([np.clip(.5+.2*np.tanh(v[:,0]),0,1) for v in view])
        old_mask=np.ones((n,m),dtype=bool)
        alpha=weights(score,old_mask)
        a=fuse(view,alpha)
        km=KMeans(n_clusters=10,n_init=10,random_state=100+fold).fit(a)
        orig=km.labels_.astype(int)
        for p in [0.,.02,.05,.1,.2,.4]:
            rng=np.random.default_rng(700+fold*100+round(1000*p))
            newmask=old_mask.copy()
            drop=rng.random(n)<p
            sel=rng.integers(m,size=n)
            newmask[np.flatnonzero(drop),sel[drop]]=False
            eps,q=dropout_eps(alpha,old_mask,newmask)
            alpha_new=weights(score,newmask)
            zz=fuse(view,alpha_new)
            err=np.linalg.norm(a-zz,axis=1)
            if not np.allclose(err,eps,atol=1e-10):
                raise AssertionError('the exact dropout law failed')
            cert,new_labels,drift=lloyd_refit_certificate(a,zz,eps,orig,10)
            stable=(orig==new_labels)
            rows.append(dict(dataset='digits_derived_views',fold=fold,dropout_probability=p,
                n_records=n, n_changed=int(drop.sum()),
                certified_n=int(cert.sum()),certified_pct=float(cert.mean()),
                stationary_n=int(stable.sum()),stationary_pct=float(stable.mean()),
                unchanged_ARI=float(adjusted_rand_score(orig,new_labels)),
                false_certificates=int((cert & ~stable).sum()),
                max_exact_eps=float(eps.max()),mean_exact_eps=float(eps.mean()),
                max_centroid_drift=float(drift.max())))
            point_rows.extend(dict(fold=fold,dropout_probability=p,record_id=int(idxts[i]),
                dropped=bool(drop[i]),removed_reliability_mass=float(q[i]),
                exact_representation_change=float(eps[i]),certified=bool(cert[i]),
                refitted_nearest_unchanged=bool(stable[i])) for i in range(n))
        print('Audited Digits fold',fold,'n',n,flush=True)
    df=pd.DataFrame(rows)
    pd.DataFrame(point_rows).to_csv(out/'pointwise.csv',index=False)
    df.to_csv(out/'fold_results.csv',index=False)
    summ=df.groupby('dropout_probability',as_index=False).agg(
       folds=('fold','count'),records=('n_records','sum'),
       certified_mean=('certified_pct','mean'),certified_min=('certified_pct','min'),
       stationary_mean=('stationary_pct','mean'),stationarity_ARI_mean=('unchanged_ARI','mean'),
       false_certificates=('false_certificates','sum'),changed_total=('n_changed','sum'))
    summ.to_csv(out/'summary.csv',index=False)
    return df,summ


def run_observational_feature_groups(out):
    """Optional natural feature GROUP audit, NOT independent sensing modalities.

    Uses the same ANES and transport-mode source datasets already in the paper.
    This is another label-blind transductive centroid-consistency mechanism
    audit, not a new independent-domain or method-comparison score.
    """
    import benchmark_first_run_observational as o
    rows=[]
    for name in o.DATASETS:
      for seed in o.SEEDS:
        (xtr,xval,xt),y,k,split_hash,data_hash,meta,ns=o.split_data(name,seed)
        v=[normalize(u) for u in xt]
        assert all(np.linalg.norm(u,axis=1).min()>.999 for u in v)
        n=len(v[0]);m=len(v)
        score=np.column_stack([.5+.2*np.tanh(u[:,0]) for u in v])
        old=np.ones((n,m),bool)
        alpha=weights(score,old); z=fuse(v,alpha)
        km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(z)
        for p in [.1,.2,.4]:
            rng=np.random.default_rng(9800+seed+round(1000*p))
            dropped=rng.random(n)<p;which=rng.integers(m,size=n)
            mask=old.copy();mask[np.flatnonzero(dropped),which[dropped]]=False
            a2=weights(score,mask); zz=fuse(v,a2)
            eps,q=dropout_eps(alpha,old,mask)
            assert np.allclose(np.linalg.norm(z-zz,axis=1),eps,atol=1e-10)
            certified,nxt,_=lloyd_refit_certificate(z,zz,eps,km.labels_,k)
            rows.append(dict(dataset=name,seed=seed,dropout_probability=p,n=n,
                n_dropped=int(dropped.sum()),certified=float(certified.mean()),
                changed_rows_certified=float(certified[dropped].mean()) if dropped.any() else 0.,
                unchanged=float((nxt==km.labels_).mean()),
                false_certificates=int((certified&(nxt!=km.labels_)).sum()),
                split_sha256=split_hash,data_sha256=data_hash,
                inference='overlapping repeated-subject split; no independent population CI'))
    df=pd.DataFrame(rows);df.to_csv(out/'observational_group_raw.csv',index=False)
    sm=df.groupby(['dataset','dropout_probability'],as_index=False).agg(
      seeds=('seed','count'),certified=('certified','mean'),
      changed_certified=('changed_rows_certified','mean'),
      unchanged=('unchanged','mean'),false_certificates=('false_certificates','sum'))
    sm.to_csv(out/'observational_group_summary.csv',index=False)
    assert not df.false_certificates.any()
    return df,sm


def run_floor_tradeoff(out, repetitions=5):
    """Idealized low-participation view removal, not a learned quality test.

    View-0 is assigned score 0 and the other views score 1 BY DESIGN.
    Drop only view-0 for ~10% of records; compare gate concentrations.
    Thus kappa>0.752 deliberately relaxes the nominal 10% floor.
    """
    x,y=load_digits(return_X_y=True)
    sf=StratifiedKFold(n_splits=repetitions,shuffle=True,random_state=20261123)
    records=[]
    for fold,(idxtr,idxts) in enumerate(sf.split(x,y)):
        v=fit_pseudoviews(x[idxtr],x[idxts]); n=len(idxts)
        score=np.tile([0.,1.,1.],(n,1))
        old=np.ones((n,3),bool)
        rng=np.random.default_rng(8800+fold)
        dropped=rng.random(n)<.1
        new=old.copy();new[dropped,0]=False
        for kap in [0.,KAPPA,2.,4.]:
            alpha=weights(score,old,kappa=kap)
            a=fuse(v,alpha)
            km=KMeans(n_clusters=10,n_init=10,random_state=100+fold).fit(a)
            orig=km.labels_.astype(int)
            new_alpha=weights(score,new,kappa=kap)
            aa=fuse(v,new_alpha)
            eps,q=dropout_eps(alpha,old,new)
            assert np.allclose(np.linalg.norm(a-aa,axis=1),eps,atol=1e-10)
            cert,nxt,_=lloyd_refit_certificate(a,aa,eps,orig,10)
            assert not (cert & (nxt!=orig)).any()
            records.append(dict(fold=fold,kappa=float(kap),min_assigned_view_share=float(alpha[0,0]),
                lost_view_count=int(dropped.sum()),
                changed_view_certified=float(cert[dropped].mean()) if dropped.any() else 0.,
                changed_view_unchanged=float((nxt[dropped]==orig[dropped]).mean()) if dropped.any() else 0.,
                all_certified=float(cert.mean()),
                new_assignments_unchanged=float((nxt==orig).mean()),
                false_certificates=int((cert & (nxt!=orig)).sum())))
    df=pd.DataFrame(records)
    df.to_csv(out/'floor_tradeoff_raw.csv',index=False)
    grp=df.groupby('kappa',as_index=False).agg(
        min_share=('min_assigned_view_share','mean'),
        changed_certified=('changed_view_certified','mean'),
        changed_unchanged=('changed_view_unchanged','mean'),
        all_certified=('all_certified','mean'),
        false_certificates=('false_certificates','sum'),
        n_removed=('lost_view_count','sum'))
    grp.to_csv(out/'floor_tradeoff_summary.csv',index=False)
    return df,grp


def run_exact_small(out, nseeds=20):
    """Reuses the prior exact 8-point global partition evaluator.
    Drops at most one channel; report conditional exact stability only.
    """
    import global_reoptimized_certificate as g
    rows=[]; plist=list(g.partitions(8,2))
    for scenario in ['separated','overlapping']:
      for prob_drop in [0.,.1,.25,.5]:
       for s in range(nseeds):
        rng=np.random.default_rng(10101+s)
        base=np.repeat([-1,1],4)* (3. if scenario=='separated' else .7)
        e=[]
        for j in range(3):
          z=np.zeros((8,3));z[:,0]=base
          z+=rng.normal(0,.10 if scenario=='separated' else 1.5,z.shape)
          e.append(normalize(z))
        score=rng.uniform(.35,.75,(8,3)); mask=np.ones((8,3),bool)
        a=weights(score,mask); x=fuse(e,a)
        newmask=mask.copy()
        drop=rng.random(8)<prob_drop; sel=rng.integers(3,size=8)
        newmask[np.flatnonzero(drop),sel[drop]]=False
        ap=weights(score,newmask);xp=fuse(e,ap)
        eps,q=dropout_eps(a,mask,newmask)
        assert np.allclose(np.linalg.norm(x-xp,axis=1),eps,atol=1e-10)
        z=g.certificate(x,xp,eps,plist)
        rows.append(dict(scenario=scenario,dropout_probability=prob_drop,seed=s,
            certified=z['certified'],optimal_unchanged=z['optimal_unchanged'],
            actual_dropped=int(drop.sum()),margin=z['second_best_gap'],
            certified_gap_lower=z['certified_gap_lower']))
    f=pd.DataFrame(rows); f.to_csv(out/'global_exact_raw.csv',index=False)
    summary=f.groupby(['scenario','dropout_probability'],as_index=False).agg(
      n=('seed','count'),certified=('certified','sum'),unchanged=('optimal_unchanged','sum'),
      dropped=('actual_dropped','sum'))
    summary.to_csv(out/'global_exact_summary.csv',index=False)
    assert not (f.certified & ~f.optimal_unchanged).any()
    return f,summary


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',required=True)
    args=pa.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    real,summ=run_digits(out)
    small,smallsummary=run_exact_small(out)
    ideal,ideal_summary=run_floor_tradeoff(out)
    obs,obs_summary=run_observational_feature_groups(out)
    manifest=dict(status='POST_HOC_EXPLORATORY_MASK_SWITCH_MECHANISM',
        source='scikit-learn Digits; three derived views of same image; no natural modalities',
        method='train-only PCA and standardization; held-out transductive KMeans mechanism audit',
        score='deterministic geometric pseudo-score; not a calibrated reliability estimator',
        assumption='unit Euclidean norm of each view; removed weights renormalize with fixed logits',
        interpretations=['exact per-record bound for dropout',
          'centroid-refit one-step Lloyd self-consistency; not global optimum or convergence',
          'global exact partition proof checked only n=8, K=2'],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),numpy=np.__version__,n_digits_rows=len(real),
        n_small_global_trials=len(small), false_certificates=int(real.false_certificates.sum()),
        n_source_datasets=3, no_specialist_official_baselines=True,
        observational_note='ANES and modechoice are reused observational feature groups, not newly acquired independent modalities; split seeds overlap respondents',
        idealized_floor_tradeoff='score [0,1,1] assigned deliberately: not an observed reliability estimate; stronger kappa reduces nominal participation floor')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('\nREAL DIGITS:');print(summ.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nEXACT SMALL N:');print(smallsummary.to_string(index=False))
    print('\nIDEALIZED FLOOR TRADEOFF:');print(ideal_summary.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nOBSERVATIONAL FEATURE GROUPS:');print(obs_summary.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nAll certificates verified; 0 false. First-run exploratory, not confirmation.')

if __name__=='__main__': main()
