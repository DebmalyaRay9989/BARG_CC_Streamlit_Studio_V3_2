"""Application adapters for the corrected IJSS BARG-CC equations.
Live experiments are exploratory; archived evidence is never overwritten.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from time import perf_counter
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
from hamv_gate import masked_gate, strength_for_floor, block_fuse
from mask_switch_certificate import dropout_eps, fit_pseudoviews
from global_reoptimized_certificate import partitions, certificate

NAMES=['Structural','Visual','Text']

def fuse(views,mask,scores,kappa=0.,row_normalized=True):
    a=masked_gate(scores,mask,kappa)
    b=block_fuse(views,mask,a)
    return (b/np.sqrt(mask.sum(1))[:,None] if row_normalized else b),a

def fit_lloyd(x,k,seed,n_init=10):
    with threadpool_limits(limits=1):
        model=KMeans(n_clusters=k,n_init=n_init,random_state=seed,max_iter=1000).fit(x)
    if len(np.unique(model.labels_))!=k: raise ValueError('Fewer distinct clusters than requested K.')
    lab=model.labels_; c=np.array([x[lab==j].mean(0) for j in range(k)])
    d=cdist(x,c)
    if np.any(d[np.arange(len(x)),lab]>d.min(1)+1e-10): raise ValueError('Reference partition is not a Lloyd fixed point.')
    return lab,c

def view_loss_audit(views,mask,scores,kappa,k,seed,probability,labels=None,drop_seed=None):
    mask=np.asarray(mask,bool); scores=np.asarray(scores,float)
    for j,v in enumerate(views):
        if not np.allclose(np.linalg.norm(v[mask[:,j]],axis=1),1.,atol=1e-8): raise ValueError('Eq. 22 requires unit-norm observed blocks.')
    x,a=fuse(views,mask,scores,kappa)
    lab,_=fit_lloyd(x,k,seed) if labels is None else (np.asarray(labels,int),None)
    if set(lab)!=set(range(k)): raise ValueError('Reference clusters must be nonempty.')
    rng=np.random.default_rng(seed+101 if drop_seed is None else drop_seed)
    new=mask.copy(); eligible=mask.sum(1)>1
    for i in np.flatnonzero(eligible & (rng.random(len(mask))<probability)):
        new[i,rng.choice(np.flatnonzero(mask[i]))]=False
    e,q=dropout_eps(a,mask,new); xp,_=fuse(views,new,scores,kappa)
    actual=np.linalg.norm(xp-x,axis=1)
    if not np.allclose(e,actual,atol=1e-8): raise AssertionError('Exact view-loss identity failed.')
    c=np.array([x[lab==j].mean(0) for j in range(k)]); cp=np.array([xp[lab==j].mean(0) for j in range(k)])
    d=cdist(x,c); dn=cdist(xp,cp)
    own=d[np.arange(len(x)),lab]
    if np.any(own>d.min(1)+1e-10): raise ValueError('Reference is not a Lloyd fixed point.')
    eb=np.array([e[lab==j].mean() for j in range(k)])
    slack=d-own[:,None]-(2*e[:,None]+eb[lab,None]+eb[None,:]); slack[np.arange(len(x)),lab]=np.inf
    certified=slack.min(1)>1e-10; newlab=dn.argmin(1); stable=newlab==lab
    if np.any(certified & ~stable): raise AssertionError('False certificate detected.')
    return pd.DataFrame({'record':np.arange(len(x)),'eligible':eligible,'dropped':np.any(mask!=new,axis=1),'removed_mass':q,'exact_drift':e,'measured_drift':actual,'certificate_slack':slack.min(1),'certified':certified,'unchanged':stable,'original_cluster':lab,'refitted_nearest_cluster':newlab})

def coverage(df):
    rows=[]
    for name,sub in [('All records',df),('View removed',df[df.dropped]),('No view removed',df[~df.dropped])]:
        rows.append({'group':name,'n':len(sub),'certified_pct':100*sub.certified.mean() if len(sub) else None,'unchanged_pct':100*sub.unchanged.mean() if len(sub) else None,'false_certificates':int((sub.certified & ~sub.unchanged).sum())})
    return pd.DataFrame(rows)

def digits_demo(kappa,probability,seed,k=10):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import StratifiedKFold
    x,y=load_digits(return_X_y=True)
    tr,te=next(iter(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)))
    with threadpool_limits(limits=1): views=fit_pseudoviews(x[tr],x[te])
    scores=np.column_stack([np.clip(.5+.2*np.tanh(v[:,0]),0,1) for v in views]); mask=np.ones_like(scores,bool)
    df=view_loss_audit(views,mask,scores,kappa,k,seed,probability)
    df['record']=te
    return df

def exact_demo(seed,scenario,probability,kappa):
    rng=np.random.default_rng(seed); n=8;k=2
    y=np.repeat([-1.,1.],4); base=np.c_[y,np.zeros((n,2))]
    views=[normalize(base+rng.normal(0,.06 if scenario=='Separated' else 1.2,(n,3))) for _ in range(3)]
    r=rng.uniform(.2,.8,(n,3)); mask=np.ones_like(r,bool); x,a=fuse(views,mask,r,kappa)
    new=mask.copy(); ix=np.flatnonzero(rng.random(n)<probability);new[ix,rng.integers(3,size=len(ix))]=False
    xp,_=fuse(views,new,r,kappa); e,q=dropout_eps(a,mask,new)
    out=certificate(x,xp,e,list(partitions(n,k))); out['dropped_records']=len(ix)
    return out

def coordinate_demo(seed,kappa):
    from scipy.spatial.distance import pdist
    rng=np.random.default_rng(seed); v=[normalize(rng.normal(size=(40,5))) for _ in range(3)]; r=rng.random((40,3)); a=rng.random((40,3))>.2;a[~a.any(1),0]=True
    x,w=fuse(v,a,r,kappa); rot=[b@np.linalg.qr(rng.normal(size=(5,5)))[0] for b in v]; xp,_=fuse(rot,a,r,kappa)
    u=sum(w[:,j,None]*b for j,b in enumerate(v));up=sum(w[:,j,None]*b for j,b in enumerate(rot))
    return {'block_max_pairwise_change':float(np.max(np.abs(pdist(x)-pdist(xp)))),'unaligned_sum_max_pairwise_change':float(np.max(np.abs(pdist(u)-pdist(up))))}

def prepare_documents(records,seed):
    n=len(records)
    if not n: raise ValueError('No readable documents.')
    img=np.array([r.get('image_features') is not None for r in records]); raw=np.array([r['image_features'] if ok else [0.]*41 for r,ok in zip(records,img)],float)
    texts=[r.get('text','') for r in records]; txt=np.array([bool(t.strip()) for t in texts])
    def image_block(x):
        out=np.zeros_like(x)
        if img.any():
            # No centering: preserve nonzero feature vectors in small/constant corpora.
            out[img]=StandardScaler(with_mean=False).fit_transform(x[img])
        return normalize(out)
    S=image_block(np.c_[raw[:,:32],raw[:,35:]]); V=image_block(raw[:,32:35]); T=np.zeros((n,1))
    if txt.any():
        try:
            tf=TfidfVectorizer(max_features=4000,ngram_range=(1,2),token_pattern=r'(?u)\b\w+\b').fit_transform(texts)
            if tf.shape[1]>16 and n>1:
                with threadpool_limits(limits=1): T=TruncatedSVD(n_components=min(16,n-1,tf.shape[1]-1),random_state=seed).fit_transform(tf)
            else:T=tf.toarray()
            T=normalize(T)
        except ValueError: pass
    views=[S,V,T]; mask=np.column_stack([img,img,txt]) & np.column_stack([np.linalg.norm(v,axis=1)>1e-12 for v in views]); valid=mask.any(1)
    scores=np.column_stack([np.clip(1-np.abs(raw[:,33]-.25),0,1),np.clip(1-raw[:,34],0,1),np.clip(np.array([len(t.split()) for t in texts])/100,0,1)])
    excluded=[{'filename':r['filename'],'document_id':r['document_id'],'reason':'No usable nonzero encoded block','assignment_status':'UNASSIGNABLE'} for r,ok in zip(records,valid) if not ok]
    return [r for r,ok in zip(records,valid) if ok],[v[valid] for v in views],mask[valid],scores[valid],excluded

PROFILES={'Research': {'n_init':10,'silhouette_limit':None}, 'Quick': {'n_init':3,'silhouette_limit':600}}

@threadpool_limits.wrap(limits=1)
def cluster_documents(records,k_min,k_max,seed,kappa,row_normalized=True,profile='Research',prepared=None,progress=None):
    start=perf_counter()
    config=PROFILES[profile]
    records,v,a,r,excluded=prepare_documents(records,seed) if prepared is None else prepared
    prep_time=perf_counter()-start
    if not 2<=k_min<=k_max<len(records): raise ValueError(f'K range must be 2 ≤ Kmin ≤ Kmax < {len(records)} assignable documents.')
    x,w=fuse(v,a,r,kappa,row_normalized)
    identical=(kappa==0 or np.all(a.sum(1)==1))
    baseline=x if identical else fuse(v,a,r,0,row_normalized)[0]
    # Same seeded sample for every candidate and branch. Research uses every row.
    limit=config['silhouette_limit']; n=len(x)
    ix=np.sort(np.random.default_rng(seed).choice(n,limit,replace=False)) if limit and n>limit else np.arange(n)
    def distance_matrix(z):
        d=cdist(z[ix],z[ix]); np.fill_diagonal(d,0.); return d
    dx=distance_matrix(x); db=dx if identical else distance_matrix(baseline)
    rows=[]; labs={}; fits=0
    def sil(z,d,labels):
        sub=labels[ix]
        if len(np.unique(sub))<2 or len(np.unique(sub))>=len(sub) or len(np.unique(sub))!=len(np.unique(labels)):
            return float(silhouette_score(z,labels)),len(z)
        return float(silhouette_score(d,sub,metric='precomputed')),len(ix)
    for k in range(k_min,k_max+1):
        if progress: progress(k-k_min,k_max-k_min+1,f'Fitting K={k}')
        try:
            lab,c=fit_lloyd(x,k,seed,config['n_init']);fits+=1
            if identical: bl=lab
            else: bl,_=fit_lloyd(baseline,k,seed,config['n_init']);fits+=1
        except ValueError: continue
        score,ns=sil(x,dx,lab); bs,nb=(score,ns) if identical else sil(baseline,db,bl)
        # If one branch needs the full sample, use full scoring for both branches.
        if ns!=nb:
            score=float(silhouette_score(x,lab));bs=float(silhouette_score(baseline,bl));ns=nb=n
        rows.append({'K':k,'silhouette':score,'baseline_silhouette':bs,'inertia':float(((x-c[lab])**2).sum()),'davies_bouldin':davies_bouldin_score(x,lab),'baseline_partition_ARI':adjusted_rand_score(lab,bl),'silhouette_rows':ns}); labs[k]=lab,bl
    if progress: progress(k_max-k_min+1,k_max-k_min+1,'Clustering complete')
    if not rows: raise ValueError('No valid K: check distinct records/features.')
    if len({t['silhouette_rows'] for t in rows})>1:
        for t in rows:
            ll,bb=labs[t['K']];t['silhouette']=float(silhouette_score(x,ll));t['baseline_silhouette']=t['silhouette'] if identical else float(silhouette_score(baseline,bb));t['silhouette_rows']=n
    best=max(rows,key=lambda t:(t['silhouette'],-t['K']))['K']; lab,bl=labs[best]
    detail=[]
    for i,rec in enumerate(records):
        row={key:rec.get(key) for key in ['document_id','filename','sha256','type','size_bytes','ocr_status']}; row.update(cluster=int(lab[i]),baseline_cluster=int(bl[i]),text_preview=rec.get('text','')[:180])
        for j,name in enumerate(NAMES): row.update({name.lower()+'_available':bool(a[i,j]),name.lower()+'_proxy':float(r[i,j]) if a[i,j] else None,name.lower()+'_weight':float(w[i,j])})
        detail.append(row)
    return dict(records=records,views=v,mask=a,scores=r,embedding=x,labels=lab,best_k=best,diagnostics=pd.DataFrame(rows),details=pd.DataFrame(detail),excluded=excluded,kappa=kappa,seed=seed,row_normalized=row_normalized,k_min=k_min,k_max=k_max,profile=profile,n_init=config['n_init'],baseline_reused=identical,fit_count=fits,preparation_seconds=prep_time,compute_seconds=perf_counter()-start)

def fixed_mask_demo(seed,kappa,delta,eta,zeta):
    """Controlled synthetic Proposition 10 diagnostic, unnormalized blocks."""
    rng=np.random.default_rng(seed);n=60; k=3
    y=np.repeat(np.arange(k),20); base=np.eye(k)[y]*3
    v=[base+rng.normal(0,.15,(n,3)) for _ in range(3)];r=rng.random((n,3));a=rng.random((n,3))>.15;a[~a.any(1),0]=True
    x,w=fuse(v,a,r,kappa,False);lab,c=fit_lloyd(x,k,seed)
    rp=np.clip(r+rng.uniform(-delta,delta,r.shape),0,1)
    drift=[normalize(rng.normal(size=b.shape))*rng.uniform(0,eta,(n,1)) for b in v]
    xp,_=fuse([b+e for b,e in zip(v,drift)],a,rp,kappa,False)
    cp=c+normalize(rng.normal(size=c.shape))*rng.uniform(0,zeta,(k,1))
    R=max(np.linalg.norm(b,axis=1).max() for b in v); M=a.sum(1)
    E=R*np.sqrt(M)*min(kappa*delta,np.sqrt(2))+np.sqrt(M)*eta
    d=cdist(x,c);margin=np.sort(d,axis=1)[:,1]-d.min(1)
    actual=np.linalg.norm(xp-x,axis=1); cert=margin>2*(E+zeta)+1e-10; stable=cdist(xp,cp).argmin(1)==d.argmin(1)
    if np.any(actual>E+1e-9) or np.any(cert&~stable):raise AssertionError('Compositional certificate violation')
    return pd.DataFrame({'record':np.arange(n),'margin':margin,'budget':E,'actual_drift':actual,'certified':cert,'unchanged':stable})
