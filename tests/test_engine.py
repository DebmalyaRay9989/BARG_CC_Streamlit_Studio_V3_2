import io, zipfile, unittest
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize
from engine import *
from document_pipeline import _extract_zip

class EngineTests(unittest.TestCase):
    def test_gate_floor_and_absence(self):
        r=np.array([[0.,1.,1.],[1.,np.nan,.4]]);m=np.array([[1,1,1],[1,0,1]],bool)
        w=masked_gate(r,m,strength_for_floor(.1,3));self.assertTrue(np.all(w[m]>=.1-1e-12));self.assertTrue(np.all(w[~m]==0));np.testing.assert_allclose(w.sum(1),1)
    def test_all_missing(self):
        with self.assertRaises(ValueError):masked_gate(np.zeros((1,3)),np.zeros((1,3),bool),0)
    def test_sqrt_block(self):
        v=[np.array([[1.,0.]]),np.array([[0.,1.]])];m=np.ones((1,2),bool);r=np.array([[0.,1.]])
        z,a=fuse(v,m,r,.75);np.testing.assert_allclose(z,[[np.sqrt(a[0,0]),0,0,np.sqrt(a[0,1])]]);np.testing.assert_allclose(np.linalg.norm(z,axis=1),1)
        b,_=fuse(v,m,r,0,False);np.testing.assert_allclose(b,[[1,0,0,1]])
    def test_exact_view_loss(self):
        r=np.array([[0.,1.,.5]]);m=np.ones((1,3),bool);v=[np.ones((1,1))]*3;z,a=fuse(v,m,r,.75);new=m.copy();new[0,1]=False;zp,_=fuse(v,new,r,.75);e,q=dropout_eps(a,m,new);np.testing.assert_allclose(np.linalg.norm(z-zp,axis=1),e)
    def test_refit(self):
        for seed in [1,2,3]:
            rng=np.random.default_rng(seed);base=np.repeat([-1.,1.],10);v=[normalize(np.c_[base,rng.normal(0,.03,20)]) for _ in range(3)];r=rng.random((20,3));m=np.ones_like(r,bool);df=view_loss_audit(v,m,r,.75,2,seed,.3);self.assertFalse((df.certified & ~df.unchanged).any());self.assertTrue(df.dropped.any())
    def test_empty_affected_coverage(self):
        df=digits_demo(0,0,42);self.assertEqual(len(df),360);self.assertTrue(df.unchanged.all());self.assertEqual(coverage(df).iloc[1]['n'],0)
    def test_digits_live(self):
        df=digits_demo(.75,.1,42);self.assertTrue(df.dropped.any());self.assertFalse((df.certified & ~df.unchanged).any())
    def test_global(self):
        for scenario in ['Separated','Overlapping']:
            d=exact_demo(42,scenario,.2,.75);self.assertEqual(d['n_partitions'],127);self.assertTrue(not d['certified'] or d['optimal_unchanged'])
    def test_rotation(self):
        d=coordinate_demo(42,.75);self.assertLess(d['block_max_pairwise_change'],1e-12);self.assertGreater(d['unaligned_sum_max_pairwise_change'],.01)
    def test_document_pipeline(self):
        records,skip=_extract_zip(Path('data/demo_documents.zip'));self.assertEqual(len(records),12);self.assertEqual(skip,[])
        for strength in [0,.75]:
            res=cluster_documents(records,2,4,17,strength);self.assertEqual(len(res['details']),12);self.assertTrue(np.allclose(np.linalg.norm(res['embedding'],axis=1),1));self.assertTrue(np.all(res['mask'].sum(1)==1))
        df=view_loss_audit(res['views'],res['mask'],res['scores'],res['kappa'],res['best_k'],17,1,res['labels']);self.assertFalse(df.dropped.any())
    def test_unassignable_and_duplicate(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            z.writestr('a.txt','hello world');z.writestr('duplicate.txt','hello world');z.writestr('empty.txt','');z.writestr('../bad.txt','bad')
        records,skip=_extract_zip(buf);self.assertEqual(len(skip),2);rec,v,a,r,excluded=prepare_documents(records,17);self.assertEqual(len(rec),1);self.assertEqual(len(excluded),1)
    def test_fixed_mask_compositional(self):
        d=fixed_mask_demo(42,.75,.1,.1,.1);self.assertFalse((d.certified & ~d.unchanged).any());self.assertTrue((d.actual_drift<=d.budget+1e-9).all())
    def test_mixed_document_archive(self):
        records,skip=_extract_zip(Path('data/demo_mixed_documents.zip'));self.assertEqual(len(records),18)
        res=cluster_documents(records,2,4,17,.75);self.assertTrue((res['mask'].sum(1)>1).any())
        df=view_loss_audit(res['views'],res['mask'],res['scores'],res['kappa'],res['best_k'],17,1,res['labels']);self.assertTrue(df.dropped.any());self.assertFalse((df.certified & ~df.unchanged).any())
    def test_source_manifest(self):
        import json,hashlib
        for f,h in json.loads(Path('reference/archive_sha256.json').read_text()).items():self.assertEqual(hashlib.sha256(Path(f).read_bytes()).hexdigest(),h)
if __name__=='__main__':unittest.main()
