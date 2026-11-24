import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import engine
from document_pipeline import _extract_zip

class OptimizedTests(unittest.TestCase):
    def setUp(self):
        self.records,_=_extract_zip(Path('data/demo_mixed_documents.zip'),ocr=False)
    def test_no_ocr(self):
        self.assertTrue(all(r['ocr_status']=='disabled_by_user' for r in self.records if r['image_features'] is not None))
    def test_identical_branch_fits_once(self):
        with patch('engine.fit_lloyd',wraps=engine.fit_lloyd) as f:
            r=engine.cluster_documents(self.records,2,4,17,0)
        self.assertEqual(f.call_count,3);self.assertEqual(r['fit_count'],3);self.assertTrue(r['baseline_reused']);self.assertTrue((r['diagnostics'].baseline_partition_ARI==1).all())
    def test_exact_silhouette(self):
        r=engine.cluster_documents(self.records,2,4,17,.75)
        self.assertAlmostEqual(silhouette_score(r['embedding'],r['labels']),r['diagnostics'].set_index('K').loc[r['best_k'],'silhouette'],places=12)
    def test_prepared_cache(self):
        prepared=engine.prepare_documents(self.records,17)
        a=engine.cluster_documents(self.records,2,4,17,.75)
        with patch('engine.prepare_documents',side_effect=AssertionError('Unexpected preparation')):
            b=engine.cluster_documents(self.records,2,4,17,.75,prepared=prepared)
        np.testing.assert_array_equal(a['labels'],b['labels'])
    def test_quick_certificate(self):
        r=engine.cluster_documents(self.records,2,4,17,.75,profile='Quick')
        self.assertEqual(r['n_init'],3)
        d=engine.view_loss_audit(r['views'],r['mask'],r['scores'],r['kappa'],r['best_k'],17,1,r['labels'])
        self.assertFalse((d.certified & ~d.unchanged).any())
    def test_cdist_parity(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(70,45));c=rng.normal(size=(8,45))
        np.testing.assert_allclose(cdist(x,c),np.linalg.norm(x[:,None]-c[None,:],axis=2),rtol=1e-14)
    def test_quick_large_sample(self):
        rng=np.random.default_rng(4);n=650;v=[rng.normal(size=(n,4))];v=[v[0]/np.linalg.norm(v[0],axis=1)[:,None]]
        # Three blocks and complete masks for same engine contract.
        records=[{'filename':str(i),'document_id':str(i)} for i in range(n)];v=v*3;m=np.ones((n,3),bool);r=np.ones((n,3))*.5
        out=engine.cluster_documents(records,2,3,17,0,profile='Quick',prepared=(records,v,m,r,[]))
        self.assertTrue((out['diagnostics'].silhouette_rows==600).all())
