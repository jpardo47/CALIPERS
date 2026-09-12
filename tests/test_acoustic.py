import unittest
import tempfile
from pathlib import Path
import numpy as np
from acoustic.metrics import measurements, binary, segmentation_metrics
from acoustic.pipeline import run, acoustic_candidate, Policy
from acoustic.benchmark import overlay
from acoustic.cad import attention_energy, paired_audit
from acoustic.cli import read,write
from acoustic.manifest import validate,split_patients


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(7).integers(30,180,(64,80,3),dtype=np.uint8)
        self.mask = np.zeros((64,80),bool)
        self.mask[20:25,20:30] = True

    def test_exact_exterior(self):
        result = acoustic_candidate(self.image,self.mask)
        self.assertTrue(np.array_equal(result[~self.mask],self.image[~self.mask]))

    def test_ambiguous_abstains(self):
        out,_,report = run(self.image,self.mask)
        self.assertEqual(report['status'],'abstained')
        np.testing.assert_array_equal(out,self.image)

    def test_protected_abstains_even_with_permissive_qc(self):
        out,_,r = run(self.image,self.mask,certified_artifact=self.mask,protected=self.mask)
        self.assertIn('protected_anatomy_intersection',r['reasons'])
        np.testing.assert_array_equal(out,self.image)

    def test_support_does_not_silently_clip(self):
        _,_,r = run(self.image,self.mask,certified_artifact=np.zeros_like(self.mask))
        self.assertIn('candidate_outside_certified_artifact',r['reasons'])

    def test_reject_resize_and_soft_masks(self):
        for mask in (np.zeros((2,2)),np.full((64,80),0.5)):
            with self.assertRaises(ValueError):
                binary(mask,(64,80))

    def test_pfpr_requires_independent_gt(self):
        result = self.image.copy()
        result[self.mask] = 0
        self.assertIsNone(measurements(self.image,result,self.mask)['pfpr_pct'])
        q = measurements(self.image,result,self.mask,artifact_gt=np.zeros_like(self.mask),tissue=np.ones_like(self.mask))
        self.assertGreater(q['pfpr_pct'],0)

    def test_local_psnr_not_diluted_by_exterior(self):
        result = self.image.copy()
        result[self.mask] = np.clip(self.image[self.mask].astype(int)+10,0,255)
        q = measurements(self.image,result,self.mask,clean=self.image)
        self.assertAlmostEqual(q['ppsnr_db'],20*np.log10(255/10),places=5)

    def test_empty_support_is_missing_not_perfect(self):
        q = measurements(self.image,self.image,np.zeros_like(self.mask),clean=self.image)
        self.assertIsNone(q['pssim'])
        self.assertIsNone(q['std_ratio'])

    def test_overlay_reproducible_and_supported(self):
        x,m = overlay(self.image,5)
        y,n = overlay(self.image,5)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(x[~m],self.image[~m])

    def test_png_exact(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x.png'
            write(p,self.image)
            np.testing.assert_array_equal(read(p),self.image)

    def test_segmentation_requires_actual_after(self):
        after = np.roll(self.mask,2,axis=0)
        q = segmentation_metrics(self.mask,after,spacing=(0.5,0.5))
        self.assertLess(q['dice'],1)
        self.assertGreater(q['hd95'],0)
        self.assertEqual(q['units'],'mm')

    def test_cam_zero_missing_and_uniform_enrichment(self):
        self.assertIsNone(attention_energy(np.zeros_like(self.mask),self.mask)['artifact_energy'])
        self.assertAlmostEqual(attention_energy(np.ones_like(self.mask),self.mask)['artifact_area_enrichment'],1)

    def test_patient_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'x.png'
            write(p,self.image)
            rows = [dict(image=str(p),patient_id='1',dataset='BUS',split=s) for s in ('train','test')]
            with self.assertRaisesRegex(ValueError,'Patient leakage'):
                validate(rows)

    def test_counterfactual_score_sensitivity(self):
        r = paired_audit([0,0,1,1],[0.1,0.2,0.8,0.9],[0.9,0.8,0.2,0.1],['a','b','c','d'],repeats=50)
        self.assertEqual(r['auc_delta_clean_minus_marked'],1)

    def test_accepted_path_preserves_exterior(self):
        # Permissive limits exercise acceptance, not clinical threshold calibration.
        out,_,r = run(self.image,self.mask,certified_artifact=self.mask,
                      policy=Policy(min_std_ratio=0,max_std_ratio=100,max_sobel_jump=10000))
        self.assertEqual(r['status'],'accepted_research')
        np.testing.assert_array_equal(out[~self.mask],self.image[~self.mask])

    def test_empty_and_full_masks(self):
        out,_,r = run(self.image,np.zeros_like(self.mask))
        self.assertEqual(r['status'],'unchanged')
        np.testing.assert_array_equal(out,self.image)
        _,_,r = run(self.image,np.ones_like(self.mask),certified_artifact=np.ones_like(self.mask))
        self.assertIn('excessive_mask_coverage',r['reasons'])


if __name__ == '__main__':
    unittest.main()
