import unittest
from pathlib import Path
import numpy as np
from acoustic.physics import half_max_width,analyze
from acoustic.detector_study import upper_binomial,risk_table
from acoustic.pipeline import detect,run
from acoustic.study_v6 import marks,letterbox
from acoustic.ablation_v6 import patient_mean_ci


class V6Tests(unittest.TestCase):
    def test_fwhm_gaussian_phantom(self):
        x=np.arange(-100,101)*0.01
        width=half_max_width(np.exp(-x*x/(2*0.1**2)),0.01)
        self.assertAlmostEqual(width,2*np.sqrt(2*np.log(2))*0.1,places=3)

    def test_png_not_linear_envelope(self):
        with self.assertRaises(ValueError):
            analyze(np.ones((32,32)),dict(representation='bmode_png',device='GE',transducer='unknown',
                axial_spacing_mm=0.1,lateral_spacing_mm=0.1,source_id='x'))

    def test_zero_errors_not_zero_risk(self):
        self.assertAlmostEqual(upper_binomial(0,10),1-0.05**0.1)
        self.assertGreater(upper_binomial(0,10),0.05)

    def test_empty_detector_not_eligible(self):
        prob=np.zeros((3,8,8));masks=np.zeros_like(prob,dtype=bool);masks[:,3,3]=True
        rows=[dict(patient_id=str(i),device='A') for i in range(3)]
        table=risk_table(prob,masks,rows,[0.9])
        self.assertFalse(table[0]['eligible'])

    def test_yellow_candidate_requires_certification(self):
        im=np.full((64,64,3),70,np.uint8)
        im[20:22,20:30]=[255,255,0]
        self.assertTrue(detect(im)[20:22,20:30].all())
        out,_,report=run(im)
        np.testing.assert_array_equal(out,im)
        self.assertEqual(report['status'],'abstained')

    def test_mark_support_exact_and_reproducible(self):
        im=np.full((128,128,3),80,np.uint8)
        marked,m=marks(im,7,1)
        np.testing.assert_array_equal(marked[~m],im[~m])
        np.testing.assert_array_equal(marked,marks(im,7,1)[0])

    def test_patient_bootstrap_does_not_weight_views(self):
        result=patient_mean_ci([0,0,0,10],['a','a','a','b'],repeats=50)
        self.assertEqual(result['mean_patient'],5)

    def test_reader_analysis_perfect_agreement(self):
        from acoustic.reader_analysis import analyze_study, fleiss_kappa
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            p_td = Path(td)
            key = [{"case_id": "c1", "variant": "reference", "patient_id": "p1", "original": "o1"},
                   {"case_id": "c2", "variant": "inpaint", "patient_id": "p2", "original": "o2"}]
            (p_td / "key.json").write_text(json.dumps(key), encoding="utf-8")
            r1 = [{"reader_id": "R1", "case_id": "c1", "manip": "No", "distortion": 0, "confidence": 5},
                  {"reader_id": "R1", "case_id": "c2", "manip": "Sí", "distortion": 1, "confidence": 4}]
            r2 = [{"reader_id": "R2", "case_id": "c1", "manip": "No", "distortion": 0, "confidence": 5},
                  {"reader_id": "R2", "case_id": "c2", "manip": "Sí", "distortion": 1, "confidence": 4}]
            (p_td / "r1.json").write_text(json.dumps(r1), encoding="utf-8")
            (p_td / "r2.json").write_text(json.dumps(r2), encoding="utf-8")
            res = analyze_study(p_td / "key.json", [p_td / "r1.json", p_td / "r2.json"])
            self.assertEqual(res["inter_observer_agreement"]["R1_vs_R2"]["cohen_kappa_manipulation"], 1.0)
            self.assertEqual(res["reader_individual_metrics"]["R1"]["sensitivity"], 1.0)
            self.assertEqual(res["reader_individual_metrics"]["R1"]["specificity"], 1.0)

    def test_fleiss_kappa_calculation(self):
        from acoustic.reader_analysis import fleiss_kappa
        # 3 raters, 2 subjects, all agree
        matrix = [[3, 0], [0, 3]]
        self.assertEqual(fleiss_kappa(matrix), 1.0)


if __name__=='__main__':unittest.main()

