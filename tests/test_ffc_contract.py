import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from acoustic.losses import physics_loss
from inferencia_ecografia import InpainterEcografia,load_weights


class Dummy(torch.nn.Module):
    def forward(self,x):
        return torch.zeros_like(x[:,:3])+0.4


class FFCContractTests(unittest.TestCase):
    def test_nonfinite_prediction_is_rejected(self):
        class Broken(torch.nn.Module):
            def forward(self,x):
                return x[:,:3]*float('nan')
        model = InpainterEcografia.__new__(InpainterEcografia)
        model.device = torch.device('cpu')
        model.model = Broken()
        im = np.zeros((24,24),np.uint8)
        mask = np.zeros((24,24),bool)
        mask[5:8,5:8] = True
        with self.assertRaisesRegex(RuntimeError,'FFC'):
            model.inpaint(im,mask)

    def test_gray_rgb_boolean_and_exact_exterior(self):
        model = InpainterEcografia.__new__(InpainterEcografia)
        model.device = torch.device('cpu')
        model.model = Dummy()
        for shape in ((23,29),(23,29,3)):
            im = np.random.default_rng(1).integers(0,256,shape,dtype=np.uint8)
            m = np.zeros(shape[:2],bool)
            m[4:8,5:9] = True
            result = model.inpaint(im,m)
            self.assertEqual(result.shape,im.shape)
            np.testing.assert_array_equal(result[~m],im[~m])
            np.testing.assert_array_equal(model.inpaint(im,np.zeros_like(m)),im)

    def test_incomplete_weights_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'weights.pt'
            torch.save({},p)
            with self.assertRaises(RuntimeError):
                load_weights(torch.nn.Linear(2,1),str(p),torch.device('cpu'))

    def test_losses_have_finite_gradients(self):
        torch.manual_seed(1)
        x = torch.rand(1,3,16,16,requires_grad=True)
        y = torch.rand_like(x)
        mask = torch.zeros(1,1,16,16)
        mask[:,:,4:10,4:10] = 1
        loss = sum(physics_loss(x,y,mask).values())
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0)
        zeros = physics_loss(y,y,mask)
        self.assertLess(sum(float(z) for z in zeros.values()),1e-7)


if __name__ == '__main__':
    unittest.main()
