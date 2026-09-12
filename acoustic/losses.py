"""Optional differentiable B-mode constraints for future FFC fine-tuning.

These losses are NOT retroactively attributed to existing checkpoints.
Inputs B,C,H,W in [0,1], mask B,1,H,W; targets are clean paired images.
"""
import torch
import torch.nn.functional as F


def physics_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != (target.shape[0],1,*target.shape[2:]):
        raise ValueError('Loss input shape mismatch')
    if not torch.all((mask==0)|(mask==1)):
        raise ValueError('Binary mask required')
    m = mask.expand_as(target)
    count = m.sum(dim=(-2,-1)).clamp_min(1)
    error = ((prediction-target).abs()*m).sum()/m.sum().clamp_min(1)
    def std(x):
        mean = (x*m).sum(dim=(-2,-1))/count
        return (((x-mean[...,None,None])**2*m).sum(dim=(-2,-1))/count+1e-8).sqrt()
    variance = (std(prediction)-std(target)).abs().mean()
    edge = prediction.sum()*0
    autocorr = prediction.sum()*0
    for axis in (-2,-1):
        dp,dt = torch.diff(prediction,dim=axis),torch.diff(target,dim=axis)
        if axis==-2:
            support = torch.maximum(m[...,1:,:],m[...,:-1,:])
        else:
            support = torch.maximum(m[...,1:],m[...,:-1])
        edge = edge+((dp-dt).abs()*support).sum()/support.sum().clamp_min(1)
        for lag in (1,2,3):
            if prediction.shape[axis]<=lag:
                continue
            s,t = [slice(None)]*4,[slice(None)]*4
            s[axis],t[axis] = slice(None,-lag),slice(lag,None)
            s,t = tuple(s),tuple(t)
            pairs = m[s]*m[t]
            # Centered pair covariance on the same locations in GT and prediction.
            def cov(x):
                n = pairs.sum().clamp_min(1)
                u,v = x[s],x[t]
                return (u*v*pairs).sum()/n-(u*pairs).sum()*(v*pairs).sum()/n**2
            autocorr = autocorr+(cov(prediction)-cov(target)).abs()
    return {'masked_l1':error,'std_l1':variance,'edge_l1':edge/2,'covariance_l1':autocorr/6}
