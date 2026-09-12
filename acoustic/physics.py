"""Measure linear-envelope statistics and isolated-target PSF with physical axes.

PNG B-mode intensities are deliberately rejected as acquisition envelopes.
"""
import argparse,json
from pathlib import Path
import numpy as np


def envelope_parameters(values):
    r=np.asarray(values,dtype=float)
    if r.size<32 or np.any(r<0) or not np.isfinite(r).all():
        raise ValueError('Need >=32 finite nonnegative linear envelope samples')
    power=r*r;omega=float(power.mean());variance=float(power.var())
    return {'nakagami_m_moments':omega**2/variance if variance>0 else None,
            'nakagami_omega':omega,'rayleigh_scale_mle':float(np.sqrt(omega/2)),
            'samples':r.size,'fit_certifies_model':False}


def half_max_width(profile,spacing):
    p=np.asarray(profile,dtype=float)
    if p.ndim!=1 or p.size<3 or not np.isfinite(p).all() or p.max()<=0 or spacing<=0:
        raise ValueError('Invalid PSF profile or spacing')
    peak=int(p.argmax());level=p[peak]*0.5
    left=peak;right=peak
    while left>0 and p[left]>=level:left-=1
    while right<len(p)-1 and p[right]>=level:right+=1
    if p[left]>=level or p[right]>=level:return None
    xl=left+(level-p[left])/(p[left+1]-p[left])
    xr=right-1+(level-p[right-1])/(p[right]-p[right-1])
    return float((xr-xl)*spacing)


def analyze(array,metadata):
    required=['representation','device','transducer','axial_spacing_mm','lateral_spacing_mm','source_id']
    if any(not metadata.get(k) for k in required):
        raise ValueError('Missing acquisition metadata: '+', '.join(required))
    kind=metadata['representation']
    if kind not in ('linear_envelope','complex_iq'):
        raise ValueError('Log B-mode PNG is not calibrated linear envelope or complex IQ')
    a=np.asarray(array)
    if kind=='complex_iq' and not np.iscomplexobj(a):
        raise ValueError('Complex IQ must contain complex samples')
    if a.ndim!=2:raise ValueError('Expected axial x lateral beamformed array')
    envelope=np.abs(a) if kind=='complex_iq' else a.astype(float)
    if not np.isfinite(envelope).all() or np.any(envelope<0):raise ValueError('Invalid envelope')
    result={'source_id':metadata['source_id'],'device':metadata['device'],'psf':None,'envelope':None}
    roi=metadata.get('homogeneous_roi_yxyx')
    if roi is not None:
        y0,x0,y1,x1=roi
        if not(0<=y0<y1<=a.shape[0] and 0<=x0<x1<=a.shape[1]):raise ValueError('ROI outside acquisition')
        result['envelope']=envelope_parameters(envelope[y0:y1,x0:x1])
    if metadata.get('isolated_point_target') is True:
        y,x=np.unravel_index(envelope.argmax(),envelope.shape)
        result['psf']={'axial_fwhm_mm':half_max_width(envelope[:,x],metadata['axial_spacing_mm']),
                       'lateral_fwhm_mm':half_max_width(envelope[y,:],metadata['lateral_spacing_mm']),
                       'definition':'half linear amplitude (-6.0206 dB); background must be removed upstream'}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--array',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    if Path(a.output).exists():p.error('Output exists')
    result=analyze(np.load(a.array,allow_pickle=False),json.loads(Path(a.metadata).read_text(encoding='utf-8')))
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
