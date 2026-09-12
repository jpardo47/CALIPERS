"""Selective inpainting: a candidate is exported only after explicit QC."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy import ndimage
from .metrics import binary, gray, ring_of, measurements


@dataclass(frozen=True)
class Policy:
    max_coverage: float = 0.08
    min_std_ratio: float = 0.80
    max_std_ratio: float = 1.20
    max_sobel_jump: float = 60.0
    protected_margin_px: int = 3
    min_ring_pixels: int = 32


def detect(image):
    """Heuristic candidates, NEVER a calibrated tissue/annotation classifier.

    Keep stroke support only; no expansion into surrounding anatomy.
    Includes bright crosses, thin measurement lines and aligned text groups.
    """
    g = gray(image).astype(np.uint8)
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((9,9), np.uint8))
    bright = ((g >= 220) & (top >= 40)).astype(np.uint8)
    core = cv2.erode((g >= 235).astype(np.uint8), np.ones((3,3), np.uint8))
    solid = cv2.dilate(core, np.ones((9,9), np.uint8)) > 0
    bright[solid] = 0
    h = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((1,5),np.uint8))
    v = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5,1),np.uint8))
    d1 = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.eye(5,dtype=np.uint8))
    d2 = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.fliplr(np.eye(5,dtype=np.uint8)))
    k = np.ones((3,3), np.uint8)
    seeds = (cv2.dilate(h,k)&cv2.dilate(v,k)) | (cv2.dilate(d1,k)&cv2.dilate(d2,k))
    clusters = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((2,11),np.uint8))
    n, labels, info, _ = cv2.connectedComponentsWithStats(clusters)
    for idx in range(1,n):
        x,y,w,hh,area = info[idx]
        if w >= 18 and 5 <= hh <= 24 and w/hh >= 1.5:
            seeds[labels == idx] = 1
    # Long, bright straight strokes; still ambiguous in tissue and require support.
    for kernel in (np.ones((1,21),np.uint8), np.ones((21,1),np.uint8)):
        seeds |= cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    n, labels = cv2.connectedComponents(bright)
    ids = np.unique(labels[seeds > 0])
    result = np.isin(labels, ids[ids>0]) & (bright>0)
    if image.ndim==3:
        rgb=image.astype(np.int16)
        # Yellow overlays are suppressed by grayscale-only thresholds. Candidate
        # support only: color is not a clinical certification of annotation.
        yellow=(rgb[...,0]>130)&(rgb[...,1]>100)&(rgb[...,2]<0.60*np.minimum(rgb[...,0],rgb[...,1]))
        n,yl,info,_=cv2.connectedComponentsWithStats(yellow.astype(np.uint8))
        for idx in range(1,n):
            if 2<=info[idx,cv2.CC_STAT_AREA]<=2000:
                result |= yl==idx
    return result


def acoustic_candidate(image, mask, neural=None):
    """FFC structure plus bounded local residual gain; no white noise injection.

    Correct only under-dispersed components, and keep all exterior pixels exact.
    This regularizer is a B-mode texture prior, not recovery of transducer PSF.
    """
    m = binary(mask, image.shape[:2])
    if not m.any():
        return image.copy()
    base = neural.inpaint(image, m.astype(np.uint8)*255) if neural else cv2.inpaint(image, m.astype(np.uint8)*255, 3, cv2.INPAINT_TELEA)
    g = gray(base).astype(np.float32)
    low = cv2.GaussianBlur(g, (0,0), 1.2)
    residual = g-low
    # base has already replaced annotation pixels before filtering: no bright leakage.
    adjusted = g.copy()
    labels, n = ndimage.label(m)
    for idx in range(1,n+1):
        comp = labels == idx
        ring = ring_of(comp) & ~m
        if comp.sum() < 8 or ring.sum() < 32:
            continue
        sp, sr = residual[comp].std(), residual[ring].std()
        if sp > 1e-6 and sr > sp:
            gain = min(1.5, float(sr/sp))
            adjusted[comp] = low[comp]+residual[comp]*gain
    delta = adjusted-g
    proposal = np.clip(np.rint(base.astype(float)+(delta[...,None] if base.ndim==3 else delta)),0,255).astype(np.uint8)
    result = image.copy()
    result[m] = proposal[m]
    return result


def run(image, mask=None, *, certified_artifact=None, protected=None, neural=None, policy=None):
    """No certified support => abstention, preserving original + candidate mask.

    certified_artifact is independent pixelwise annotation support, NOT a lesion
    segmentation. In unattended use it may come from an independently validated
    upstream detector. Supplying it is an explicit trust boundary.
    """
    policy = policy or Policy()
    gray(image)
    m = detect(image) if mask is None else binary(mask, image.shape[:2])
    report = {'policy': asdict(policy), 'status': 'abstained', 'reasons': [],
              'backend': 'ffc_acoustic' if neural else 'telea_acoustic',
              'candidate_pixels': int(m.sum()), 'clinical_validation': False}
    if not m.any():
        report.update(status='unchanged', reasons=['no_candidate_not_proof_of_cleanliness'])
        return image.copy(), m, report
    if certified_artifact is None:
        report['reasons'].append('independent_artifact_support_required')
    elif np.any(m & ~binary(certified_artifact,m.shape)):
        report['reasons'].append('candidate_outside_certified_artifact')
    if m.mean() > policy.max_coverage:
        report['reasons'].append('excessive_mask_coverage')
    if protected is not None:
        p = binary(protected,m.shape)
        if policy.protected_margin_px > 0:
            p = ndimage.binary_dilation(p,iterations=policy.protected_margin_px)
        if np.any(m&p):
            report['reasons'].append('protected_anatomy_intersection')
    if report['reasons']:
        return image.copy(), m, report
    candidate = acoustic_candidate(image,m,neural)
    qc = measurements(image,candidate,m)
    report['qc'] = qc
    # Each component must pass, so a large good patch cannot hide a small bad one.
    labels, n = ndimage.label(m)
    component_qc = []
    for idx in range(1,n+1):
        comp = labels == idx
        clean_ring = ring_of(comp)&~m
        q = measurements(image,candidate,comp,tissue=~m)
        component_qc.append(q)
        if clean_ring.sum() < policy.min_ring_pixels or q['std_ratio'] is None or q['sobel_jump'] is None:
            report['reasons'].append('insufficient_component_support')
        elif not policy.min_std_ratio <= q['std_ratio'] <= policy.max_std_ratio or q['sobel_jump'] >= policy.max_sobel_jump:
            report['reasons'].append('component_acoustic_qc_failed')
    report['components'] = component_qc
    if qc['outside_changed_pixels'] != 0:
        report['reasons'].append('outside_mask_changed')
    if report['reasons']:
        report['reasons'] = sorted(set(report['reasons']))
        return image.copy(), m, report
    report['status'] = 'accepted_research'
    return candidate,m,report
