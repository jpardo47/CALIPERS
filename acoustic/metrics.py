"""Explicit support, units and missingness for B-mode image measurements."""
import cv2
import numpy as np
from scipy import ndimage, stats
from skimage.metrics import structural_similarity


def binary(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != tuple(shape) or not np.isfinite(mask).all():
        raise ValueError('Mask must be finite and match image H,W exactly')
    if not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Mask must be binary: 0/1 or 0/255')
    return mask != 0


def gray(image):
    a = np.asarray(image)
    if a.dtype != np.uint8 or a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 3):
        raise ValueError('Expected uint8 grayscale or RGB; no implicit intensity scaling')
    return a.astype(float) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(float)


def ring_of(mask, radius=5):
    return ndimage.binary_dilation(mask, iterations=radius) & ~mask


def glcm_features(image, support, distance=1, levels=16):
    """Only count pairs with BOTH endpoints in support, four directions."""
    q = np.minimum((image * levels / 256).astype(int), levels - 1)
    counts = np.zeros((levels, levels), float)
    h, w = q.shape
    for dy, dx in [(0, distance), (distance, 0), (distance, distance), (distance, -distance)]:
        y0, y1 = max(0, -dy), min(h, h-dy)
        x0, x1 = max(0, -dx), min(w, w-dx)
        a = (slice(y0, y1), slice(x0, x1))
        b = (slice(y0+dy, y1+dy), slice(x0+dx, x1+dx))
        valid = support[a] & support[b]
        np.add.at(counts, (q[a][valid], q[b][valid]), 1)
    if counts.sum() < 8:
        return None
    counts += counts.T.copy()
    p = counts / counts.sum()
    i, j = np.indices(p.shape)
    mu = float((p*i).sum())
    variance = float((p*(i-mu)**2).sum())
    return {'contrast': float((p*(i-j)**2).sum()),
            'homogeneity': float((p/(1+(i-j)**2)).sum()),
            'entropy': float(-(p[p>0]*np.log2(p[p>0])).sum()),
            'correlation': float((p*(i-mu)*(j-mu)).sum()/variance) if variance > 0 else None}


def measurements(original, restored, mask, *, clean=None, artifact_gt=None, tissue=None, lesion=None):
    a, b = gray(original), gray(restored)
    if original.shape != restored.shape:
        raise ValueError('Restoration changed shape/channels')
    m = binary(mask, a.shape)
    changed = np.any(original != restored, axis=2) if original.ndim == 3 else original != restored
    r = ring_of(m)
    if tissue is not None:
        r &= binary(tissue, a.shape)
    if lesion is not None:
        r &= ~binary(lesion, a.shape)
    # Exclude all annotations, including detector false negatives, when GT exists.
    if artifact_gt is not None:
        r &= ~binary(artifact_gt, a.shape)
    out = {'mask_pixels': int(m.sum()), 'outside_changed_pixels': int((changed & ~m).sum()),
           'pfpr_pct': None, 'detector_fpr_pct': None, 'artifact_recall': None}
    if artifact_gt is not None:
        gt = binary(artifact_gt, a.shape)
        out['artifact_recall'] = float((m & gt).sum()/gt.sum()) if gt.any() else None
        if tissue is not None:
            biological = binary(tissue, a.shape) & ~gt
            if biological.any():
                out['pfpr_pct'] = float(100*(changed & biological).sum()/biological.sum())
                out['detector_fpr_pct'] = float(100*(m & biological).sum()/biological.sum())
    out.update({'std_ratio': None, 'variance_ratio': None, 'ks_statistic': None,
                'ks_p_iid_descriptive': None, 'sobel_jump': None, 'glcm': {}, 'acf_delta': {}})
    if m.sum() >= 8 and r.sum() >= 8:
        sp, sr = b[m].std(), a[r].std()
        if sr > 0:
            out['std_ratio'] = float(sp/sr)
            out['variance_ratio'] = float((sp/sr)**2)
        ks = stats.ks_2samp(b[m], a[r], method='asymp')
        out['ks_statistic'], out['ks_p_iid_descriptive'] = float(ks.statistic), float(ks.pvalue)
        for d in (1, 2, 3):
            p, q = glcm_features(b, m, d), glcm_features(a, r, d)
            out['glcm'][str(d)] = {k: abs(p[k]-q[k]) if p[k] is not None and q[k] is not None else None for k in p} if p and q else None
            # Directional pair correlation is a B-mode texture proxy, not measured PSF.
            for axis in (0, 1):
                def acf(im, support):
                    sl1, sl2 = [slice(None)]*2, [slice(None)]*2
                    sl1[axis], sl2[axis] = slice(None, -d), slice(d, None)
                    sl1, sl2 = tuple(sl1), tuple(sl2)
                    valid = support[sl1] & support[sl2]
                    x, y = im[sl1][valid], im[sl2][valid]
                    return float(np.corrcoef(x, y)[0, 1]) if len(x)>=8 and x.std()>0 and y.std()>0 else None
                x, y = acf(b, m), acf(a, r)
                out['acf_delta'][f'{axis}:{d}'] = abs(x-y) if x is not None and y is not None else None
    # Sobel 3x3, unnormalized, intensity range 0..255. Adjacent boundary pairs.
    gx = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    jumps = []
    for axis in (0, 1):
        s1, s2 = [slice(None)]*2, [slice(None)]*2
        s1[axis], s2[axis] = slice(None, -1), slice(1, None)
        s1, s2 = tuple(s1), tuple(s2)
        boundary = m[s1] != m[s2]
        jumps.extend(np.abs(magnitude[s1]-magnitude[s2])[boundary].tolist())
    out['sobel_jump'] = float(np.mean(jumps)) if jumps else None
    out.update({'ppsnr_db': None, 'pssim': None, 'edge_l1': None, 'perfect_reconstruction': False})
    if clean is not None:
        c = gray(clean)
        if c.shape != a.shape:
            raise ValueError('Clean GT shape mismatch')
        if m.any():
            mse = float(np.mean((c[m]-b[m])**2))
            out['perfect_reconstruction'] = mse == 0
            out['ppsnr_db'] = float(10*np.log10(255**2/mse)) if mse > 0 else None
            if min(c.shape) >= 11:
                _, smap = structural_similarity(c, b, data_range=255, gaussian_weights=True,
                                                sigma=1.5, use_sample_covariance=False, full=True)
                out['pssim'] = float(smap[m].mean())
            if lesion is not None:
                l = binary(lesion, a.shape)
                edge = (ndimage.binary_dilation(l) ^ ndimage.binary_erosion(l)) & m
                if edge.any():
                    cx = cv2.Sobel(c, cv2.CV_64F, 1, 0, ksize=3)
                    cy = cv2.Sobel(c, cv2.CV_64F, 0, 1, ksize=3)
                    out['edge_l1'] = float((np.abs(gx-cx)+np.abs(gy-cy))[edge].mean())
    return out


def segmentation_metrics(before, after, spacing=None):
    x = np.asarray(before) != 0
    y = binary(after, x.shape)
    if not x.any() or not y.any():
        return {'dice': 1.0 if not x.any() and not y.any() else 0.0, 'hd95': None, 'units': 'mm' if spacing else 'px'}
    bx = x ^ ndimage.binary_erosion(x)
    by = y ^ ndimage.binary_erosion(y)
    d = np.r_[ndimage.distance_transform_edt(~bx, sampling=spacing)[by],
              ndimage.distance_transform_edt(~by, sampling=spacing)[bx]]
    return {'dice': float(2*(x&y).sum()/(x.sum()+y.sum())), 'hd95': float(np.percentile(d,95)), 'units': 'mm' if spacing else 'px'}
