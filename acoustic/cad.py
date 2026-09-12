"""Paired counterfactual CAD audit. Grad-CAM alone cannot prove no shortcuts."""
import numpy as np
from sklearn.metrics import roc_auc_score


def attention_energy(cam, artifact, lesion=None):
    cam = np.asarray(cam,dtype=float)
    artifact = np.asarray(artifact,dtype=bool)
    if cam.shape != artifact.shape or not np.isfinite(cam).all() or np.any(cam<0):
        raise ValueError('CAM must be nonnegative, finite and registered to masks')
    if cam.sum() == 0:
        return {'artifact_energy':None,'artifact_area_enrichment':None,'lesion_energy':None}
    fraction = float(cam[artifact].sum()/cam.sum())
    if lesion is not None and np.asarray(lesion).shape != cam.shape:
        raise ValueError('Lesion shape mismatch')
    return {'artifact_energy':fraction,
            'artifact_area_enrichment':fraction/float(artifact.mean()) if artifact.any() else None,
            'lesion_energy':float(cam[np.asarray(lesion,dtype=bool)].sum()/cam.sum()) if lesion is not None else None}


def paired_audit(labels, clean_scores, marked_scores, patient_ids, seed=42, repeats=2000):
    y,c,m = [np.asarray(v) for v in (labels,clean_scores,marked_scores)]
    ids = np.asarray(patient_ids)
    if not (y.ndim==c.ndim==m.ndim==ids.ndim==1 and len(y)==len(c)==len(m)==len(ids)):
        raise ValueError('Paired arrays must have equal length')
    if set(np.unique(y)) != {0,1} or not np.isfinite(c).all() or not np.isfinite(m).all():
        raise ValueError('Both binary labels and finite scores required')
    groups = np.unique(ids)
    if len(groups)<2:
        raise ValueError('At least two patient groups required')
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        ix = np.concatenate([np.flatnonzero(ids==g) for g in rng.choice(groups,len(groups),replace=True)])
        if len(np.unique(y[ix])) == 2:
            deltas.append(roc_auc_score(y[ix],c[ix])-roc_auc_score(y[ix],m[ix]))
    return {'auc_clean':float(roc_auc_score(y,c)), 'auc_marked':float(roc_auc_score(y,m)),
            'auc_delta_clean_minus_marked':float(roc_auc_score(y,c)-roc_auc_score(y,m)),
            'delta_ci95_patient_bootstrap':np.percentile(deltas,[2.5,97.5]).tolist() if deltas else None,
            'mean_abs_score_change':float(np.abs(c-m).mean()),'valid_bootstraps':len(deltas),
            'patients':len(groups),'images':len(y),'shortcut_eradication_proven':False}


if __name__ == '__main__':
    import argparse, csv, json
    from pathlib import Path
    p = argparse.ArgumentParser(description='Audit paired predictions from a frozen CAD model')
    p.add_argument('--predictions',required=True,help='CSV: label,clean_score,marked_score,patient_id')
    p.add_argument('--output',required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('Output exists')
    with open(a.predictions,newline='',encoding='utf-8') as f:
        r = list(csv.DictReader(f))
    report = paired_audit([int(x['label']) for x in r],[float(x['clean_score']) for x in r],
                          [float(x['marked_score']) for x in r],[x['patient_id'] for x in r])
    Path(a.output).write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
