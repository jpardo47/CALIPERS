"""Paired synthetic overlays, distinct oracle-mask and end-to-end experiments."""
import argparse
import csv
import json
import platform
from pathlib import Path
import cv2
import numpy as np
from .cli import read, write, sha256
from .metrics import measurements
from .pipeline import detect, acoustic_candidate, run


def overlay(clean, seed):
    """Antialiased '+'/'x', measurement dots and text; exact affected support."""
    rng = np.random.default_rng(seed)
    h,w = clean.shape[:2]
    if min(h,w)<64:
        raise ValueError('Synthetic overlay requires >=64 pixels per axis')
    alpha = np.zeros((h,w),np.uint8)
    g = cv2.cvtColor(clean,cv2.COLOR_RGB2GRAY) if clean.ndim==3 else clean
    visible = ((g>20)&(g<230)).astype(np.float32)
    # Geometric placement eligibility only, NOT a certified biological tissue mask.
    eligible = cv2.boxFilter(visible,-1,(31,31)) > 0.95
    eligible[:16] = eligible[-16:] = False
    eligible[:,:16] = eligible[:,-16:] = False
    ys,xs = np.where(eligible)
    if len(ys)<2:
        raise ValueError('Insufficient visible context for synthetic overlay')
    selected = rng.choice(len(ys),2,replace=False)
    pts = [(int(xs[i]),int(ys[i])) for i in selected]
    for idx,(x,y) in enumerate(pts):
        arm = int(rng.integers(5,11))
        if idx == 0:
            ends = [((x-arm,y),(x+arm,y)),((x,y-arm),(x,y+arm))]
        else:
            ends = [((x-arm,y-arm),(x+arm,y+arm)),((x-arm,y+arm),(x+arm,y-arm))]
        for p,q in ends:
            cv2.line(alpha,p,q,255,1,cv2.LINE_AA)
    for t in np.linspace(0,1,20):
        x,y = np.rint(np.array(pts[0])*(1-t)+np.array(pts[1])*t).astype(int)
        cv2.circle(alpha,(int(x),int(y)),0,220,-1)
    # Place text within a predominantly visible rectangle, not the black footer.
    text_context = cv2.boxFilter(visible,-1,(90,20)) > 0.95
    text_context[:20] = text_context[-20:] = False
    text_context[:,:46] = text_context[:,-46:] = False
    ty,tx = np.where(text_context)
    if len(ty):
        j = int(rng.integers(len(ty)))
        cv2.putText(alpha,'D1: 1.23cm',(int(tx[j])-42,int(ty[j])+5),cv2.FONT_HERSHEY_SIMPLEX,0.4,255,1,cv2.LINE_AA)
    a = alpha.astype(float)/255
    if clean.ndim == 3:
        a = a[...,None]
    corrupted = np.rint(clean*(1-a)+255*a).astype(np.uint8)
    return corrupted,alpha>0


def bbox(mask):
    # Union of component boxes; the convention is recorded in the report.
    out = np.zeros(mask.shape,np.uint8)
    n,_,info,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    for x,y,w,h,area in info[1:]:
        out[y:y+h,x:x+w] = 255
    return out>0


def evaluate(paths, outdir, weights=None, base_weights=None, seed=42):
    outdir = Path(outdir)
    if outdir.exists():
        raise ValueError('Benchmark output must be new')
    outdir.mkdir(parents=True)
    models = {}
    if weights or base_weights:
        import torch
        torch.set_num_threads(4)
        from inferencia_ecografia import InpainterEcografia
        if weights:
            models['ffc_finetuned'] = InpainterEcografia(weights,'cpu')
        if base_weights:
            models['lama_places2'] = InpainterEcografia(base_weights,'cpu')
    rows, cases = [], []
    for idx,path in enumerate(paths):
        clean = read(path)
        corrupted, gt = overlay(clean,seed+idx)
        predicted = detect(corrupted)
        # No human certification is manufactured: production is measured abstaining.
        safe,_,audit = run(corrupted)
        box = bbox(gt)
        candidates = {
            'telea_bbox': (cv2.inpaint(corrupted,box.astype(np.uint8)*255,3,cv2.INPAINT_TELEA), box),
            'telea_stroke': (cv2.inpaint(corrupted,gt.astype(np.uint8)*255,3,cv2.INPAINT_TELEA), gt),
            'acoustic_telea': (acoustic_candidate(corrupted,gt),gt),
            'detector_acoustic_telea': (acoustic_candidate(corrupted,predicted),predicted),
            'selective_production': (safe,predicted)}
        for name,model in models.items():
            candidates[name] = (model.inpaint(corrupted,gt.astype(np.uint8)*255),gt)
        if 'ffc_finetuned' in models:
            candidates['ffc_acoustic'] = (acoustic_candidate(corrupted,gt,models['ffc_finetuned']),gt)
        case_dir = outdir/f'case_{idx:03d}'
        case_dir.mkdir()
        write(case_dir/'reference.png',clean)
        write(case_dir/'corrupted.png',corrupted)
        write(case_dir/'artifact_gt.png',gt.astype(np.uint8)*255)
        write(case_dir/'detected.png',predicted.astype(np.uint8)*255)
        # Background/tissue is not annotated; use whole-frame non-overlay denominator,
        # explicitly named below, and keep clinical pFPR unavailable.
        for name,(res,support) in candidates.items():
            qc = measurements(corrupted,res,support,clean=clean,artifact_gt=gt)
            # Fair reconstruction comparison ALWAYS evaluated on common overlay GT.
            reconstruction = measurements(corrupted,res,gt,clean=clean)
            changed = np.any(corrupted!=res,axis=2) if res.ndim==3 else corrupted!=res
            qc.update({'image':Path(path).name,'method':name,'case':idx})
            for metric in ('ppsnr_db','pssim','perfect_reconstruction'):
                qc[metric] = reconstruction[metric]
            qc['outside_overlay_changed_pixels'] = int((changed & ~gt).sum())
            qc['nonoverlay_frame_change_pct'] = float(100*(changed&~gt).sum()/(~gt).sum())
            qc['tissue_gain_geometry_pct'] = float(100*(1-support.sum()/box.sum()))
            qc['status'] = audit['status'] if name == 'selective_production' else 'ungated_research'
            rows.append(qc)
            write(case_dir/(name+'.png'),res)
        cases.append({'path':str(Path(path).resolve()),'sha256':sha256(path),'seed':seed+idx})
        print(f'Completed {idx+1}/{len(paths)}: {Path(path).name}',flush=True)
    report = {'scope':'exploratory_existing_curated_images_not_independent_test',
              'clean_verification':'not_independently_verified',
              'training_overlap':'unknown_for_existing_checkpoint',
              'clinical_pfpr':'unavailable_without_tissue_and_artifact_annotations',
              'bbox_convention':'union_of_connected_component_boxes',
              'overlay_placement':'visible_context_v2_not_tissue_certification',
              'seed':seed,'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
              'weights':{k:{'path':str(Path(v).resolve()),'sha256':sha256(v)} for k,v in [('finetuned',weights),('places2',base_weights)] if v},
              'cases':cases,'rows':rows}
    (outdir/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    fields = [k for k in rows[0] if k not in ('glcm','acf_delta')] if rows else []
    with (outdir/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--limit',type=int,default=5)
    p.add_argument('--weights')
    p.add_argument('--base-weights')
    p.add_argument('--pattern',action='append',help='Glob prefix; repeat to sample multiple cohorts')
    a = p.parse_args()
    paths = []
    for pattern in a.pattern or ['*.png']:
        paths.extend(sorted(p for p in Path(a.images).glob(pattern) if '_mask' not in p.stem)[:a.limit])
    paths = list(dict.fromkeys(paths))
    if not paths:
        p.error('No images found')
    evaluate(paths,a.outdir,a.weights,a.base_weights)
