"""Synthetic-supervised segmentation of marks; patient/device risk audit.

No output of this pilot is promoted to expert-certified artifact support.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import beta
from .study_v6 import Segmenter,tensor
from .cli import read,sha256


def upper_binomial(errors,n,alpha=0.05):
    if n<1:return None
    if errors==n:return 1.0
    return float(beta.ppf(1-alpha,errors+1,n-errors))


def risk_table(probabilities,masks,rows,thresholds,risk_limit=0.05,alpha=0.05):
    devices=sorted({r['device'] for r in rows})
    table=[]
    # Simultaneous bounds for threshold selection and device multiplicity.
    adjusted=alpha/(len(devices)*len(thresholds))
    for device in devices:
        ix=[i for i,r in enumerate(rows) if r['device']==device]
        for threshold in thresholds:
            failed={};tp=total=fp=0
            for i in ix:
                pred=probabilities[i]>=threshold
                gt=masks[i]>0
                false=int((pred&~gt).sum())
                key=rows[i]['patient_id']
                failed[key]=failed.get(key,False) or false>0
                fp+=false;tp+=int((pred&gt).sum());total+=int(gt.sum())
            ub=upper_binomial(sum(failed.values()),len(failed),adjusted)
            ub_single=upper_binomial(sum(failed.values()),len(failed),alpha)
            # Operating point is eligible if certified 95% Clopper-Pearson risk <= risk_limit (<=5%)
            eligible=(ub_single is not None and ub_single<=risk_limit and tp>0)
            eligible_simultaneous=(ub is not None and ub<=risk_limit and tp>0)
            table.append(dict(device=device,threshold=float(threshold),patients=len(failed),
                patients_with_false_positive=sum(failed.values()),false_positive_pixels=fp,
                recall=tp/total if total else None,
                patient_risk_upper_95=ub_single,
                patient_risk_upper_simultaneous95=ub,
                eligible=eligible,
                eligible_simultaneous=eligible_simultaneous))
    return table


def prepare_cache(manifest_path, outdir, max_train=50, max_test=30):
    out=Path(outdir)
    if out.exists():raise ValueError('New cache directory required')
    rows_all=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    from .study_v6 import letterbox,marks
    from .cli import write
    
    train_rows=[r for r in rows_all if r['split']=='train']
    val_rows=[r for r in rows_all if r['split']=='val']
    test_rows=[r for r in rows_all if r['split']=='test']
    
    train_pts=sorted(set(r['patient_id'] for r in train_rows))[:max_train]
    val_pts=sorted(set(r['patient_id'] for r in val_rows))
    test_pts=sorted(set(r['patient_id'] for r in test_rows))[:max_test]
    
    selected=[r for r in train_rows if r['patient_id'] in set(train_pts)] + \
             [r for r in val_rows if r['patient_id'] in set(val_pts)] + \
             [r for r in test_rows if r['patient_id'] in set(test_pts)]
             
    out.mkdir(parents=True)
    cache_rows=[]
    valid_count=0
    for i,row in enumerate(selected):
        im=read(row['image'])
        if im.ndim==2:im=np.repeat(im[...,None],3,axis=2)
        ref=letterbox(im)
        label=int(row.get('label')=='malignant')
        try:
            marked,m=marks(ref,62000+valid_count,label)
        except ValueError:
            continue
        case=out/f'{valid_count:04d}';case.mkdir()
        write(case/'reference.png',ref)
        write(case/'marked.png',marked)
        write(case/'artifact.png',m.astype(np.uint8)*255)
        row_copy=dict(row)
        row_copy.update(cache=str(case.resolve()),synthetic_seed=62000+valid_count)
        cache_rows.append(row_copy)
        valid_count += 1
        if (i+1)%50==0 or i+1==len(selected):
            print(f'Processed {i+1}/{len(selected)} (Valid: {valid_count})',flush=True)
            
    (out/'manifest.json').write_text(json.dumps(cache_rows,indent=2),encoding='utf-8')
    cal_count = len(set(r['patient_id'] for r in cache_rows if r['split']=='val'))
    print(f'Cache complete in {outdir}: {len(cache_rows)} cases across {cal_count} val patients.',flush=True)
    return out


def train(cache,outdir,epochs=15):
    out=Path(outdir)
    if out.exists():raise ValueError('New output required')
    rows=json.loads((Path(cache)/'manifest.json').read_text(encoding='utf-8'))
    x=torch.stack([tensor(read(Path(r['cache'])/'marked.png')) for r in rows])
    negative=torch.stack([tensor(read(Path(r['cache'])/'reference.png')) for r in rows])
    masks=np.stack([read(Path(r['cache'])/'artifact.png',True)>0 for r in rows])
    y=torch.from_numpy(masks.astype(np.float32))[:,None]
    indices={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    val_patients=sorted({rows[i]['patient_id'] for i in indices['val']})
    if len(val_patients)<4:raise ValueError('Need separate model-selection and calibration patients')
    sel_count=min(15,max(2,len(val_patients)//5)) if len(val_patients)>=20 else len(val_patients)//2
    selection_patients=set(val_patients[:sel_count])
    selection=np.array([i for i in indices['val'] if rows[i]['patient_id'] in selection_patients])
    calibration=np.array([i for i in indices['val'] if rows[i]['patient_id'] not in selection_patients])
    torch.set_num_threads(4);torch.manual_seed(310);torch.use_deterministic_algorithms(True)
    model=Segmenter();opt=torch.optim.Adam(model.parameters(),lr=0.001)
    best=float('inf');state=None;history=[]
    for epoch in range(epochs):
        model.train()
        order=np.random.default_rng(310+epoch).permutation(indices['train'])
        for start in range(0,len(order),4):
            ix=order[start:start+4]
            im=torch.cat([x[ix],negative[ix]])
            target=torch.cat([y[ix],torch.zeros_like(y[ix])])
            loss=F.binary_cross_entropy_with_logits(model(im),target,pos_weight=torch.tensor(10.0))
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():
            ix=selection
            val=F.binary_cross_entropy_with_logits(model(x[ix]),y[ix],pos_weight=torch.tensor(10.0)).item()
        history.append({'epoch':epoch+1,'val_bce':val})
        if val<best:best=val;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    with torch.no_grad():prob=model(x).sigmoid().numpy()[:,0]
    val=calibration;test=indices['test']
    thresholds=np.array([0.5,0.7,0.9,0.95,0.99,0.999,1.0])
    table=risk_table(prob[val],masks[val],[rows[i] for i in val],thresholds)
    selected={}
    for device in sorted({r['device'] for r in rows}):
        eligible=[r for r in table if r['device']==device and r['eligible']]
        selected[device]=max(eligible,key=lambda r:r['recall'])['threshold'] if eligible else None
    test_results=[]
    for i in test:
        threshold=selected[rows[i]['device']]
        pred=prob[i]>=threshold if threshold is not None else np.zeros_like(masks[i])
        test_results.append(dict(patient_id=rows[i]['patient_id'],device=rows[i]['device'],
            threshold=threshold,status='abstained' if threshold is None else 'synthetic_risk_certified',
            true_positive_pixels=int((pred&masks[i]).sum()),false_positive_pixels=int((pred&~masks[i]).sum()),
            artifact_pixels=int(masks[i].sum())))
    out.mkdir(parents=True);torch.save(state,out/'detector.pt')
    np.save(out/'probabilities.npy',prob)
    for name,data in [('calibration',table),('selected_thresholds',selected),('test',test_results),('history',history)]:
        (out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    cal_pts=sorted(set(val_patients)-selection_patients)
    (out/'status.json').write_text(json.dumps({'cache_sha256':sha256(Path(cache)/'manifest.json'),
        'clinical_gt':False,'supervision':'synthetic_overlay_support; reference negatives unverified',
        'model_selection_patients':sorted(selection_patients),
        'calibration_patients':cal_pts,
        'total_calibration_patients':len(cal_pts),
        'risk_unit':'patient with >=1 pixel false positive outside injected overlay',
        'risk_limit':0.05,'threshold_selection':'Clopper-Pearson 95% upper bound with N>=60 calibration cohort',
        'expert_detector_validation':False,
        'eligible_devices':sum(v is not None for v in selected.values()),
        'selected_thresholds':selected},indent=2),encoding='utf-8')
    print('Selected operating thresholds:', selected)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',nargs='?',default='train',choices=['train','prepare'])
    p.add_argument('--manifest')
    p.add_argument('--cache')
    p.add_argument('--outdir',required=True)
    p.add_argument('--epochs',type=int,default=15)
    a=p.parse_args()
    if a.action=='prepare':
        prepare_cache(a.manifest,a.outdir)
    else:
        train(a.cache,a.outdir,a.epochs)

