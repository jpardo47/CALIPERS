"""Native-resolution held-out patient ablations with paired bootstrap intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .cli import read,write,sha256
from .study_v6 import marks
from .metrics import measurements


def patient_mean_ci(values,patients,seed=42,repeats=1000):
    groups=sorted(set(patients))
    means=np.array([np.mean([v for v,p in zip(values,patients) if p==g]) for g in groups])
    if not len(means):return None
    rng=np.random.default_rng(seed)
    samples=np.array([rng.choice(means,len(means),replace=True).mean() for _ in range(repeats)])
    return {'mean_patient':float(means.mean()),'ci95':np.percentile(samples,[2.5,97.5]).tolist(),'patients':len(groups)}


def evaluate(manifest,weights,outdir):
    out=Path(outdir)
    if out.exists():raise ValueError('New output directory required')
    rows=[r for r in json.loads(Path(manifest).read_text(encoding='utf-8')) if r['split']=='test']
    out.mkdir(parents=True)
    from inferencia_ecografia import InpainterEcografia
    torch.set_num_threads(4)
    cases=[]
    for i,row in enumerate(rows):
        ref=read(row['image'])
        if ref.ndim==2:ref=np.repeat(ref[...,None],3,axis=2)
        corrupted,m=marks(ref,71000+i,int(row['label']=='malignant'))
        lesion=read(row['lesion_mask'],True)
        cases.append((ref,corrupted,m,lesion))
        case=out/f'case_{i:03d}';case.mkdir()
        write(case/'reference.png',ref);write(case/'corrupted.png',corrupted);write(case/'mask.png',m.astype(np.uint8)*255)
    results=[]
    for name,path in weights.items():
        model=InpainterEcografia(path,'cpu')
        for i,(ref,corrupted,m,lesion) in enumerate(cases):
            pred=model.inpaint(corrupted,m)
            qc=measurements(corrupted,pred,m,clean=ref,artifact_gt=m,lesion=lesion)
            qc.update(method=name,patient_id=rows[i]['patient_id'],device=rows[i]['device'],case=i)
            results.append(qc);write(out/f'case_{i:03d}'/(name+'.png'),pred)
        del model
        print('Ablation completed: '+name,flush=True)
    summaries={}
    for name in weights:
        rs=[r for r in results if r['method']==name]
        summaries[name]={}
        for metric in ('ppsnr_db','pssim','std_ratio','sobel_jump','edge_l1'):
            valid=[r for r in rs if r[metric] is not None]
            summaries[name][metric]=patient_mean_ci([r[metric] for r in valid],[r['patient_id'] for r in valid])
    # Paired differences, not subtraction of independently bootstrapped means.
    differences={}
    for name in weights:
        if name=='places2':continue
        for metric in ('ppsnr_db','pssim'):
            delta=[];patients=[]
            for row in [r for r in results if r['method']==name]:
                ref=next(r for r in results if r['method']=='places2' and r['case']==row['case'])
                if row[metric] is not None and ref[metric] is not None:
                    delta.append(row[metric]-ref[metric]);patients.append(row['patient_id'])
            differences[name+'-'+metric]=patient_mean_ci(delta,patients)
    report={'scope':'heldout_patients_pilot_unverified_cleanliness','manifest_sha256':sha256(manifest),
            'weights':{k:sha256(v) for k,v in weights.items()},'summaries':summaries,
            'paired_difference_vs_places2':differences,'rows':results}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--places2',required=True)
    p.add_argument('--l1',required=True);p.add_argument('--acoustic',required=True);p.add_argument('--outdir',required=True)
    a=p.parse_args();evaluate(a.manifest,{'places2':a.places2,'l1':a.l1,'acoustic':a.acoustic},a.outdir)
