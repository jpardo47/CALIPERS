"""Patient split and decoded-pixel leakage audit; dataset names never imply patient IDs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .cli import read


def validate(rows):
    required = {'image','patient_id','dataset','split'}
    patients, images = {}, {}
    for row in rows:
        if not required <= row.keys() or any(not row[k] for k in required):
            raise ValueError('Required: image, patient_id, dataset, split')
        if row['split'] not in ('train','val','test','external'):
            raise ValueError('Unknown split')
        key = (row['dataset'],row['patient_id'])
        if key in patients and patients[key] != row['split']:
            raise ValueError(f'Patient leakage: {key}')
        patients[key] = row['split']
        im = read(row['image'])
        digest = hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
        if digest in images:
            raise ValueError(f'Duplicate decoded image: {row["image"]}')
        images[digest] = row['split']
        row['pixel_sha256'] = digest
    return rows


def split_patients(rows, seed=42):
    groups = sorted({(r['dataset'],r['patient_id']) for r in rows})
    if len(groups) < 3:
        raise ValueError('At least three patient groups required')
    order = np.random.default_rng(seed).permutation(len(groups))
    ntest = max(1,int(len(groups)*0.15))
    nval = max(1,int(len(groups)*0.15))
    assignments = {groups[j]: ('test' if i<ntest else 'val' if i<ntest+nval else 'train') for i,j in enumerate(order)}
    return [dict(r,split=assignments[(r['dataset'],r['patient_id'])]) for r in rows]


def bus_bra(metadata, images, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Output already exists')
    with open(metadata,encoding='utf-8-sig',newline='') as f:
        rows = [{'image':str((Path(images)/(r['ID']+'.png')).resolve()),
                 'patient_id':r['Case'],'dataset':'BUS_BRA','label':r['Pathology'],
                 'device':r['Device'],'clean_verified':False} for r in csv.DictReader(f)]
    rows = validate(split_patients(rows))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--metadata',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    rows = bus_bra(a.metadata,a.images,a.output)
    print({s:sum(r['split']==s for r in rows) for s in ('train','val','test')})
