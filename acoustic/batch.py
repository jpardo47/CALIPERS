"""Unattended selective processing with per-case audit and a failure ledger.

Input JSON list: image, optional artifact_mask, certified_artifact, protected.
Paths are absolute or resolved relative to the input manifest file.
"""
import argparse
import csv
import json
from pathlib import Path
from .cli import read,write,sha256
from .pipeline import run


def process(manifest,outdir,weights=None):
    manifest = Path(manifest).resolve()
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(rows,list) or not rows:
        raise ValueError('Manifest must be a nonempty list')
    out = Path(outdir)
    if out.exists():
        raise ValueError('Output directory must be new')
    model = None
    if weights and any(r.get('certified_artifact') for r in rows):
        from inferencia_ecografia import InpainterEcografia
        model = InpainterEcografia(weights)
    out.mkdir(parents=True)
    ledger = []
    for idx,row in enumerate(rows):
        entry = {'case':idx,'image':row.get('image',''),'status':'error','reason':''}
        try:
            paths = {k:(manifest.parent/row[k]).resolve() for k in ('image','artifact_mask','certified_artifact','protected') if row.get(k)}
            image = read(paths['image'])
            kwargs = {k:read(paths[source],True) if source in paths else None for k,source in [('mask','artifact_mask'),('certified_artifact','certified_artifact'),('protected','protected')]}
            result,mask,audit = run(image,neural=model,**kwargs)
            case = out/f'case_{idx:06d}'
            case.mkdir()
            write(case/'result.png',result)
            write(case/'candidate_mask.png',mask.astype('uint8')*255)
            if not (read(case/'result.png')==result).all():
                raise RuntimeError('PNG round trip failed')
            audit['inputs'] = {k:{'path':str(v),'sha256':sha256(v)} for k,v in paths.items()}
            (case/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
            entry.update(status=audit['status'],reason=';'.join(audit['reasons']))
        except (ValueError,OSError,RuntimeError,KeyError) as exc:
            entry['reason'] = str(exc)
        ledger.append(entry)
        # Flush the ledger after every case so interruption leaves an audit trail.
        with (out/'ledger.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=['case','image','status','reason'])
            writer.writeheader()
            writer.writerows(ledger)
    summary = {'manifest_sha256':sha256(manifest),'images':len(rows),
               'weights_sha256':sha256(weights) if weights else None,
               'counts':{s:sum(r['status']==s for r in ledger) for s in ('accepted_research','abstained','unchanged','error')}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--weights')
    a = p.parse_args()
    summary = process(a.manifest,a.outdir,a.weights)
    print(json.dumps(summary))
    raise SystemExit(1 if summary['counts']['error'] else 0)
