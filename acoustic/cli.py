"""python -m acoustic.cli --image image.png --outdir new_directory"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
from .pipeline import run


def sha256(path):
    digest = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path, mask=False):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f'Unreadable image: {path}')
    if not mask and a.ndim == 3:
        if a.shape[2] != 3:
            raise ValueError('Alpha channels are not supported')
        a = cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
    return a


def write(path, image):
    a = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    if not cv2.imwrite(str(path),a):
        raise OSError(f'Failed to save {path}')


def main():
    p = argparse.ArgumentParser(description='Selective research inpainting with abstention')
    p.add_argument('--image', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--mask', help='Optional candidate artifact mask; otherwise automatic detection')
    p.add_argument('--certified-artifact', help='Independent pixelwise artifact support, never lesion mask')
    p.add_argument('--protected', help='Lesion/anatomy exclusion mask')
    p.add_argument('--weights', help='FFC tensor checkpoint; omission uses explicitly labelled Telea acoustic baseline')
    args = p.parse_args()
    out = Path(args.outdir)
    if out.exists():
        p.error('Use a new output directory to preserve previous runs')
    im = read(args.image)
    optional = {k: read(v,True) if v else None for k,v in [('mask',args.mask),('certified_artifact',args.certified_artifact),('protected',args.protected)]}
    # Load only when inference can be attempted; no fallback weights.
    neural = None
    if args.weights and args.certified_artifact:
        from inferencia_ecografia import InpainterEcografia
        neural = InpainterEcografia(args.weights)
    result,mask,report = run(im,neural=neural,**optional)
    report['inputs'] = {k:{'path':str(Path(v).resolve()),'sha256':sha256(v)} for k,v in vars(args).items() if k!='outdir' and v}
    out.mkdir(parents=True)
    write(out/'result.png',result)
    write(out/'candidate_mask.png',mask.astype('uint8')*255)
    # Verify persisted image, not merely the in-memory compositor.
    if not (read(out/'result.png') == result).all():
        raise RuntimeError('Lossless round trip verification failed')
    report['output_sha256'] = sha256(out/'result.png')
    (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':report['status'],'reasons':report['reasons'],'outdir':str(out)}))


if __name__ == '__main__':
    main()
