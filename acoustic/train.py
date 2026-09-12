"""Reproducible FFC fine-tuning on independently verified clean patient splits.

Example: python -m acoustic.train --manifest clean.json --weights generator.pt
         --outdir experiment_new --epochs 10 --device cuda
No test/external images contribute to optimization or model selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .manifest import validate
from .cli import read, sha256
from .benchmark import overlay
from .losses import physics_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--weights',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--crop',type=int,default=256)
    p.add_argument('--exploratory-unverified',action='store_true',help='Explicit nonclinical pilot; does not certify clean GT')
    p.add_argument('--loss-preset',choices=['l1','acoustic'],default='acoustic')
    p.add_argument('--freeze-bn',action=argparse.BooleanOptionalAction,default=True,
                   help='Preserve pretrained BatchNorm population statistics for small batches')
    a = p.parse_args()
    if a.crop<64 or a.crop%8 or a.epochs<1:
        p.error('crop >=64 and multiple of 8; epochs >=1')
    out = Path(a.outdir)
    if out.exists():
        p.error('Use a new experiment directory')
    rows = validate(json.loads(Path(a.manifest).read_text(encoding='utf-8')))
    subsets = {s:[r for r in rows if r['split']==s] for s in ('train','val')}
    if not all(subsets.values()):
        p.error('Manifest must contain train and val patients')
    if not a.exploratory_unverified and any(r.get('clean_verified') is not True for s in subsets.values() for r in s):
        p.error('Training/validation images need independent clean_verified=true')
    # Native crops preserve pixel scale; no implicit distortion of PSF/morphometry.
    for subset in subsets.values():
        for row in subset:
            if min(read(row['image']).shape[:2])<a.crop:
                p.error(f'Image too small for native crop: {row["image"]}')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    from inferencia_ecografia import build_generator,load_weights
    device = torch.device(a.device)
    model = build_generator(device)
    load_weights(model,a.weights,device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    out.mkdir(parents=True)
    provenance = vars(a)|{'weights_sha256':sha256(a.weights),'manifest_sha256':sha256(a.manifest),
                         'torch':torch.__version__,'loss_weights':{'masked_l1':1,'std_l1':0.2,'edge_l1':0.1,'covariance_l1':0.2}}
    provenance['scientific_status'] = 'exploratory_unverified_cleanliness' if a.exploratory_unverified else 'reviewed_training_data_not_clinical_validation'
    if a.loss_preset=='l1':
        provenance['loss_weights'] = {'masked_l1':1,'std_l1':0,'edge_l1':0,'covariance_l1':0}
    (out/'config.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    history,best = [],float('inf')
    for epoch in range(-1,a.epochs):
        log = {'epoch':epoch+1}
        for split,subset in subsets.items():
            training = split=='train'
            if epoch==-1 and training:
                continue
            model.train(training)
            if a.freeze_bn:
                for module in model.modules():
                    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
            # Same validation masks/crops for every epoch; train order changes reproducibly.
            order = np.random.default_rng(a.seed+epoch).permutation(len(subset)) if training else range(len(subset))
            values = []
            for idx in order:
                seed = a.seed+int(idx)+(epoch*len(subset) if training else 1000000)
                rng = np.random.default_rng(seed)
                im = read(subset[idx]['image'])
                if im.ndim==2:
                    im = np.repeat(im[...,None],3,axis=2)
                h,w = im.shape[:2]
                for attempt in range(32):
                    y,x = int(rng.integers(h-a.crop+1)),int(rng.integers(w-a.crop+1))
                    crop = im[y:y+a.crop,x:x+a.crop]
                    try:
                        _,mask = overlay(crop,seed)
                        break
                    except ValueError:
                        if attempt==31:
                            raise ValueError(f'No suitable crop for {subset[idx]["image"]}')
                im = crop
                target = torch.from_numpy(im.astype(np.float32)/255).permute(2,0,1)[None].to(device)
                m = torch.from_numpy(mask.astype(np.float32))[None,None].to(device)
                with torch.set_grad_enabled(training):
                    predicted = model(torch.cat([target*(1-m),m],dim=1))
                    losses = physics_loss(predicted,target,m)
                    loss = sum(losses[k]*weight for k,weight in provenance['loss_weights'].items())
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                        optimizer.step()
                values.append(float(loss.detach().cpu()))
            log[split+'_loss'] = float(np.mean(values))
        history.append(log)
        if log['val_loss'] < best:
            best = log['val_loss']
            torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'best_generator.pt')
            (out/'selection.json').write_text(json.dumps({'selected_epoch':epoch+1,'val_loss':best,
                'criterion':'lowest validation loss including initial checkpoint at epoch 0'},indent=2),encoding='utf-8')
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(log,flush=True)


if __name__ == '__main__':
    main()
