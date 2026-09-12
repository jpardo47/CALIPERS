"""Executable CPU pilot: frozen patient splits, three CAD arms, segmentation, XAI.

Reference images are legacy-curated, NOT expert-certified clean. Annotations in
this controlled challenge are synthetic. No clinical validation is inferred.
"""
import argparse,copy,csv,json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from .cli import read,write,sha256
from .manifest import validate
from .benchmark import overlay,bbox
from .cad import paired_audit,attention_energy
from .metrics import segmentation_metrics


def letterbox(image,size=128,mask=False):
    h,w=image.shape[:2]
    scale=size/max(h,w)
    hh,ww=max(1,round(h*scale)),max(1,round(w*scale))
    small=cv2.resize(image,(ww,hh),interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)
    shape=(size,size)+image.shape[2:]
    out=np.zeros(shape,image.dtype)
    y,x=(size-hh)//2,(size-ww)//2
    out[y:y+hh,x:x+ww]=small
    return out


def marks(image,seed,label):
    """Random white/yellow overlays + explicit label-confounded extra '+' count.

    The confound is an artificial positive control, not a claim about BUS_BRA.
    """
    corrupted,m=overlay(image,seed)
    alpha=np.max((corrupted.astype(float)-image)/(255-image.astype(float)+1e-6),axis=2)
    rng=np.random.default_rng(seed)
    color=np.array([255,255,0] if rng.random()<0.5 else [255,255,255])
    out=np.rint(image*(1-alpha[...,None])+color*alpha[...,None]).clip(0,255).astype(np.uint8)
    stamp=np.zeros(image.shape[:2],np.uint8)
    for i in range(1+3*int(label)):
        x,y=10+i*15,12
        cv2.drawMarker(stamp,(x,y),255,cv2.MARKER_CROSS,7+2*(seed%3),1+(seed%2),cv2.LINE_AA)
    a=stamp.astype(float)/255
    out=np.rint(out*(1-a[...,None])+color*a[...,None]).clip(0,255).astype(np.uint8)
    return out,m|(stamp>0)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(24,32,3,padding=1),nn.ReLU())
        self.head=nn.Linear(32,1)
    def forward(self,x):
        self.activation=self.features(x)
        return self.head(self.activation.mean((-2,-1))).squeeze(1)


class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=nn.Sequential(nn.Conv2d(3,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,12,3,padding=1),nn.ReLU())
        self.mid=nn.Sequential(nn.Conv2d(12,24,3,padding=1),nn.ReLU(),nn.Conv2d(24,24,3,padding=1),nn.ReLU())
        self.out=nn.Sequential(nn.Conv2d(36,12,3,padding=1),nn.ReLU(),nn.Conv2d(12,1,1))
    def forward(self,x):
        a=self.enc(x)
        b=self.mid(F.avg_pool2d(a,2))
        return self.out(torch.cat([a,F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False)],1))


def tensor(im):
    return torch.from_numpy(im.astype(np.float32)/255).permute(2,0,1)


def prepare(manifest,outdir,weights):
    out=Path(outdir)
    if out.exists(): raise ValueError('New cache required')
    rows=validate(json.loads(Path(manifest).read_text(encoding='utf-8')))
    from inferencia_ecografia import InpainterEcografia
    torch.set_num_threads(4)
    model=InpainterEcografia(weights,'cpu')
    out.mkdir(parents=True)
    for i,row in enumerate(rows):
        im=read(row['image'])
        if im.ndim==2: im=np.repeat(im[...,None],3,axis=2)
        ref=letterbox(im)
        lesion=letterbox(read(row['lesion_mask'],True),mask=True)>0
        label=int(row['label']=='malignant')
        marked,m=marks(ref,60100+i,label)
        swapped,sm=marks(ref,60100+i,1-label)
        painted=model.inpaint(marked,m)
        black=marked.copy();black[bbox(m)]=0
        case=out/f'{i:04d}';case.mkdir()
        for name,a in [('reference',ref),('marked',marked),('swapped',swapped),('bbox',black),('inpaint',painted),
                       ('artifact',m.astype(np.uint8)*255),('lesion',lesion.astype(np.uint8)*255)]:
            write(case/(name+'.png'),a)
        row.update(cache=str(case.resolve()),synthetic_seed=60100+i,label_binary=label)
        print(f'Cache {i+1}/{len(rows)}',flush=True)
    (out/'manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'weights_sha256':sha256(weights),
        'manifest_sha256':sha256(manifest),'scope':'synthetic_positive_control_unverified_clean_reference',
        'cad_grid':'128x128 aspect-preserving letterbox; not native PSF measurements',
        'label_confound':'1 vs 4 added crosses; synthetic positive control'},indent=2),encoding='utf-8')


def train(cache,outdir,epochs=12,seeds=(11,22,33)):
    out=Path(outdir)
    if out.exists(): raise ValueError('New experiment directory required')
    rows=json.loads((Path(cache)/'manifest.json').read_text(encoding='utf-8'))
    # Revalidate original pixels and membership before any optimization.
    validate(rows)
    out.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    idx={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','val','test')}
    for s,ix in idx.items():
        if len(ix)==0 or len({rows[i]['label_binary'] for i in ix})<2: raise ValueError(f'{s}: both labels required')
    images={v:torch.stack([tensor(read(Path(r['cache'])/(v+'.png'))) for r in rows])
            for v in ('reference','marked','swapped','bbox','inpaint')}
    lesions=torch.from_numpy(np.stack([read(Path(r['cache'])/'lesion.png',True)>0 for r in rows]).astype(np.float32))[:,None]
    y=torch.tensor([r['label_binary'] for r in rows],dtype=torch.float32)
    histories=[];summary=[];predictions=[];xai=[]
    for seed in seeds:
        for arm,view in [('raw','marked'),('bbox','bbox'),('inpaint','inpaint')]:
            torch.manual_seed(seed)
            model=Classifier();optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
            best=float('inf');best_state=None
            for epoch in range(epochs):
                model.train();losses=[]
                order=np.random.default_rng(seed+epoch).permutation(idx['train'])
                for start in range(0,len(order),8):
                    ix=order[start:start+8]
                    loss=F.binary_cross_entropy_with_logits(model(images[view][ix]),y[ix])
                    optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
                model.eval()
                with torch.no_grad(): val=F.binary_cross_entropy_with_logits(model(images[view][idx['val']]),y[idx['val']]).item()
                histories.append(dict(seed=seed,arm=arm,epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=val))
                if val<best: best=val;best_state=copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state);model.eval()
            torch.save(best_state,out/f'cad_{arm}_{seed}.pt')
            with torch.no_grad(): scores={v:model(x[idx['test']]).sigmoid().numpy() for v,x in images.items()}
            test=idx['test'];labels=y[test].numpy();patients=[rows[i]['patient_id'] for i in test]
            audit=paired_audit(labels,scores['reference'],scores['marked'],patients,seed=seed,repeats=1000)
            audit.update(seed=seed,arm=arm,auc_inpaint=float(roc_auc_score(labels,scores['inpaint'])),
                         auc_swapped=float(roc_auc_score(labels,scores['swapped'])),
                         mean_swap_score_change=float(np.mean(np.abs(scores['marked']-scores['swapped']))))
            summary.append(audit)
            for j,i in enumerate(test):
                predictions.append(dict(seed=seed,arm=arm,patient_id=rows[i]['patient_id'],label=int(labels[j]),
                    **{v:float(s[j]) for v,s in scores.items()}))
                # Actual Grad-CAM from this trained model, evaluated on curated input.
                model(images['inpaint'][i:i+1]).sum().backward()
                # Recompute to retain derivative wrt activation, independent of parameter .grad.
                logit=model(images['inpaint'][i:i+1])
                grad=torch.autograd.grad(logit.sum(),model.activation)[0]
                cam=F.relu((grad.mean((-2,-1),keepdim=True)*model.activation).sum(1,keepdim=True))
                cam=F.interpolate(cam,size=(128,128),mode='bilinear',align_corners=False)[0,0].detach().numpy()
                m=read(Path(rows[i]['cache'])/'artifact.png',True)>0
                energy=attention_energy(cam,m,lesions[i,0].numpy()>0)
                xai.append(dict(seed=seed,arm=arm,patient_id=rows[i]['patient_id'],**energy))
                np.save(out/f'cam_{arm}_{seed}_{i}.npy',cam)
            print(f'CAD {seed}/{arm}: AUC reference={audit["auc_clean"]:.3f}',flush=True)
    # Independent small segmenter trained on original lesion annotations, frozen for comparison.
    torch.manual_seed(2026)
    seg=Segmenter();optimizer=torch.optim.Adam(seg.parameters(),lr=0.001)
    best=float('inf');seg_state=None
    for epoch in range(epochs):
        seg.train()
        for start in range(0,len(idx['train']),4):
            ix=idx['train'][start:start+4]
            pred=seg(images['reference'][ix]);prob=pred.sigmoid();target=lesions[ix]
            dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
            loss=F.binary_cross_entropy_with_logits(pred,target)+dice
            optimizer.zero_grad();loss.backward();optimizer.step()
        seg.eval()
        with torch.no_grad():
            vpred=seg(images['reference'][idx['val']]);vprob=vpred.sigmoid();vtarget=lesions[idx['val']]
            val=(F.binary_cross_entropy_with_logits(vpred,vtarget)+1-(2*(vprob*vtarget).sum()+1)/(vprob.sum()+vtarget.sum()+1)).item()
        if val<best:best=val;seg_state=copy.deepcopy(seg.state_dict())
    seg.load_state_dict(seg_state);seg.eval();torch.save(seg_state,out/'segmenter.pt')
    segmentations=[]
    with torch.no_grad():
        before=seg(images['reference'][idx['test']]).sigmoid().numpy()[:,0]>0.5
        after=seg(images['inpaint'][idx['test']]).sigmoid().numpy()[:,0]>0.5
    for j,i in enumerate(idx['test']):
        gt=lesions[i,0].numpy()>0
        segmentations.append(dict(patient_id=rows[i]['patient_id'],before=segmentation_metrics(gt,before[j]),
            after=segmentation_metrics(gt,after[j]),stability=segmentation_metrics(before[j],after[j])))
    for name,value in [('history',histories),('cad_results',summary),('predictions',predictions),('xai',xai),('segmentation',segmentations)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    (out/'config.json').write_text(json.dumps({'cache_manifest_sha256':sha256(Path(cache)/'manifest.json'),
        'epochs':epochs,'seeds':list(seeds),'scope':'exploratory_synthetic_confound_not_real_annotation_validation',
        'classifier':'small_CNN_from_scratch','segmentation_units':'128px letterbox, not mm',
        'cleanliness_verified':False,'torch':torch.__version__},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','train'])
    p.add_argument('--manifest');p.add_argument('--weights');p.add_argument('--cache')
    p.add_argument('--outdir',required=True);p.add_argument('--epochs',type=int,default=12)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.manifest,a.outdir,a.weights)
    else:train(a.cache,a.outdir,a.epochs)
