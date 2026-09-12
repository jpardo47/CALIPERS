"""Extract tensors from the upstream Lightning archive without executing its globals.

Only inert local metadata record types are allowlisted; their state is discarded.
The FFC state dictionary is strictly verified before saving.
"""
import argparse
import collections
import json
import typing
from pathlib import Path
import torch
from .cli import sha256


def extract(source,output):
    output=Path(output)
    if output.exists():
        raise ValueError('Output already exists')
    globals_=[dict,list,int,collections.defaultdict,typing.Any]
    names=['pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint',
           'omegaconf.base.ContainerMetadata','omegaconf.base.Metadata',
           'omegaconf.dictconfig.DictConfig','omegaconf.listconfig.ListConfig',
           'omegaconf.nodes.AnyNode']
    for name in names:
        module,cls=name.rsplit('.',1)
        globals_.append(type(cls,(),{'__module__':module}))
    previous=torch.serialization.get_safe_globals()
    torch.serialization.add_safe_globals(globals_)
    try:
        state=torch.load(source,map_location='cpu',weights_only=True)['state_dict']
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    tensors={k[len('generator.'):]:v for k,v in state.items() if k.startswith('generator.')}
    if not tensors or any(not isinstance(v,torch.Tensor) for v in tensors.values()):
        raise ValueError('Expected generator tensors')
    from inferencia_ecografia import build_generator
    model=build_generator(torch.device('cpu'))
    model.load_state_dict(tensors,strict=True)
    torch.save(tensors,output)
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'source_sha256':sha256(source),'output_sha256':sha256(output),
        'upstream':'https://github.com/advimman/lama',
        'download':'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip',
        'loader':'weights_only_true_inert_metadata_records','tensor_count':len(tensors)},indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    extract(a.source,a.output)
