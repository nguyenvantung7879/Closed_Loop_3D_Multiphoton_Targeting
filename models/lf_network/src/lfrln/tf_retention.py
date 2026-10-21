"""Paired corrected-TF supervision for H1 in otherwise unchanged joint training."""
from pathlib import Path
from dataclasses import asdict,replace
import hashlib,json,math
import h5py
import torch
from .training import H1PhysicsLossConfig,h1_physics_loss
from .datasets import SyntheticPairDataset

class H1TFRetention:
    def __init__(self, root, raw_root, source_h1, weight, *, axial_fwhm_um=None):
        self.root=Path(root);self.weight=float(weight)
        if not math.isfinite(self.weight) or self.weight<=0:raise ValueError('TF retention weight must be finite and positive')
        contract=json.loads((self.root/'generation_contract.json').read_text())
        if hashlib.sha256((Path(raw_root)/'manifest.json').read_bytes()).hexdigest()!=contract['source_manifest_sha256']:
            raise ValueError('Corrected TF cache was generated from a different raw dataset')
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest.get('target_operator')!='temporal_focusing' or not manifest.get('completion'):raise ValueError('Require completed corrected TF cache')
        payload=torch.load(source_h1,map_location='cpu',weights_only=False)
        self.config=H1PhysicsLossConfig(**payload['loss_config'])
        if axial_fwhm_um is not None:self.config=replace(self.config,use_axial_l2_weighting=True,axial_fwhm_um=axial_fwhm_um)
        c=self.config
        if c.l2_weight!=1 or any(getattr(c,k)!=0 for k in ('charbonnier_weight','sum_projection_weight','generalized_projection_weight','gradient_weight','feedback_weight','foreground_weight')):
            raise ValueError('Retention requires the original L2-only H1 loss configuration')
        self.paths={}
        for split in ('training','validation','testing'):
            files=list((Path(raw_root)/split).glob('*.h5'))
            for path in files:
                target=self.root/split/path.name
                if not target.is_file():raise FileNotFoundError(target)
                self.paths[(split,path.stem)]=target
        self.record=dict(weight=self.weight,target_root=str(self.root),source_h1=str(source_h1),
            source_h1_sha256=hashlib.sha256(Path(source_h1).read_bytes()).hexdigest(),loss_config=asdict(c),target_contract=contract,
            interpretation='H1 versus cached corrected T(raw); same spatial transform as raw GT/LF; target unchanged for noisy LF; final raw objective unchanged')
    def target(self,names,splits,device,transforms=None):
        values=[]
        transforms=[0]*len(names) if transforms is None else transforms
        if not len(names)==len(splits)==len(transforms):raise ValueError('TF batch identifiers differ in length')
        for name,split,transform in zip(names,splits,transforms):
            key=(split,name)
            if key not in self.paths:raise KeyError('Missing corrected TF sample: '+str(key))
            with h5py.File(self.paths[key],'r') as f:value=torch.from_numpy(f['gt'][()].copy()).float()
            values.append(SyntheticPairDataset._apply_transform(value,int(transform)))
        return torch.stack(values).to(device)
    def apply(self,losses,stages,batch):
        target=self.target(batch['name'],batch['synthetic_split'],stages['h1'].device,batch.get('synthetic_transform'))
        term=h1_physics_loss(stages['h1'],target,batch['lf'].to(target.device),forward_model=None,config=self.config)['total']
        losses=dict(losses)
        losses['h1_tf_retention']=term
        losses['h1_tf_retention_weighted']=self.weight*term
        losses['total']=losses['total']+self.weight*term
        return losses

def build_h1_tf_retention(args):
    weight=getattr(args,'h1_tf_retention_weight',0.)
    if not math.isfinite(weight) or weight<0:raise ValueError('Invalid TF retention weight')
    continuing=getattr(args,'continue_checkpoint',None)
    if continuing:
        saved=Path(continuing).parent/'h1_tf_retention.json'
        old=json.loads(saved.read_text()) if saved.exists() else None
        if (old is None and weight>0) or (old is not None and float(old['weight'])!=weight):
            raise ValueError('Resume must preserve its TF retention weight; use a new source-checkpoint branch to change the objective')
    if weight==0:return None
    if args.stage!='joint':raise ValueError('H1 TF retention currently requires joint stage')
    if args.h1_tf_target_root is None or args.h1_tf_loss_checkpoint is None:raise ValueError('TF retention requires its target cache and original H1 loss checkpoint')
    return H1TFRetention(args.h1_tf_target_root,args.dataset,args.h1_tf_loss_checkpoint,weight)
