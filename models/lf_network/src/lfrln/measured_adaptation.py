"""Measured bead feedback and biological TF supervision, without mandatory PS.

Profiled LF residual is a shape-consistency surrogate, NOT an EMCCD likelihood.
Positions exported by bead fits remain conditional labels, not independent GT.
"""
from pathlib import Path
import json
import hashlib
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import load_lf_tensor
from .boundary_context import weighted_shape_mse


class NoiseConditionMixture(Dataset):
    """Equal-weight replay conditions; no geometry-specific noise assignment.

    Underlying datasets share targets/indices. Validation alternates conditions
    deterministically; training draws a condition independently each access.
    """
    def __init__(self, datasets, labels, *, deterministic=False):
        if not datasets or len(datasets)!=len(labels) or len({len(d) for d in datasets})!=1:
            raise ValueError('Noise conditions need matching datasets and labels')
        self.datasets=list(datasets);self.labels=list(labels)
        self.dataset=datasets[0].dataset
        self.deterministic=deterministic

    def __len__(self):return len(self.datasets[0])

    def __getitem__(self,index):
        if len(self.datasets)==1:
            condition=0;sample=dict(self.datasets[0][index])
        elif self.deterministic:
            condition=index%len(self.datasets);sample=dict(self.datasets[condition][index])
        else:
            condition=int(torch.randint(len(self.datasets),()))
            seed=int(torch.randint(2**31,()))
            # Noise samplers consume different numbers of random draws. Isolate
            # their CPU RNG so changing the blank model cannot alter subsequent
            # condition choices or phantom augmentation draws in matched runs.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                sample=dict(self.datasets[condition][index])
        sample['lf_noise_condition']=self.labels[condition]
        return sample


class LFReplayOverlay(Dataset):
    """Replace only clean LF; retain original raw/TF identifiers and targets."""
    def __init__(self,dataset,root,*,kernel_sha256,response_sha256=None,excitation_sha256=None):
        self.dataset=dataset;self.root=Path(root);self.files=dataset.files
        self.directory=dataset.directory;self.augment=False
        if dataset.augment:raise ValueError('Calibrated replay requires render-consistent augmentation; cached paired rotations are disabled')
        self.contract=json.loads((self.root/'contract.json').read_text())
        if not self.contract.get('complete'):raise ValueError('Require a completed LF replay overlay')
        if self.contract['kernel_sha256']!=kernel_sha256 or self.contract.get('response_sha256')!=response_sha256:
            raise ValueError('Replay optics/response differ from feedback operator')
        if self.contract.get('excitation_sha256')!=excitation_sha256:
            raise ValueError('Replay excitation differs from the selected acquisition model')
        if hashlib.sha256((self.directory.parent/'manifest.json').read_bytes()).hexdigest()!=self.contract['source_manifest_sha256']:
            raise ValueError('Replay source phantoms differ')
        inventory=self.root/'samples.json'
        if hashlib.sha256(inventory.read_bytes()).hexdigest()!=self.contract['samples_sha256']:raise ValueError('Replay inventory changed')
        self.lf_checksums={r['name']:r['lf_sha256'] for r in json.loads(inventory.read_text()) if r['split']==self.directory.name}
        for path in self.files:
            if not (self.root/self.directory.name/(path.stem+'.npy')).exists():raise FileNotFoundError(path.stem)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        sample=dict(self.dataset[index])
        if sample['synthetic_transform']!=0:raise ValueError('Cached calibrated LF cannot be rotated independently of optics')
        path=self.root/self.directory.name/(self.files[index].stem+'.npy')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.lf_checksums[self.files[index].stem]:raise ValueError('Calibrated LF cache changed')
        lf=torch.from_numpy(np.load(path,allow_pickle=False).copy()).float()
        if lf.shape!=sample['lf'].shape or not torch.isfinite(lf).all() or lf.min()<0:raise ValueError('Invalid calibrated LF cache')
        sample.update(lf=lf,lf_feedback=lf,raw_feedback_compatible=True)
        return sample


def fixed_replay_view(mixture):
    """Evaluate training phantoms without augmentation using fixed camera draws.

    Copy wrappers so evaluation cannot change the stochastic optimization stream.
    The noise profiles and intensity distribution are inherited from the supplied
    deterministic validation mixture, with a caller-provided training base.
    """
    datasets=[]
    for original in mixture.datasets:
        wrapped=copy.copy(original)
        wrapped.dataset=copy.copy(original.dataset)
        wrapped.dataset.augment=False
        wrapped.deterministic=True
        datasets.append(wrapped)
    return NoiseConditionMixture(datasets,mixture.labels,deterministic=True)


def stratified_indices(groups,limit):
    """Deterministic group-balanced subset, spread within each group."""
    if limit<0:raise ValueError('Negative validation limit')
    if limit==0 or limit>=len(groups):return list(range(len(groups)))
    buckets={g:[i for i,v in enumerate(groups) if v==g] for g in sorted(set(groups))}
    counts={g:0 for g in buckets}
    while sum(counts.values())<limit:
        for g,ids in buckets.items():
            if counts[g]<len(ids) and sum(counts.values())<limit:counts[g]+=1
    return sorted(i for g,ids in buckets.items() for i in [ids[j] for j in np.linspace(0,len(ids)-1,counts[g]).astype(int)])


def profiled_lf_loss(predicted, observed, margin=11,mask=None):
    if predicted.shape!=observed.shape or predicted.ndim!=4:raise ValueError('LF shapes must agree')
    if margin<0 or 2*margin>=min(predicted.shape[-2:]):raise ValueError('Invalid LF margin')
    if mask is not None:
        if tuple(mask.shape)!=tuple(predicted.shape[-2:]):raise ValueError('LF mask must match the sensor ROI')
        mask=mask.to(predicted)
        if margin:mask=mask[margin:-margin,margin:-margin]
        if mask.sum()<=0:raise ValueError('No scored LF pixels')
    if margin:predicted=predicted[...,margin:-margin,margin:-margin];observed=observed[...,margin:-margin,margin:-margin]
    p=predicted.float().flatten(1);y=observed.float().flatten(1)
    if mask is not None:
        w=mask.float().flatten()[None];den=w.sum()
        pc=p-(p*w).sum(1,keepdim=True)/den;yc=y-(y*w).sum(1,keepdim=True)/den
        gain=((w*pc*yc).sum(1)/(w*pc.square()).sum(1).clamp_min(1e-12)).clamp_min(0).detach()
        return ((w*(gain[:,None]*pc-yc).square()).sum(1)/(w*yc.square()).sum(1).clamp_min(1e-12)).mean()
    pc=p-p.mean(1,keepdim=True);yc=y-y.mean(1,keepdim=True)
    # Detached nuisance optimum: envelope gradient for the profiled residual.
    gain=((pc*yc).sum(1)/pc.square().sum(1).clamp_min(1e-12)).clamp_min(0).detach()
    error=(gain[:,None]*pc-yc).square().mean(1)/yc.square().mean(1).clamp_min(1e-12)
    return error.mean()


class FittedBeadDataset(Dataset):
    def __init__(self,manifest,split,date=None):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        if not self.manifest.get('complete'):raise ValueError('Require a complete bead fitting manifest')
        contract=Path(self.manifest['contract'])
        if hashlib.sha256(contract.read_bytes()).hexdigest()!=self.manifest['contract_sha256']:raise ValueError('Bead fitting contract changed')
        self.samples=[r for r in self.manifest['samples'] if r['dataset_split']==split and (date is None or r['date']==date)]
        if not self.samples:raise ValueError('No bead samples in requested split')
        self._mask_cache={}
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        r=self.samples[i];raw=load_lf_tensor(r['lf_path']).float()
        # Fixed per-frame profiled offset, not subtraction of a noise realization.
        lf=(raw-r['background_adu']).clamp_min(0)/3000.
        report=json.loads(Path(r['report']).read_text())
        mask=None
        if r.get('valid_mask_path'):
            path=r['valid_mask_path']
            if path not in self._mask_cache:
                with open(path,'rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=r['valid_mask_sha256']:raise ValueError('Bead score mask archive changed')
                with np.load(path) as z:self._mask_cache[path]={int(k):torch.from_numpy(v.copy()).bool() for k,v in zip(z['frames'],z['valid_mask'])}
            mask=self._mask_cache[path][r['frame']]
        result=dict(lf=lf,raw_lf=raw,name=r['case']+'_'+str(r['frame']),case=r['case'],
            center_zyx_um=torch.tensor(r['center_zyx_um']),center_full_yx=torch.tensor(r['source_center_full_yx']),
            roi_start_yx=report['data']['roi_start_yx'],roi_shape=report['data']['roi_shape'],
            fit_relative_mse=r['relative_mse'],date=r['date'])
        if mask is not None:result['lf_valid_mask']=mask
        return result


def crop_bead_lf(prediction,sample):
    y,x=sample['roi_start_yx'];h,w=sample['roi_shape']
    return prediction[...,y:y+h,x:x+w]


def bead_centroid_loss(volume,sample,spacing=(.5,.21306818181818185,.21306818181818185)):
    """Optional local centroid constraint only; no fabricated absolute brightness."""
    cy,cx=[float(v) for v in sample['center_full_yx']];half=22
    y0=max(0,int(cy)-half);x0=max(0,int(cx)-half)
    v=volume[...,y0:int(cy)+half+1,x0:int(cx)+half+1].clamp_min(0)
    z=(torch.arange(v.shape[1],device=v.device)-(v.shape[1]-1)/2)*spacing[0]
    y=(torch.arange(v.shape[2],device=v.device)+y0-cy)*spacing[1]
    x=(torch.arange(v.shape[3],device=v.device)+x0-cx)*spacing[2]
    den=v.sum().clamp_min(1e-12)
    center=torch.stack(((v*z[None,:,None,None]).sum()/den,(v*y[None,None,:,None]).sum()/den,(v*x[None,None,None,:]).sum()/den))
    target=sample['center_zyx_um'].to(v)
    return ((center-target)/v.new_tensor([.5,.3,.3])).square().mean()


def biological_tf_loss(h1,target,valid_mask,axial_weights=None):
    # Target is an observed TF image; do not blur it again or use raw PS as GT.
    weight=valid_mask
    if axial_weights is not None:
        if axial_weights.shape!=(1,h1.shape[1],1,1):raise ValueError('Axial TF weights differ from depth')
        weight=valid_mask.to(h1)*axial_weights[0].to(h1)
    return weighted_shape_mse(h1,target,weight)


def direct_tf_interior_mask(shape, margins=(10,22,22), *, first_slice=0, excluded_global_planes=(), device='cpu'):
    """Explicit geometric loss region, independent of fluorescence intensity.

    Margins are experimental settings, not evidence of registration validity.
    Exclusions refer to acquisition-global plane indices, not local windows.
    """
    if len(shape)!=3 or len(margins)!=3 or any(m<0 or int(m)!=m or 2*m>=n for n,m in zip(shape,margins)):
        raise ValueError('Invalid direct TF mask margins')
    mask=torch.zeros(tuple(shape),dtype=torch.bool,device=device)
    mask[tuple(slice(int(m),int(n-m)) for n,m in zip(shape,margins))]=True
    for plane in excluded_global_planes:
        if int(plane)!=plane or plane<0:raise ValueError('Invalid global TF plane')
        local=int(plane)-int(first_slice)
        if 0<=local<shape[0]:mask[local]=False
    if not mask.any():raise ValueError('No valid TF voxels remain')
    return mask
