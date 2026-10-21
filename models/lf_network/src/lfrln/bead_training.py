"""Measured isolated-bead crops with conditional geometry, never voxel GT."""
from pathlib import Path
import hashlib,json
import torch
from torch.utils.data import Dataset
from .data import load_lf_tensor


class ConditionalBeadDataset(Dataset):
    def __init__(self,audit_folder,*,count_scale=3000.,maximum_fit_residual=.35):
        self.root=Path(audit_folder);self.count_scale=float(count_scale)
        if self.count_scale<=0:raise ValueError('Invalid LF count scale')
        labels=json.loads((self.root/'candidate_bead_labels.json').read_text())['labels']
        self.samples=[];self.excluded=[];self.reports={}
        for row in labels:
            key=row['case']
            report=self.reports.setdefault(key,json.loads((self.root/f'{key}_fixed_stage.json').read_text()))
            if report['bead_split']!='calibration':raise ValueError('Whole-bead holdout leaked into candidate labels')
            frame=next(f for f in report['frames'] if f['frame']==row['frame'])
            if frame['split']!='pose_fit':raise ValueError('Unseen-depth frame leaked into candidate labels')
            spread=row['near_optimal_multistart_spread_zyx_um']
            # Declared screening heuristics; not confidence intervals or a
            # declaration that the projector/nominal bead shape is correct.
            reasons=[]
            if frame['relative_mse']>maximum_fit_residual:reasons.append('Large conditional LF residual')
            if any(a>b for a,b in zip(spread,(.5,.3,.3))):reasons.append('Unstable multi-start position')
            if len(report['near_optimal_signs'])!=1:reasons.append('Ambiguous depth sign')
            if reasons:self.excluded.append(dict(case=key,frame=row['frame'],reasons=reasons))
            else:self.samples.append(row)
        if not self.samples:raise ValueError('No candidate bead geometry passed the declared screening')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        row=self.samples[index];report=self.reports[row['case']]
        frame=next(f for f in report['frames'] if f['frame']==row['frame'])
        file=next(f for f in report['data']['files'] if f['frame']==row['frame'])
        path=Path(file['path'])
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=file['sha256']:raise ValueError('Measured bead changed since pose fitting')
        raw=load_lf_tensor(path)
        y,x=report['data']['roi_start_yx'];h,w=report['data']['roi_shape']
        raw=raw[:,y:y+h,x:x+w]
        lf=(raw-frame['background_adu']).clamp_min(0)/self.count_scale
        return dict(lf=lf,center_zyx_um=torch.tensor(frame['center_zyx_um'],dtype=torch.float32),
            name=f"{row['case']}/{row['frame']:06d}",diameter_um=row['diameter_um'],
            label_domain='conditional finite-bead geometry; no voxel ground truth',
            preprocessing='subtract fitted scalar background once, clamp, divide by fixed ADU scale; no geometric transform',
            em_setting=500,exposure_seconds=1. if row['case'].startswith('20260722') else .05)
