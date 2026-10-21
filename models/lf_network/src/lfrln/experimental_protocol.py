"""Matched experimental objectives and explicit prerequisites for each branch."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import torch
from .boundary_context import context_ps_loss
from .point_scanning import ps_forward_relative_mse
from .calibrated_feedback import variance_weighted_feedback,weak_bead_geometry_loss


@dataclass(frozen=True)
class ExperimentalBranch:
    name:str
    context:bool=False
    measured_lf_weight:float=0.
    bead_geometry_weight:float=0.


BRANCHES=(ExperimentalBranch('baseline'),ExperimentalBranch('boundary_context',True),
          ExperimentalBranch('boundary_feedback',True,.02),
          ExperimentalBranch('boundary_feedback_beads',True,.02,.002))


class ExperimentalObjective:
    """Paired PS/LF objective with gradients only to the reconstruction.

    The training loop multiplies this paired objective by its existing paired
    schedule. Bead geometry is a separate calibration-bead minibatch objective.
    A context branch requires an actual enlarged prediction, never zero padding.
    """
    def __init__(self,branch,ps_operator,*,lf_observation=None):
        self.branch=branch;self.ps_operator=ps_operator;self.lf_observation=lf_observation
        if branch.context and not hasattr(ps_operator,'geometry'):
            raise ValueError('Context branch requires an enlarged-object PS operator')
        if branch.measured_lf_weight and lf_observation is None:
            raise ValueError('Measured feedback requires a calibrated observation operator')

    def paired(self,prediction,batch):
        reference=batch['gt'].to(prediction).float()
        ps=(context_ps_loss(prediction,reference,self.ps_operator) if self.branch.context else
            ps_forward_relative_mse(prediction,reference,self.ps_operator))
        feedback=ps.new_zeros(())
        if self.branch.measured_lf_weight:
            required=('lf_observation','lf_variance')
            if any(k not in batch for k in required):
                raise ValueError('Feedback needs observation and variance in a common unclipped count domain; lf model input is insufficient')
            expected=self.lf_observation(prediction.float())
            mask=batch.get('lf_observation_mask',None)
            if mask is not None:mask=mask.to(device=expected.device,dtype=torch.bool)
            feedback=variance_weighted_feedback(expected,batch['lf_observation'].to(expected),
                batch['lf_variance'].to(expected),mask=mask)
        return dict(total=ps+self.branch.measured_lf_weight*feedback,ps=ps,measured_lf=feedback)

    def beads(self,prediction,center_zyx_um,**kwargs):
        # Compress large moment errors so a diffuse early prediction cannot
        # dominate the supervised replay objective just because the crop is big.
        raw=weak_bead_geometry_loss(prediction,center_zyx_um,**kwargs)
        return self.branch.bead_geometry_weight*torch.log1p(raw)

    def describe(self):
        ps='RMS-normalized MSE(C P(f_extended), PS_observed)' if self.branch.context else 'historical masked RMS-normalized MSE(P(f_91), PS_observed)'
        return (f'Branch: {self.branch.name}\nL_paired = {ps} + {self.branch.measured_lf_weight:g} * L_measured_LF\n'
            'L_measured_LF = mean_valid[(g R A(W f) + b - observed_LF)^2 / detached_variance]\n'
            'Noise discrepancy is diagonal-variance approximate; exact resized noise has correlations.\n'
            f'L_bead_step = {self.branch.bead_geometry_weight:g} * log(1 + mean_axis[(centroid error / tolerance)^2 + (second-moment error / tolerance)^2])\n'
            'Bead labels are conditional geometry, not independently measured voxel GT.\n'
            'Synthetic replay is unchanged; paired schedule multiplies the complete L_paired.\n'
            'No GAN loss; no gradient from biological residuals to calibration parameters in these four branches.')


def experiment_plan(root):
    root=Path(root)
    checkpoint=root/'training_runs/experimental_ps_indexed517_em800_joint100_physlf0.02_v1/best_model.pt'
    context_path=root/'analysis/ps_boundary_context_20260913_v1/report.json'
    context=json.loads(context_path.read_text()) if context_path.exists() else None
    rows=[]
    for branch in BRANCHES:
        reasons=[]
        if not checkpoint.exists():reasons.append('Notebook21 source checkpoint is missing')
        if branch.context:
            reasons.extend(context['feasibility']['reasons'] if context else ['Run actual-PSF boundary control'])
        if branch.measured_lf_weight:
            reasons.extend(['Need native-camera LF forward model using verified export geometry',
                            'Need fixed excitation/photometric/noise contract for biological EM800/2s feedback'])
        if branch.bead_geometry_weight:reasons.append('Integrate the screened ConditionalBeadDataset into the enlarged-model minibatch schedule')
        rows.append({**asdict(branch),'source_checkpoint':str(checkpoint),
            'probe_epochs':10,'long_joint_epochs':100,'steps_per_epoch':300,
            'seed':20260913,'synthetic_physical_lf_weight':.02,
            'ready':not reasons,'prerequisites':reasons})
    return dict(branches=rows,scope='Matched continuation proposal from notebook21; no optimizer updates by this function',
        initialization='An enlarged architecture must use a documented migration from the same notebook21 weights for all comparable context branches',
        architecture_comparison='Compare migrated baseline too; a 91-versus-151 architecture comparison is not a loss-only ablation',
        split_rule='Synthetic replay and specimen splits unchanged; whole calibration beads held out from shared parameter fitting',
        selection='Inspect common core, whole-volume/boundary diagnostics, synthetic clean/noisy guards, unseen-bead residuals and PS structure')


def plot_boundary_history(history):
    import matplotlib.pyplot as plt
    keys=['outside_scored_squared_intensity_fraction','whole_volume_ps_shape_mse_diagnostic']
    if any('context_ps_shape_mse' in r['validation']['paired_details'].get('boundary_diagnostics',{}) for r in history):
        keys+=['context_ps_shape_mse','halo_ps_to_total_l2']
    titles=dict(zip(keys,['Outside-region squared intensity fraction','Whole-volume core-only PS MSE',
                         'PS MSE with estimated context','Halo / total PS prediction L2']))
    fig,axes=plt.subplots(1,len(keys),figsize=(5*len(keys),3),layout='constrained')
    for ax,key in zip(axes,keys):
        pairs=[(row['epoch'],row['validation']['paired_details'].get('boundary_diagnostics',{}).get(key)) for row in history]
        pairs=[(x,y) for x,y in pairs if y is not None]
        if pairs:ax.plot(*zip(*pairs))
        else:ax.text(.5,.5,'Not recorded in historical runs',ha='center',transform=ax.transAxes)
        ax.set_xlabel('Epoch');ax.set_title(titles[key])
    return fig
