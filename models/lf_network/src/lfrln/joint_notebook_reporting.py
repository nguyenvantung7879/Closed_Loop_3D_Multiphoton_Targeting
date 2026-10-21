"""Explicit loss descriptions and replaceable notebook21 displays.

Reporting only: these helpers do not change the objective or optimizer.
"""
from dataclasses import asdict
import html
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML


def describe_joint_loss(config, *, physical_weight, paired_weight_max, paired_every,
                        ramp_epochs, steps_per_epoch, kernel=(3,5,5), paired_epsilon=1e-8):
    if min(paired_every, ramp_epochs, steps_per_epoch) <= 0:
        raise ValueError('Step counts and ramp must be positive')
    values=asdict(config)
    paired_steps=len(range(0,steps_per_epoch,paired_every))
    active=[name for name,value in values.items() if name.endswith('_weight') and value!=0]
    lines=[
        'EFFECTIVE JOINT TRAINING OBJECTIVE (actual code; no change to the trained loss)',
        'O(y) = fusion(H1(y), H2(y, H1(y))) is the final RAW-domain reconstruction.',
        f'L_step = L_supervised + {physical_weight:g} * L_synthetic_physical_LF + I_paired(step) * w_PS(epoch) * L_PS',
        f'w_PS(epoch) = {paired_weight_max:g} * min(1, epoch/{ramp_epochs})',
        f'I_paired(step) = 1 when zero-based step % {paired_every} == 0; {paired_steps}/{steps_per_epoch} steps per epoch.',
        f'At full ramp the average paired coefficient per optimizer step is {paired_weight_max*paired_steps/steps_per_epoch:g}.',
        '', '1. Synthetic raw-volume supervision',
        f'  weight = {config.final_weight:g}; reduction = {config.l2_reduction}; scale s = {config.supervised_scale:.12g}',
    ]
    numerator='mean_voxels(((O - GT_raw)/s)^2)'
    if config.l2_reduction=='relative_per_sample':
        lines += [f'  L_volume = mean_batch[{numerator} / max(mean_voxels((GT_raw/s)^2), {config.relative_l2_epsilon:g})]',
                  '  Energy cap is INACTIVE; scale cancels except when the epsilon floor is active.']
    elif config.l2_reduction=='energy_capped_per_sample':
        lines += [f'  L_volume = mean_batch[{numerator} / max(mean_voxels((GT_raw/s)^2), {config.relative_l2_energy_floor:g})]']
    else:
        lines += [f'  L_volume = mean_batch[{numerator}]']
    lines += [
        '  Training uses relative MSE (squared error), not the relative L2 norm shown in validation.',
        '', '2. Synthetic physical LF consistency',
        f'  weight = {physical_weight:g}; '+('ACTIVE' if physical_weight>0 else 'DISABLED'),
        '  L_synthetic_physical_LF = mean_batch[mean((A(O)-y_clean)^2) / max(mean(y_clean^2), 1e-12)]',
        '  A = fixed detection-only raw-emitter projector, including the recorded optical gain.',
        '  y_clean = lf_feedback; the network input may be noisy. No fitted output gain or RMS normalization.',
        '  This is separate from the learned H2 forward-head fp_consistency_weight.',
        '', '3. Measured LF / registered PS reference term',
        f'  B = avg_pool3d(kernel={tuple(kernel)}, stride=1, zero padding, padding included in average).',
        f'  R(v) = B(v) / max(sqrt(mean(B(v)^2)), {paired_epsilon:g}), separately for each volume.',
        '  L_PS = mean_batch[mean_voxels((R(O(y_measured))-R(PS_reference))^2)]',
        '  LF preprocessing: subtract the registered blank once, clip at zero, divide by 3000.',
        '  PS preprocessing: acquisition-level background and scale; exact indexed 91-slice window.',
        '  This is a coarse, scale-invariant PS-reference heuristic; it is not a calibrated PS forward loss.',
        '  Direct measured-LF physical consistency weight = 0.',
        '  Direct unsmoothed/high-frequency experimental structural supervision = 0.',
        '', '4. Synthetic auxiliary weights (these are not experimental detail losses)',
    ]
    for name in ('h2_auxiliary_weight','high_frequency_weight','fusion_residual_weight',
                 'view_attention_identity_weight','view_attention_smoothness_weight',
                 'fp_consistency_weight','fp_flux_weight'):
        value=values[name]
        lines.append(f'  {name:<38} = {value:g}  '+('ACTIVE' if value else 'disabled'))
    lines += ['', 'Raw effective H2FusionLossConfig:']
    for name,value in values.items():
        lines.append(f'  {name:<38} {value}'+('  <-- active weight' if name in active else ''))
    lines += ['', 'Checkpoint selection: specimen-macro PS-reference score, guarded by clean AND noisy synthetic relative L2.',
              'Loss-value coefficients do not measure the relative gradient influence of each term.']
    return '\n'.join(lines)


class LatestDisplays:
    """One IPython display_id per slot; update the existing output and close figures."""
    def __init__(self, display_function=None):
        self._display=display_function or display
        self.handles={}

    def show(self, key, value):
        handle=self.handles.get(key)
        if handle is None:
            self.handles[key]=self._display(value,display_id=True)
        else:
            handle.update(value)

    def text(self, key, text):
        self.show(key,HTML('<pre style="white-space:pre-wrap">'+html.escape(str(text))+'</pre>'))

    def figure(self, key, figure):
        try:self.show(key,figure)
        finally:plt.close(figure)


def plot_joint_history(rows, source_metrics, *, physical_weight, paired_every, steps_per_epoch,
                       relative_guard=.05, paired_label='PS-reference MSE',
                       paired_title='Experimental coarse PS-reference score', empty_halo_weight=0.):
    if not rows:raise ValueError('No recorded history')
    epochs=np.array([r['epoch'] for r in rows])
    volume=np.array([r['train_synthetic_volume'] for r in rows])
    physical=np.array([r['train_synthetic_physical_lf'] for r in rows])*physical_weight
    paired_fraction=len(range(0,steps_per_epoch,paired_every))/steps_per_epoch
    paired=np.array([r['train_paired']*r['paired_weight']*paired_fraction for r in rows])
    fig,axes=plt.subplots(1,4,figsize=(19,4),layout='constrained')
    halo=np.array([r.get('train_synthetic_empty_halo',0.) for r in rows])*empty_halo_weight
    contributions=[('volume',volume),('physical LF × weight',physical),('PS × weight × update fraction',paired)]
    if empty_halo_weight:contributions.append(('synthetic empty halo × weight',halo))
    for label,values in contributions+[('approx. total per step',volume+physical+paired+halo)]:
        axes[0].plot(epochs,values,label=label)
    axes[0].set_yscale('log');axes[0].set_title('Weighted training contributions')
    axes[0].set_ylabel('Loss value (log scale)')
    for key in ('synthetic_clean','synthetic_noisy'):
        axes[1].plot(epochs,[r['validation'][key] for r in rows],label=key)
        axes[1].axhline(source_metrics[key]*(1+relative_guard),ls=':',alpha=.5,label=key+' guard')
    axes[1].set_title('Synthetic validation: relative L2 norm')
    axes[2].plot(epochs,[r['validation']['paired'] for r in rows],label=paired_label)
    axes[2].axhline(source_metrics['paired'],ls=':',label='source')
    eligible=[r for r in rows if r['eligible']]
    if eligible:
        best=min(eligible,key=lambda r:r['selection_score'])
        axes[2].scatter(best['epoch'],best['validation']['paired'],s=40,label=f'selected epoch {best["epoch"]}')
    axes[2].set_title(paired_title)
    for label,key in [('physical LF (unweighted)','train_synthetic_physical_lf'),('PS (unweighted)','train_paired')]:
        axes[3].plot(epochs,[r[key] for r in rows],label=label)
    axes[3].set_title('Unweighted diagnostic components')
    for ax in axes:ax.set_xlabel('joint epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Training means include AMP-skipped batches; loss coefficients do not measure gradient influence.',fontsize=10)
    return fig


def describe_psf_joint_loss(config,operator,*,context_operator=None,synthetic_empty_halo_weight=0.,**options):
    text=describe_joint_loss(config,**options)
    text=text.replace('EFFECTIVE JOINT TRAINING OBJECTIVE (actual code; no change to the trained loss)',
        'EFFECTIVE PSF-SUPERVISED JOINT OBJECTIVE (new experiment; notebook21 retained)')
    start=text.index('3. Measured LF / registered PS reference term')
    stop=text.index('4. Synthetic auxiliary weights',start)
    replacement='\n'.join([
        '3. Measured LF / PS forward-response term',
        '  f_hat = final output interpreted as latent fluorophore structure on the registered LF grid.',
        '  P(f_hat) = full supplied 920 nm two-photon PSF convolved with f_hat, with zero extension.',
        '  The supplied response is already squared and normalized; no extra squaring.',
        '  The observed PS reference is NOT convolved or average-filtered.',
        f'  M = P(ones) >= {operator.minimum_coverage:g}; valid voxel fraction = {float(operator.valid_mask.float().mean()):.4f}.',
        '  R_M(v) = v / max(sqrt(mean_M(v^2)), 1e-8).',
        '  L_PS = mean_batch[mean_M((R_M(P(f_hat))-R_M(PS_reference))^2)].',
        '  Registered PS grid equals LF grid by user confirmation on 2026-09-10; .208 metadata is superseded.',
        '  Cross-modality gain remains removed by RMS normalization; PSF is fixed, gradients reach the network.',
        '  Synthetic excitation is uniform (W=1), consistent with raw synthetic supervision and A.',
        '  Measured-LF physical consistency weight remains 0 pending separate illumination/optics calibration.',
        '  No biological-shaped synthetic training controls are added.',
        '',
    ])
    if context_operator is not None:
        replacement=replacement.replace('  P(f_hat) = full supplied 920 nm two-photon PSF convolved with f_hat, with zero extension.',
            '  f_ext = embed(f_hat) + H(y); H(y) is an explicitly predicted nonnegative emitter halo.\n'
            '  p = C P(f_ext): linear convolution on the enlarged emitter domain, then crop the observed core.\n'
            '  Zero extension is applied only beyond this enlarged domain; real exterior is estimated, not known.')
        replacement=replacement.replace(
            f'  M = P(ones) >= {operator.minimum_coverage:g}; valid voxel fraction = {float(operator.valid_mask.float().mean()):.4f}.',
            f'  M = C P(ones_extended) >= {context_operator.full.minimum_coverage:g}; valid core fraction = {float(context_operator.valid_mask.float().mean()):.4f}.')
        replacement=replacement.replace('R_M(P(f_hat))','R_M(p)')
    result=text[:start]+replacement+'\n'+text[stop:]
    if synthetic_empty_halo_weight:
        result=result.replace('L_step = L_supervised +',f'L_step = L_supervised + {synthetic_empty_halo_weight:g} * L_empty_synthetic_halo +')
        result+='\nL_empty_synthetic_halo = mean_batch[mean_outside(H(y)^2) / max(mean(GT_raw^2),1e-12)].\n'
        result+='Only the finite simulated object exterior is known empty; no penalty on real boundary intensity.\n'
    result+='\nTF comparison weight = 0; no TF training or H1 target change.\n'
    return result
