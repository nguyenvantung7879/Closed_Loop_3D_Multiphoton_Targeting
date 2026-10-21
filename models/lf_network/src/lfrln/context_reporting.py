"""Boundary and TF diagnostics kept separate from checkpoint selection."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch

from .boundary_context import weighted_shape_mse


def matched_context_summary(root, phase='probe'):
    """Only completed records from the new matched branches, never older runs."""
    histories, metadata, rows, pending = {}, {}, [], []
    for branch in ('baseline','boundary_aux_context'):
        folder=Path(root)/'training_runs'/f'experimental_auxcontext_{branch}_{phase}_20260915_v1'
        if not (folder/'history.json').exists():
            pending.append(str(folder));continue
        history=json.loads((folder/'history.json').read_text())
        if not history:
            pending.append(str(folder));continue
        meta=json.loads((folder/'run_metadata.json').read_text())
        base=json.loads((folder/'source_baseline.json').read_text())['metrics']
        histories[branch]=history;metadata[branch]=meta
        eligible=[r for r in history if r['eligible']]
        selected=min(eligible,key=lambda r:r['selection_score']) if eligible else dict(epoch=0,validation=base)
        for state,row in [('selected',selected),('last',history[-1])]:
            val=row['validation'];diag=val['paired_details']['boundary_diagnostics']
            rows.append(dict(branch=branch,state=state,epoch=row['epoch'],common_core_ps_mse=val['paired'],
                             synthetic_clean=val['synthetic_clean'],synthetic_noisy=val['synthetic_noisy'],**diag))
    common_epochs=0
    if len(histories)==2:
        a,b=metadata.values()
        for key in ('source_sha256','random_seed','cache_fingerprint','synthetic_root','synthetic_loss',
                    'physical_synthetic_lf_weight','steps_per_epoch','paired_every_n_steps','paired_weight_max'):
            if a[key]!=b[key]:raise ValueError(f'Matched-branch invariant differs: {key}')
        x,y=histories.values()
        for first,second in zip(x,y):
            if first['input_digest_sha256']!=second['input_digest_sha256']:
                raise ValueError(f'Matched branches saw different inputs at epoch {first["epoch"]}')
            common_epochs+=1
    return dict(rows=rows,pending=pending,input_matched_epochs=common_epochs,
                note='Context fit/halo spill are auxiliary diagnostics. Common core scores and images assess the requested reconstruction.')


def plot_volume_columns(columns, *, title, spacing=(.5, .213068181818, .213068181818)):
    import matplotlib.pyplot as plt
    from .visualization import safe_quantile
    fig, axes = plt.subplots(3, len(columns), figsize=(3.4*len(columns), 7.5), layout='constrained')
    for col, (name, tensor) in enumerate(columns.items()):
        value = torch.as_tensor(tensor).detach().float().cpu()
        if value.ndim == 4:
            value = value[0]
        scale = safe_quantile(value.clamp_min(0), .999).clamp_min(1e-8)
        value = value/scale
        z,y,x = value.shape
        for row, (projection, extent) in enumerate((
            (value.amax(0), (0,x*spacing[2],y*spacing[1],0)),
            (value.amax(1), (0,x*spacing[2],z*spacing[0],0)),
            (value.amax(2), (0,y*spacing[1],z*spacing[0],0)))):
            axes[row,col].imshow(projection, cmap='magma', vmin=0, vmax=1, extent=extent)
            axes[row,col].set_title(name if row == 0 else ('XZ' if row == 1 else 'YZ'))
            axes[row,col].set_xlabel('µm')
    fig.suptitle(title+'\nIndependent q99.9 display scales; XY/XZ/YZ MIPs', fontsize=11)
    return fig


@torch.no_grad()
def stage_figure(stages, title):
    return plot_volume_columns({'H1':stages['h1'], 'H2 multiplicative':stages['h2_multiplicative'],
        'Positive additive detail':stages['h2_additive_detail'].clamp_min(0), 'Final raw volume':stages['merged']}, title=title)


@torch.no_grad()
def tf_comparison(model, dataset, operator, *, device, context_decoder=None, context_operator=None,
                  max_batches=None, make_figures=True):
    """Equal frames -> acquisitions -> specimens. TF never selects a checkpoint.

    Report the core-only score for every model on exactly the same mask, plus
    an explicitly distinct context score. The latter can improve by changing
    the auxiliary halo without improving the requested core reconstruction.
    """
    modes = [(m,m.training) for m in model.modules()]
    model.eval()
    rows, figures, seen = [], {}, set()
    try:
        for index in range(len(dataset)):
            if max_batches is not None and index >= max_batches:
                break
            sample = dataset[index]
            lf = sample['lf'][None].to(device)
            raw = model.predict_stages(lf)['merged'].float()
            reference = sample['tf_reference'][None].to(device)
            core_tf = operator(raw)
            row = dict(name=sample['name'], acquisition=sample['acquisition'], specimen=sample['specimen'],
                       core_tf_shape_mse=float(weighted_shape_mse(core_tf, reference, operator.valid_mask)),
                       whole_volume_tf_mse_zero_exterior_diagnostic=float(weighted_shape_mse(core_tf,reference,torch.ones_like(operator.valid_mask))))
            columns = {'Measured TF':reference, 'T(final core), zero exterior':core_tf}
            if context_decoder is not None:
                if context_operator is None:
                    raise ValueError('Context TF comparison needs its actual extended-volume operator')
                extended, halo = context_decoder(lf, raw)
                full_tf = context_operator(extended)
                row['context_tf_shape_mse'] = float(weighted_shape_mse(full_tf, reference, context_operator.valid_mask))
                row['halo_tf_to_total_l2'] = float(torch.linalg.vector_norm(full_tf-core_tf) /
                                                 torch.linalg.vector_norm(full_tf).clamp_min(1e-12))
                columns['T(core + estimated halo)'] = full_tf
                del extended, halo
            columns['Final raw volume'] = raw
            if make_figures and sample['specimen'] not in seen:
                figures[sample['specimen']] = plot_volume_columns(columns,
                    title=sample['name']+' | TF observation comparison; not H1 supervision')
                seen.add(sample['specimen'])
            rows.append(row)
        if not rows:
            raise ValueError('Empty TF comparison')
        metrics = {}
        for key in rows[0].keys()-{'name','acquisition','specimen'}:
            acquisitions = defaultdict(list)
            specimen_for = {}
            for row in rows:
                acquisitions[row['acquisition']].append(row[key])
                specimen_for[row['acquisition']] = row['specimen']
            specimens = defaultdict(list)
            for name, values in acquisitions.items():
                specimens[specimen_for[name]].append(float(np.mean(values)))
            by_specimen = {k:float(np.mean(v)) for k,v in specimens.items()}
            metrics[key] = dict(specimen_mean=float(np.mean(list(by_specimen.values()))), by_specimen=by_specimen)
        return dict(frames=len(rows), metrics=metrics, rows=rows,
                    core_scored_fraction=float(operator.valid_mask.float().mean()),
                    context_scored_fraction=float(context_operator.valid_mask.float().mean()) if context_operator else None,
                    boundary_note='At 99% coverage the TF core mask can be very small. Whole-volume score assumes empty exterior and is a biased diagnostic; neither proves reconstruction quality.',
                    use='Separate diagnostic; no TF loss, no H1 target change, no checkpoint selection',
                    normalization='Same weighted RMS for prediction/reference per frame; raw TF cache retained',
                    context_warning='Halo contribution is estimated and may absorb model error; compare common core score too'), figures
    finally:
        for module, mode in modes:
            module.training = mode
