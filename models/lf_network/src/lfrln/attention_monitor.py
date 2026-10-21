"""Fixed held-out angular-attention probes; does not update model or training RNG."""
from pathlib import Path
import json
import numpy as np
import torch


def _files(dataset):
    while not hasattr(dataset, "files") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset.files


@torch.no_grad()
def record_attention(model, dataset, *, epoch, output_dir, device,
                     families=("1um_spheres", "2um_spheres", "10um_shells"),
                     gains=(0.25, 1., 4.), suppression_share=0.001):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir) / "attention_monitor"
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _files(dataset)
    indices = [next((i for i, p in enumerate(files) if p.stem.endswith('_' + family)), None) for family in families]
    modules = [(m, m.training) for m in model.modules()]
    records = []
    connected = hasattr(model, "h1") and hasattr(model, "h2")
    stages = {"H1": model.h1, "H2": model.h2} if connected else {"H1": model}
    priors = {}
    for key, network in stages.items():
        prior = torch.as_tensor(network.config.view_attention_psf_prior, dtype=torch.float32).flatten()
        priors[key] = (prior/prior.sum()).cpu().numpy()
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            for family, index in zip(families, indices):
                if index is None: continue
                sample = dataset[index]
                for gain in gains:
                    lf = sample['lf'][None].to(device) * gain
                    first = stages['H1'](lf, return_intermediates=True)
                    weights = {'H1': first['ratio_attention_weights'].flatten().cpu().numpy()}
                    if connected:
                        second = stages['H2'](lf, first['reconstruction'], return_intermediates=True)
                        weights['H2'] = second['attention_weights'].flatten().cpu().numpy()
                        del second
                    del first, lf
                    for stage, weight in weights.items():
                        if not np.isfinite(weight).all() or weight.sum() <= 0:
                            raise FloatingPointError(f'Nonfinite attention probe: {stage}, {sample["name"]}, gain={gain}')
                        share = weight/weight.sum()
                        relative = share/np.maximum(priors[stage], 1e-12)
                        records.append(dict(stage=stage, family=family, name=sample['name'], gain=gain,
                            shares=share.tolist(), strongest_share=float(share.max()),
                            suppressed_views=int((share < suppression_share).sum()),
                            effective_views=float(1/np.square(share).sum()),
                            minimum_relative_to_prior=float(relative.min()), maximum_relative_to_prior=float(relative.max())))
    finally:
        for module, training in modules: module.training = training
    result = dict(epoch=epoch, input='Fixed clean validation LF, multiplied by gain; no added noise',
                  suppression_share=suppression_share, priors={k:v.tolist() for k,v in priors.items()}, records=records)
    path = output_dir / f'epoch_{epoch:04d}.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig,axes=plt.subplots(len(stages),len(families)+1,figsize=(14,3.8*len(stages)),squeeze=False,layout='constrained')
    for row,stage in zip(axes,stages):
        normal=[next((r for r in records if r['stage']==stage and r['family']==f and r['gain']==1.),None) for f in families]
        arrays=[priors[stage]]+[np.array(r['shares']) if r else None for r in normal]
        limit=max(a.max() for a in arrays if a is not None)*100
        for ax,array,title in zip(row,arrays,['PSF prior']+list(families)):
            if array is None: ax.axis('off');continue
            side=int(len(array)**.5);im=ax.imshow(array.reshape(side,side)*100,vmin=0,vmax=limit,cmap='magma')
            ax.set_title(f'{stage}: {title}\nstrongest {100*array.max():.1f}%',fontsize=9)
            ax.set_xlabel('View column');ax.set_ylabel('View row')
        fig.colorbar(im,ax=list(row),label='Share (%)',shrink=.8)
    fig.suptitle(f'Attention at epoch {epoch}: fixed validation examples, original intensity')
    figure_path=output_dir/f'epoch_{epoch:04d}.png';fig.savefig(figure_path,dpi=130);plt.close(fig)
    history=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(output_dir.glob('epoch_*.json'))]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for record in records:
        key=(record['stage'],record['name'],record['gain'])
        points=[(h['epoch'],r) for h in history for r in h['records'] if (r['stage'],r['name'],r['gain'])==key]
        for ax,metric in zip(axes,('strongest_share','suppressed_views','effective_views')):
            ax.plot([e for e,_ in points],[r[metric] for _,r in points],marker='.',label=f'{key[0]} {record["family"]} {key[2]:g}x')
            ax.set_title(metric.replace('_',' '));ax.set_xlabel('Epoch')
    axes[-1].legend(fontsize=5,ncol=2);fig.savefig(output_dir/'attention_history.png',dpi=130);plt.close(fig)
    return result, figure_path
