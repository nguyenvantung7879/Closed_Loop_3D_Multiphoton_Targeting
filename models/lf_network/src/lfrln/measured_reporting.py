"""Deterministic training/validation displays, separate from optimization."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch


def plot_measured_history(history):
    import matplotlib.pyplot as plt
    definitions=[('replay','total','Synthetic: final raw + optional H2 raw + H1 TF','Combined synthetic loss'),
                 ('beads','bead_lf','Beads: reconstructed volume → predicted LF','LF shape error (before source coefficient)'),
                 ('biology','bio_tf','Biology: H1 estimate versus measured TF','Axial-weighted TF shape error (before source coefficient)')]
    if any(r.get('validation',{}).get('replay',{}).get('h2_auxiliary_l2',0)>0 for r in history):
        definitions.append(('replay','h2_auxiliary_l2','Synthetic: H2 alone versus known raw volume','H2 raw voxel loss (including auxiliary coefficient)'))
    if any('bio_ps' in r.get('validation',{}).get('biology',{}) for r in history):
        definitions.append(('biology','bio_ps','Biology: PS prediction from final raw volume','Axial-weighted PS shape error (before source coefficient)'))
    if any('bio_ps_projection' in r.get('validation',{}).get('biology',{}) for r in history):
        definitions.extend([('biology','bio_ps_projection','Biology: PS XZ/YZ sum projections','Mean projection L1 (before coefficient)'),
                            ('biology','bio_ps_axial_profile','Biology: PS depth distribution','Axial mass-profile L1 (before coefficient)')])
    fig,axes=plt.subplots(1,len(definitions),figsize=(5*len(definitions),4.6))
    for ax,(kind,key,title,ylabel) in zip(axes,definitions):
        val=[r for r in history if 'validation' in r]
        online=[r for r in history if 'train' in r]
        fixed=[r for r in history if 'fixed_train' in r]
        ax.plot([r['epoch'] for r in val],[r['validation'][kind][key] for r in val],label='Validation: held-out, fixed inputs',color='#1565c0')
        ax.plot([r['epoch'] for r in online],[r['train'][kind][key] for r in online],label='Training: average during weight updates',color='#ef6c00',alpha=.65)
        if fixed:ax.plot([r['epoch'] for r in fixed],[r['fixed_train'][kind][key] for r in fixed],label='Training: fixed subset after epoch',color='#2e7d32',marker='o',markersize=3)
        ax.set(title=title,xlabel='Completed epoch',ylabel=ylabel);ax.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=9)
    fig.suptitle('Lower is better; compare trends within each panel. Fixed subset sizes and specimens differ from validation.',y=.025,fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.9))
    return fig


def projections(value):
    a=np.asarray(value,dtype=np.float32).squeeze()
    if a.ndim==2:return [a,None,None]
    if a.ndim!=3:raise ValueError('Expected an image or Z,Y,X volume')
    m=np.ma.masked_invalid(a)
    return [m.max(axis=i).filled(np.nan) for i in range(3)]


def render_columns(columns,path,title,*,shared_scale=False,domain_note=None,lf_input=None):
    """XY/XZ/YZ MIPs, with physical aspect and independent column intensity."""
    import matplotlib.pyplot as plt
    display_columns=columns
    if lf_input is not None:
        lf=np.asarray(lf_input).squeeze()
        if lf.ndim!=2:raise ValueError('LF input display requires a two-dimensional detector image')
        display_columns={'LF input (detector)': [lf,None,None],**columns}
    fig,axes=plt.subplots(3,len(display_columns),figsize=(3.1*len(display_columns),8.1),squeeze=False,
        gridspec_kw={'height_ratios':[1,.44,.44]})
    row_maxima=[max((float(np.nanmax(ims[i])) for ims in columns.values() if ims[i] is not None and np.isfinite(ims[i]).any()),default=1.) for i in range(3)] if shared_scale else None
    for j,(label,ims) in enumerate(display_columns.items()):
        maximum=max((float(np.nanmax(a)) for a in ims if a is not None and np.isfinite(a).any()),default=1.)
        for i,a in enumerate(ims):
            ax=axes[i,j]
            if a is None:ax.axis('off');continue
            if lf_input is not None and j==0:
                ax.imshow(a,origin='lower',cmap='magma',vmin=0,vmax=max(maximum,1e-12))
                ax.set_title(label,fontsize=8)
                ax.set_xlabel('Detector column (px)',fontsize=7);ax.set_ylabel('Detector row (px)',fontsize=7)
                ax.tick_params(labelsize=7)
                continue
            dxy=.21306818181818185;dy=dxy if i==0 else .5
            extent=(-a.shape[1]*dxy/2,a.shape[1]*dxy/2,-a.shape[0]*dy/2,a.shape[0]*dy/2)
            ax.imshow(a,origin='lower',cmap='magma',vmin=0,vmax=max(row_maxima[i] if shared_scale else maximum,1e-12),extent=extent,aspect='equal')
            ax.set_title(label+' / XY' if i==0 else ['XY','XZ','YZ'][i],fontsize=8)
            ax.set_xlabel(['x','x','y'][i]+' (µm)',fontsize=7);ax.set_ylabel(['y','z','z'][i]+' (µm)',fontsize=7)
            ax.tick_params(labelsize=7)
    fig.suptitle(title+'\n'+(domain_note or 'Independent column maxima: morphology only; biological PS is a different observation domain.'),fontsize=10)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(path,dpi=135);plt.close(fig)


def capture(model,sample,device):
    was_training=model.training;model.eval()
    try:
        with torch.inference_mode():
            stages=model.predict_stages(sample['lf'][None].to(device))
            outputs={key:projections(stages[value][0].float().cpu().numpy()) for key,value in [('h1','h1'),('final','merged')]}
        return outputs
    finally:model.train(was_training)


def write_snapshot(model,cases,output,epoch,device):
    """Cases yield fixed (split, kind, sample, optional synthetic TF target)."""
    root=Path(output);folder=root/f'epoch_{epoch:04d}';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    for split,kind,sample,tf_target in cases:
        key=split+'_'+kind+'_'+hashlib.sha256(sample['name'].encode()).hexdigest()[:10]
        result=capture(model,sample,device)
        columns={}
        if kind=='biology':
            columns['Measured TF']=projections(np.where(sample['tf_valid_mask'].numpy(),sample['tf_reference'].numpy(),np.nan))
        elif tf_target is not None:columns['Synthetic TF target']=projections(tf_target.numpy())
        columns['H1 TF estimate']=result['h1']
        if kind=='replay':columns['Raw-volume target']=projections(sample['gt'].numpy())
        elif kind=='biology':columns['Measured PS (different domain)']=projections(sample['gt'].numpy())
        columns['Final raw-volume estimate']=result['final']
        path=folder/(key+'.png');title=f"{split} / {kind} / {sample['name']} / epoch {epoch}"
        render_columns(columns,path,title,lf_input=sample['lf'].numpy())
        entries.append(dict(split=split,kind=kind,name=sample['name'],path=str(path.resolve()),
            input_sha256=hashlib.sha256(sample['lf'].numpy().tobytes()).hexdigest()))
    record=dict(epoch=epoch,examples=entries,interpretation='Fixed evaluation inputs, no optimizer update; previews do not affect selection.')
    (folder/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (root/'latest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record
