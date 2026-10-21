"""Portable H1/H2/fusion inference and bead speed/geometry evaluation."""
from pathlib import Path
import argparse, csv, hashlib, json, platform, sys, time
import numpy as np
import torch
import tifffile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent/'src/lfrln').is_dir() else HERE
sys.path.insert(0, str(ROOT/'src'))
from lfrln.training import load_h1_h2_training_checkpoint
from lfrln.data import load_lf_tensor

SPACING = (.5, .21306818181818185, .21306818181818185)


def read_registered(path, key='LF'):
    """Return registered sensor counts in Y,X order; never silently register/crop."""
    path = Path(path)
    if path.suffix.lower() == '.mat': return load_lf_tensor(path, key)[0].numpy()
    if path.suffix.lower() == '.npy': return np.load(path, allow_pickle=False)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z: return z[key].copy()
    if path.suffix.lower() in ['.tif', '.tiff']: return tifffile.imread(path)
    raise ValueError('Use a registered 2D .mat, .npy, .npz or .tif frame')


def preprocess(counts, background, scale=3000., normalized=False):
    a = np.asarray(counts, dtype=np.float32)
    if a.shape != (517,517) or not np.isfinite(a).all():
        raise ValueError(f'Expected finite registered Y,X frame (517,517); got {a.shape}')
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(background):
        raise ValueError('Background must be finite and count scale positive')
    if normalized:
        if a.min() < 0: raise ValueError('Normalized LF must be nonnegative')
        a = a.copy()
    else: a = np.maximum(a-background, 0.)/scale
    return torch.from_numpy(np.ascontiguousarray(a))[None,None]


class LFBeadReconstructor:
    """Reusable API: registered camera counts -> float32 volume [Z,Y,X]."""
    def __init__(self, checkpoint, device='auto', threads=4, branch='final'):
        if branch not in ['final','h1']: raise ValueError('branch must be final or h1')
        self.branch = branch
        self.output_domain = 'raw emitter estimate' if branch=='final' else 'TF-constrained/cutoff observation volume'
        if threads <= 0: raise ValueError('threads must be positive')
        torch.set_num_threads(threads)
        if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; install a CUDA PyTorch build or use --device cpu')
        # Explicit reference precision. No AMP/TF32 or automatic architecture changes.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model, self.payload = load_h1_h2_training_checkpoint(checkpoint, device='cpu')
        self.model.to(self.device).eval().requires_grad_(False)

    def synchronize(self):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _predict(self, prepared):
        return self.model.forward_h1(prepared) if self.branch=='h1' else self.model(prepared)

    @torch.inference_mode()
    def tensor(self, prepared): return self._predict(prepared.to(self.device))

    @torch.inference_mode()
    def reconstruct(self, counts, *, background, scale=3000., normalized=False):
        x = preprocess(counts, background, scale, normalized)
        return self.tensor(x)[0].cpu().numpy()

    @torch.inference_mode()
    def benchmark(self, counts, *, background, scale=3000., normalized=False, warmup=3, repeats=20):
        if warmup < 0 or repeats <= 0: raise ValueError('Invalid timing counts')
        prepared = preprocess(counts, background, scale, normalized).to(self.device)
        self.synchronize(); t=time.perf_counter()
        prediction = self._predict(prepared)
        self.synchronize(); cold_ms=(time.perf_counter()-t)*1000
        del prediction
        for _ in range(warmup): self._predict(prepared)
        self.synchronize()
        if self.device.type=='cuda': torch.cuda.reset_peak_memory_stats(self.device)
        network=[]; end_to_end=[]
        for _ in range(repeats):
            if self.device.type=='cuda':
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.cuda.device(self.device):
                    start.record(); result=self._predict(prepared); end.record()
                end.synchronize(); network.append(start.elapsed_time(end))
            else:
                t=time.perf_counter(); result=self._predict(prepared)
                network.append((time.perf_counter()-t)*1000)
            del result
        for _ in range(repeats):
            self.synchronize(); t=time.perf_counter()
            out=self.reconstruct(counts,background=background,scale=scale,normalized=normalized)
            self.synchronize(); end_to_end.append((time.perf_counter()-t)*1000)
            del out
        def stats(v):
            return dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),frames_per_second=1000/float(np.mean(v)))
        return dict(output_branch=self.branch,output_domain=self.output_domain,batch_size=1,shape_in=[1,1,517,517],shape_out=[1,91,517,517],precision='FP32; TF32 off; no AMP',
            warmup=warmup,repetitions=repeats,first_forward_ms=cold_ms,network_only=stats(network),
            host_counts_to_host_volume=stats(end_to_end),
            timing_scope='Network-only uses resident-device input and output. Host-to-host includes preprocessing, transfers and inference; excludes disk loading, plots and saving.',
            peak_allocated_gpu_mib=torch.cuda.max_memory_allocated(self.device)/2**20 if self.device.type=='cuda' else None)


def bead_metrics(volume, reference):
    """Local emitter geometry against conditional fitted poses, not voxel GT."""
    z=(np.arange(volume.shape[0])-(volume.shape[0]-1)/2)*SPACING[0]
    cy,cx=reference['source_center_full_yx']
    y0,y1=max(0,int(cy)-22),min(volume.shape[1],int(cy)+23)
    x0,x1=max(0,int(cx)-22),min(volume.shape[2],int(cx)+23)
    yy,xx=np.meshgrid((np.arange(y0,y1)-cy)*SPACING[1],(np.arange(x0,x1)-cx)*SPACING[2],indexing='ij')
    local=volume[:,y0:y1,x0:x1].astype(np.float64)
    disk=yy**2+xx**2 <= 3.**2
    profile=(local*disk).sum((1,2)); mass=float(profile.sum())
    target=np.asarray(reference['center_zyx_um'],dtype=float)
    if mass<=0: return dict(geometry_valid=False),profile
    centroid=np.array([(profile*z).sum()/mass,(local*disk*yy).sum()/mass,(local*disk*xx).sum()/mass])
    peak=int(np.argmax(profile)); left=right=peak
    while left>0 and profile[left-1]>=profile[peak]/2: left-=1
    while right<len(z)-1 and profile[right+1]>=profile[peak]/2: right+=1
    return dict(geometry_valid=True,reference_kind='Conditional bead pose fitted with calibrated LF operator; not independent voxel ground truth',
        expected_z_um=float(target[0]),peak_z_um=float(z[peak]),centroid_z_um=float(centroid[0]),
        peak_z_error_um=float(z[peak]-target[0]),centroid_z_error_um=float(centroid[0]-target[0]),
        lateral_centroid_error_um=float(np.linalg.norm(centroid[1:]-target[1:])),
        axial_fwhm_sampled_um=float((right-left+1)*SPACING[0]),
        fwhm_truncated_at_boundary=left==0 or right==len(z)-1,
        off_reference_5um_mass_fraction=float(profile[np.abs(z-target[0])>5].sum()/mass),
        local_mass=mass,volume_min=float(volume.min()),volume_max=float(volume.max())),profile


def comparison_map(raw, volume, reference, destination, title, branch='final'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    ax[0,0].imshow(np.log1p(np.maximum(raw,0)),cmap='gray'); ax[0,0].set_title('Registered LF input (log display)')
    coarse=np.maximum(raw,0).reshape(47,11,47,11).sum((1,3))
    ax[0,1].imshow(coarse,cmap='gray'); ax[0,1].set_title('LF intensity summed within lenslets')
    xy=volume.max(0); top=max(float(np.quantile(xy,.999)),1e-12)
    ax[0,2].imshow(xy,cmap='magma',vmin=0,vmax=top)
    ax[0,2].set_title('Final reconstruction XY maximum' if branch=='final' else 'H1 cutoff/TF volume XY maximum')
    ax[1,0].imshow(volume.max(1),cmap='magma',aspect='auto',extent=[0,517*SPACING[2],22.75,-22.75],vmin=0,vmax=top)
    ax[1,0].set(xlabel='X (µm)',ylabel='Z (µm)',title='XZ maximum')
    ax[1,1].imshow(volume.max(2),cmap='magma',aspect='auto',extent=[0,517*SPACING[1],22.75,-22.75],vmin=0,vmax=top)
    ax[1,1].set(xlabel='Y (µm)',ylabel='Z (µm)',title='YZ maximum')
    if reference:
        metrics,profile=bead_metrics(volume,reference)
        ax[1,2].plot((np.arange(91)-45)*.5,profile/max(float(profile.max()),1e-12),label='Reconstruction, local 3 µm radius')
        ax[1,2].axvline(reference['center_zyx_um'][0],color='green',ls='--',label='Conditional fitted bead Z')
        cy,cx=reference['source_center_full_yx']; ax[0,2].plot(cx,cy,'g+',ms=12)
    else:
        profile=volume.sum((1,2)); ax[1,2].plot((np.arange(91)-45)*.5,profile/max(float(profile.max()),1e-12),label='Full-volume depth profile')
    ax[1,2].set(xlabel='Z (µm)',ylabel='Normalized intensity',title='Axial distribution'); ax[1,2].legend(fontsize=8)
    fig.suptitle(title); fig.savefig(destination,dpi=130); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    inputs=ap.add_mutually_exclusive_group()
    inputs.add_argument('--suite',type=Path,help='Portable bead-example manifest')
    inputs.add_argument('--input',type=Path,help='One registered Y,X LF image; raw unregistered camera frames are not accepted')
    ap.add_argument('--model',type=Path,default=ROOT/'model.pt')
    ap.add_argument('--branch',choices=['final','h1'],default='final',help='final: H1/H2/fusion raw estimate; h1: H1-only cutoff/TF volume')
    ap.add_argument('--key',default='LF'); ap.add_argument('--background',type=float)
    ap.add_argument('--count-scale',type=float,default=3000.)
    ap.add_argument('--normalized',action='store_true',help='Input already equals clamp(counts-background)/3000')
    ap.add_argument('--device',default='auto'); ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--warmup',type=int,default=3); ap.add_argument('--repeats',type=int,default=20)
    ap.add_argument('--limit',type=int,default=0,help='0 evaluates the full supplied suite')
    ap.add_argument('--save-volumes',action='store_true',help='Save full float32 Z,Y,X TIFF volumes (~97 MB/frame)')
    ap.add_argument('--benchmark-only',action='store_true')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.limit<0: ap.error('--limit must be nonnegative')
    if a.input:
        if a.background is None and not a.normalized: ap.error('Specify --background for camera counts, or --normalized for prepared input')
        rows=[dict(name=a.input.stem,path=str(a.input.resolve()),background_adu=a.background or 0.,reference=None)]
        base=Path('.')
    else:
        suite=a.suite or ROOT/'examples/suite.json'
        payload=json.loads(suite.read_text(encoding='utf8')); rows=payload['samples']; base=suite.parent
    if a.limit: rows=rows[:a.limit]
    if not rows: ap.error('No input frames')
    if a.output.exists(): raise FileExistsError('Use a fresh --output directory; existing results are protected')
    start=time.perf_counter(); runner=LFBeadReconstructor(a.model,a.device,a.threads,branch=a.branch)
    runner.synchronize(); load_ms=(time.perf_counter()-start)*1000
    first=read_registered(base/rows[0]['path'],a.key)
    timing=runner.benchmark(first,background=rows[0]['background_adu'],scale=a.count_scale,normalized=a.normalized,warmup=a.warmup,repeats=a.repeats)
    a.output.mkdir(parents=True)
    timing.update(model_load_ms=load_ms,epoch=runner.payload['epoch'],model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        hardware=dict(platform=platform.platform(),python=sys.version.split()[0],torch=torch.__version__,cuda_runtime=torch.version.cuda,
            device=str(runner.device),gpu=torch.cuda.get_device_name(runner.device) if runner.device.type=='cuda' else None,cpu_threads=a.threads))
    (a.output/'timing.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
    print(json.dumps(timing,indent=2),flush=True)
    if a.benchmark_only: return
    metrics=[]
    for i,row in enumerate(rows):
        t=time.perf_counter(); raw=read_registered(base/row['path'],a.key); read_ms=(time.perf_counter()-t)*1000
        volume=runner.reconstruct(raw,background=row['background_adu'],scale=a.count_scale,normalized=a.normalized)
        if volume.shape!=(91,517,517) or not np.isfinite(volume).all() or volume.min()<0: raise ValueError('Invalid reconstructed volume')
        name=row['name'] if a.branch=='final' else row['name']+'_h1'; ref=row.get('reference')
        values,_=bead_metrics(volume,ref) if ref else ({},None)
        metrics.append(dict(name=name,output_branch=a.branch,output_domain=runner.output_domain,split=row.get('split','external'),input_read_ms=read_ms,**values))
        comparison_map(raw,volume,ref,a.output/(name+'.png'),name+' | epoch '+str(runner.payload['epoch']),branch=a.branch)
        if a.save_volumes:
            tifffile.imwrite(a.output/(name+'.tif'),volume.astype(np.float32),metadata=dict(axes='ZYX',spacing_um=list(SPACING),output_branch=a.branch,output_domain=runner.output_domain,intensity='Network relative units; not photon-calibrated'))
        print(f'{i+1}/{len(rows)} {name}: Z peak {values.get("peak_z_um")} µm',flush=True)
        del volume
    (a.output/'metrics.json').write_text(json.dumps(dict(samples=metrics,output_branch=a.branch,output_domain=runner.output_domain,spacing_zyx_um=SPACING,pose_reference='Conditional forward-fitted bead poses; no independent voxel truth'),indent=2)+'\n',encoding='utf8')
    keys=sorted({k for r in metrics for k in r})
    with (a.output/'metrics.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(metrics)
    index=['<!doctype html><meta charset="utf-8"><title>Bead inference</title><h1>LF input and reconstruction comparisons</h1><p>Conditional fitted poses, not independent voxel truth.</p>']
    for row in metrics: index.append(f'<h2>{row["name"]} ({a.branch})</h2><img style="max-width:100%" src="{row["name"]}.png">')
    (a.output/'index.html').write_text('\n'.join(index),encoding='utf8')


if __name__=='__main__': main()
