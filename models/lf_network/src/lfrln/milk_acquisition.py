"""Raw milk-experiment acquisition reader and user-confirmed scan calibration."""
from pathlib import Path
import re
import numpy as np

DATA = Path(r'F:\zr\03_Light Field\Scattering Analysis\Expdata\20260916 test scattering FB')
CONDITIONS = ('clearWater_0p05s', '0p01milk_0p05s', '0p01milk_0p5s')
# Reference bead centers selected in the clear-water raw-array MIP.
CENTERS = {'TF': [(309,346),(227,182),(158,352),(208,330)],
           'LF': [(124,895),(316,788),(418,219)]}

AXIAL_CALIBRATION_NOTE=(
    'User calibration: a nominal 0.56 um PZT command moves 0.5 um physically. '
    'Use frame-index positions at 0.5 um spacing, centered on the scan midpoint; '
    'the sign follows the command trace. vscanpos is retained as an uncalibrated '
    'command trace, not an independent measurement of physical displacement.')


def axial_calibration(folder, frame_numbers):
    command=np.loadtxt(folder/'vscanpos.txt',ndmin=1)
    numbers=np.asarray(frame_numbers,dtype=int)
    if len(command)!=len(numbers) or len(numbers)<2:
        raise ValueError('Scan coordinates and frame count must agree')
    if not np.all(np.diff(numbers)==1):
        raise ValueError('Physical scan axis requires consecutive frame numbers')
    direction=np.sign(np.median(np.diff(command)))
    if direction==0:
        raise ValueError('Cannot infer axial direction from command trace')
    physical=direction*(numbers-(numbers[0]+numbers[-1])/2)*0.5
    return dict(command_positions=command.tolist(),
        command_position_range=[float(command.min()),float(command.max())],
        command_median_step=float(np.median(np.abs(np.diff(command)))),
        physical_positions_um=physical.tolist(),
        physical_position_range_um=[float(physical.min()),float(physical.max())],
        physical_step_um=0.5,nominal_command_step_um=0.56,
        position_origin='Scan midpoint; not an absolute specimen depth',
        calibration_note=AXIAL_CALIBRATION_NOTE)


def metadata(path):
    text=path.read_text(encoding='utf8',errors='replace')
    values={}
    for line in text.splitlines():
        if ':' in line:
            k,v=line.split(':',1);values[k.strip()]=v.strip()
    values.pop('DM_V',None)
    return values


def load_stack(folder):
    meta=metadata(folder/'parameters.txt')
    shape=tuple(int(v) for v in re.findall(r'\d+',meta['Pixels (x, y)']))
    files=sorted((folder/'wbin3').glob('*.wbin3'))
    images=[]
    for f in files:
        data=np.fromfile(f,dtype='<u2')
        if data.size!=np.prod(shape):raise ValueError(f'Unexpected raw size: {f}')
        images.append(data.reshape(shape,order='F'))
    return np.stack(images),meta,files


def border_mask(shape,fraction=.05):
    width=max(8,round(min(shape)*fraction))
    mask=np.ones(shape,bool);mask[width:-width,width:-width]=False
    return mask

