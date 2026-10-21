"""Background-map preparation for a sparse, stationary illuminated stack."""

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import hashlib
import tifffile


def unique_tiff_frames(path):
    """Yield consecutive distinct native frames; repeat lengths need not be 3."""
    previous_digest = None
    with tifffile.TiffFile(path) as tif:
        for page_index, page in enumerate(tif.pages):
            frame = page.asarray()
            digest = hashlib.blake2b(frame, digest_size=16).digest()
            if digest != previous_digest:
                yield page_index, frame
            previous_digest = digest


def calibrate_stationary_stack(path):
    """MATLAB median/MAD and ACF estimators, applied to exactly deduplicated pages."""
    starts, frames = [], []
    for page, frame in unique_tiff_frames(path):
        starts.append(page)
        frames.append(frame)
    data = np.asarray(frames)
    del frames
    if data.ndim != 3 or len(data) < 30:
        raise ValueError("A stationary stack needs at least 30 distinct 2D frames")
    n, h, w = data.shape
    common = data.mean(axis=(1, 2))
    common -= common.mean()
    cal = {key: np.empty(h*w, dtype=np.float32) for key in ("offset", "sigmaRead", "sigmaTotal")}
    pixels = data.reshape(n, -1)
    for first in range(0, h*w, 32768):
        last = min(h*w, first+32768)
        x = pixels[:, first:last].astype(np.float32) - common.astype(np.float32)[:, None]
        offset = np.median(x, axis=0)
        cal["offset"][first:last] = offset
        cal["sigmaRead"][first:last] = 1.4826 * np.median(np.abs(x-offset), axis=0)
        cal["sigmaTotal"][first:last] = x.std(axis=0, ddof=1)
    cal = {key: value.reshape(h, w) for key, value in cal.items()}
    med = np.median(cal["sigmaRead"])
    robust_sd = 1.4826 * np.median(np.abs(cal["sigmaRead"] - med))
    cal["isHot"] = cal["sigmaTotal"] > med + 8 * robust_sd
    cal["sigmaCommon"] = np.array(np.std(common))
    block_h, block_w = min(400, h), min(400, w)
    yy, xx = (h-block_h)//2, (w-block_w)//2
    block = data[:, yy:yy+block_h, xx:xx+block_w].astype(np.float32)
    block -= common.astype(np.float32)[:, None, None]
    block -= block.mean(axis=0)
    block /= block.std(axis=0, ddof=1).clip(1e-6)
    for axis in (1, 2):
        max_lag = min(8, block.shape[axis]-1)
        acf = []
        for lag in range(max_lag+1):
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis] = slice(0, block.shape[axis]-max_lag)
            b[axis] = slice(lag, block.shape[axis]-max_lag+lag)
            acf.append(float(np.mean(block[tuple(a)] * block[tuple(b)])))
        cal[f"acf{axis}"] = np.asarray(acf) / acf[0]
    cal["nFrames"] = np.array(n)
    cal["size"] = np.array([h, w])
    with tifffile.TiffFile(path) as tif:
        pages = len(tif.pages)
    lengths = np.diff(starts + [pages])
    return cal, {"pages": pages, "unique_frames": n,
                 "repeat_lengths": {str(k): int((lengths == k).sum()) for k in np.unique(lengths)},
                 "deduplication": "exact consecutive page content (BLAKE2b-128)",
                 "first_page_of_unique_frames": starts}


def mask_stationary_signal(cal: dict, *, threshold_adu: float = 8., dilation: int = 3):
    """Repair scene-contaminated pixels; retain measured background elsewhere.

    A robust additive row/column fit estimates electronic fixed pattern. Positive
    departures above threshold identify sparse scene signal. This is an explicit
    approximation for the mostly-background 4-beads stack, not a dark calibration.
    """
    offset = np.asarray(cal["offset"], dtype=np.float64)
    if offset.ndim != 2 or not np.isfinite(offset).all() or threshold_adu <= 0 or dilation < 0:
        raise ValueError("Need a finite 2D offset map, positive threshold and nonnegative dilation")
    column = np.median(offset, axis=0, keepdims=True)
    for _ in range(6):
        row = np.median(offset - column, axis=1, keepdims=True)
        column = np.median(offset - row, axis=0, keepdims=True)
    baseline = row + column
    mask = offset - baseline > threshold_adu
    if dilation:
        mask = binary_dilation(mask, iterations=dilation)
    if mask.mean() > .4:
        raise ValueError("Too much scene signal for a sparse-scene background calibration")
    result = {key: np.asarray(value).copy() for key, value in cal.items()}
    result["offset"][mask] = baseline[mask]
    weights = gaussian_filter((~mask).astype(float), 12., mode="reflect")
    for key in ("sigmaRead", "sigmaTotal"):
        values = np.asarray(cal[key], dtype=np.float64)
        filled = gaussian_filter(np.where(mask, 0, values), 12., mode="reflect") / weights.clip(1e-8)
        filled[weights < 1e-4] = np.median(values[~mask])
        result[key][mask] = filled[mask]
    # A high-variance bead must not be baked in as a persistent hot pixel.
    result["isHot"][mask] = False
    result["sceneMask"] = mask
    return result, {
        "method": "robust additive row/column baseline; positive residual mask; dilated repair",
        "threshold_adu": threshold_adu, "dilation_pixels": dilation,
        "masked_pixels": int(mask.sum()), "masked_fraction": float(mask.mean()),
        "offset_median_adu": float(np.median(result["offset"])),
        "sigma_read_median_adu": float(np.median(result["sigmaRead"])),
        "remaining_hot_pixels": int(result["isHot"].sum()),
        "limitation": "Masked baseline and noise are interpolated estimates. A matching dark/blank stack would remove this ambiguity.",
    }
