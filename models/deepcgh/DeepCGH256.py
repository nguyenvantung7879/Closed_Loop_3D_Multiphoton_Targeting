import os
import warnings
import time
import gc
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat


_ROOT = Path(__file__).resolve().parent
_XY_PIXEL_SIZE_UM = 0.6483078
_Z_ZERO_INDEX = 10.0
_Z_LAYER_SPACING_UM = 2.0
_DEFAULT_GREY_COEFF = _ROOT / "Grey_coeff.npy"
# Match the validated 1024 backend physical/world convention.
_X_INPUT_SIGN = -1.0
_Y_INPUT_SIGN = -1.0
_MODEL_DIR = _ROOT / "DeepCGH_Models256"



def optimize_torch_inference():
    """Enable safe PyTorch runtime options for repeated inference.

    This does not change the network architecture or propagation code.
    It only lets cuDNN choose faster kernels for the fixed input size.
    """
    status = {
        "cuda": bool(torch.cuda.is_available()),
        "device": None,
        "cudnn_benchmark": False,
        "tf32": False,
    }

    if torch.cuda.is_available():
        try:
            status["device"] = torch.cuda.get_device_name(0)
        except Exception:
            status["device"] = "cuda"

        torch.backends.cudnn.benchmark = True
        status["cudnn_benchmark"] = True

        # Mostly useful for float32 kernels on Ampere+ GPUs.
        # AMP/float16 inference remains controlled by prepare_inference(use_amp=True).
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            status["tf32"] = True
        except Exception:
            pass

    return status



def _as_point_array(points):
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError("points must have columns [x,y,z_index,intensity]")
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("points must have shape [N,4] with columns [x,y,z_index,intensity]")
    return arr


def _point_to_yxz(point):
    x, y, z, intensity = point
    return int(np.rint(y)), int(np.rint(x)), int(np.rint(z)), float(intensity)


def _point_patch(h, w, y, x, intensity, gaussian_diameter_um=None, pixel_size_um=None):
    use_gaussian = gaussian_diameter_um is not None or pixel_size_um is not None
    if not use_gaussian:
        return y, y + 1, x, x + 1, np.array([[intensity]], dtype=np.float32)

    if gaussian_diameter_um is None or pixel_size_um is None:
        raise ValueError("Both gaussian_diameter_um and pixel_size_um must be provided")

    gaussian_diameter_um = float(gaussian_diameter_um)
    pixel_size_um = float(pixel_size_um)
    if gaussian_diameter_um <= 0:
        raise ValueError("gaussian_diameter_um must be positive")
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")

    radius_px = (gaussian_diameter_um / 2.0) / pixel_size_um
    pad = int(np.ceil(3.0 * radius_px))
    y_min = max(0, y - pad)
    y_max = min(h, y + pad + 1)
    x_min = max(0, x - pad)
    x_max = min(w, x + pad + 1)

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    rr2 = (yy - y) ** 2 + (xx - x) ** 2
    gaussian = np.exp(-2.0 * rr2 / (radius_px ** 2))
    patch = intensity * gaussian.astype(np.float32)
    return y_min, y_max, x_min, x_max, patch


def _points_to_dpm(
    points,
    shape,
    layer,
    threshold=0.2,
    gaussian_diameter_um=None,
    pixel_size_um=None,
    return_target=False,
):
    if len(shape) != 3:
        raise ValueError("shape must be [H,W,Z]")

    h, w, z_count = [int(v) for v in shape]
    z0 = -((z_count - 1) / 2.0) * float(layer)
    z_values = z0 + np.arange(z_count, dtype=np.float32) * float(layer)

    dpm = np.zeros((h, w, 2), dtype=np.float32)
    z_patches = {}

    for point in _as_point_array(points):
        y, x, z, intensity = _point_to_yxz(point)
        if not (0 <= y < h and 0 <= x < w and 0 <= z < z_count):
            raise ValueError(
                f"Point out of range: y={y}, x={x}, z_index={z}, shape={(h, w, z_count)}"
            )

        y_min, y_max, x_min, x_max, patch = _point_patch(
            h,
            w,
            y,
            x,
            intensity,
            gaussian_diameter_um=gaussian_diameter_um,
            pixel_size_um=pixel_size_um,
        )
        z_patches.setdefault(z, []).append((y_min, y_max, x_min, x_max, patch))

    target = np.zeros((h, w, z_count), dtype=np.float32) if return_target else None
    for z, patches in z_patches.items():
        y0 = min(p[0] for p in patches)
        y1 = max(p[1] for p in patches)
        x0 = min(p[2] for p in patches)
        x1 = max(p[3] for p in patches)
        layer_img = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for y_min, y_max, x_min, x_max, patch in patches:
            layer_img[y_min - y0:y_max - y0, x_min - x0:x_max - x0] += patch

        active = layer_img >= float(threshold)
        dpm[y0:y1, x0:x1, 0] += np.where(active, layer_img, 0.0)
        dpm[y0:y1, x0:x1, 1] += active * z_values[z]
        if target is not None:
            target[y0:y1, x0:x1, z] = layer_img

    if return_target:
        return dpm, target
    return dpm

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )

    def forward(self, x):
        return self.net(x)


class ConvConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class DeepCGHNet(nn.Module):
    def __init__(
        self,
        shape,
        n_kernels,
        input_channels,
        int_factor,
    ):
        super().__init__()
        self.shape = shape
        self.input_channels = int(input_channels)
        self.int_factor = int(int_factor)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be a positive integer")
        if self.int_factor <= 0:
            raise ValueError("int_factor must be a positive integer")

        in_ch = self.input_channels * (self.int_factor ** 2)
        self.enc1 = DoubleConv(in_ch, n_kernels[0])
        self.enc2 = DoubleConv(n_kernels[0], n_kernels[1])
        self.bottleneck = ConvConv(n_kernels[1], n_kernels[2])
        self.dec2 = ConvConv(n_kernels[2] + n_kernels[1], n_kernels[1])
        self.dec1 = ConvConv(n_kernels[1] + n_kernels[0], n_kernels[0])
        self.backbone_out = ConvConv(n_kernels[0], n_kernels[1])
        self.pool = nn.MaxPool2d(2, 2)

        self.phi_head = nn.Conv2d(n_kernels[1] + in_ch, self.int_factor ** 2, kernel_size=3, padding=1)
        self.amp_head = nn.Conv2d(n_kernels[1] + in_ch, self.int_factor ** 2, kernel_size=3, padding=1)

    def _interleave(self, x):
        if self.int_factor == 1:
            return x
        h = x.shape[-2]
        w = x.shape[-1]
        if h % self.int_factor != 0 or w % self.int_factor != 0:
            raise ValueError(
                f"Input spatial size {(h, w)} must be divisible by int_factor={self.int_factor}"
            )
        return F.pixel_unshuffle(x, self.int_factor)

    def _deinterleave(self, x):
        if self.int_factor == 1:
            return x
        return F.pixel_shuffle(x, self.int_factor)

    def _ifft_AmPh(self, amp_0, phi_0):
        amp = amp_0.squeeze(1)
        phi = phi_0.squeeze(1)
        img = amp.to(torch.complex64) * torch.exp(1j * phi.to(torch.complex64))
        img = torch.fft.ifftshift(img, dim=(-2, -1))
        fft = torch.fft.ifft2(img) * float(self.shape[1])
        phase = torch.angle(fft).unsqueeze(1)
        return phase

    def forward(self, x):
        x1_1 = self._interleave(x)
        x1 = self.enc1(x1_1)
        x2 = self.enc2(self.pool(x1))
        encoded = self.bottleneck(self.pool(x2))

        x3 = F.interpolate(encoded, size=x2.shape[-2:], mode="nearest")
        x3 = self.dec2(torch.cat([x3, x2], dim=1))

        x4 = F.interpolate(x3, size=x1.shape[-2:], mode="nearest")
        x4 = self.dec1(torch.cat([x4, x1], dim=1))
        x4 = self.backbone_out(x4)
        x4 = torch.cat([x4, x1_1], dim=1)

        phi_0_ = self.phi_head(x4)
        phi_0 = self._deinterleave(phi_0_)
        amp_0_ = F.relu(self.amp_head(x4), inplace=True)
        amp_0 = self._deinterleave(amp_0_)

        phi_slm = self._ifft_AmPh(amp_0, phi_0)
        return phi_slm


class DeepCGH(object):


    def __init__(self, data_params, model_params):
        self.filename = model_params["filename"]
        self.path = model_params["path"]
        self.shape = data_params["shape"]
        self.object_size = data_params["object_size"]
        self.plane_distance = model_params["plane_distance"]
        self.n_kernels = model_params["n_kernels"]
        self.wavelength = model_params["wavelength"]
        self.f = model_params["focal_length"]
        self.ps = model_params["pixel_size"]
        self.object_type = data_params["object_type"]
        self.centralized = data_params["centralized"]
        self.dpm_threshold = float(data_params.get("dpm_threshold", 0.2))
        self.dpm_layer = data_params.get("layer", 1)
        self.input_name = model_params["input_name"]
        self.output_name = model_params["output_name"]
        self.model = None
        self._loaded_weights_file = None
        self._loaded_weights_mtime = None
        self._inference_ready = False
        self._inference_amp = False
        self._inference_channels_last = False
        self.input_channels = None
        self.int_factor = int(model_params.get("int_factor", 1))
        if self.int_factor <= 0:
            raise ValueError("model_params['int_factor'] must be a positive integer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.__check_avalability()

    def __check_avalability(self):
        # The fast backend keeps weights in a stable direct folder:
        # utils/DeepCGH_Models256/{best.pt,last.pt}
        self.absolute_file_path = str(Path(self.path).expanduser().resolve())
        self.last_weights_file = os.path.join(self.absolute_file_path, "last.pt")
        self.best_weights_file = os.path.join(self.absolute_file_path, "best.pt")
        if not os.path.exists(self.absolute_file_path):
            warnings.warn(f"No trained model found at:\n {self.absolute_file_path}")

    def _set_input_channels(self, input_channels, source):
        input_channels = int(input_channels)
        if input_channels <= 0:
            raise ValueError(f"{source} input channels must be a positive integer")
        if self.input_channels is None:
            self.input_channels = input_channels
        elif self.input_channels != input_channels:
            raise ValueError(
                f"Input channel mismatch: existing={self.input_channels}, "
                f"{source}={input_channels}"
            )

    def __get_model_fn(self):
        if self.input_channels is None:
            raise RuntimeError("UNet input channels must be detected before initializing the model")
        return DeepCGHNet(
            shape=self.shape,
            n_kernels=self.n_kernels,
            input_channels=self.input_channels,
            int_factor=self.int_factor,
        ).to(self.device)


    def _load_weights_if_available(self):
        if self.model is None:
            self.model = self.__get_model_fn()
        infer_file = None
        if os.path.exists(self.best_weights_file):
            infer_file = self.best_weights_file
        elif os.path.exists(self.last_weights_file):
            infer_file = self.last_weights_file
        if infer_file is not None:
            infer_mtime = os.path.getmtime(infer_file)
            if (
                self._loaded_weights_file == infer_file
                and self._loaded_weights_mtime == infer_mtime
            ):
                self.model.eval()
                return

            ckpt = torch.load(infer_file, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            self._loaded_weights_file = infer_file
            self._loaded_weights_mtime = infer_mtime
            self.model.eval()

    def points_to_dpm(
        self,
        points,
        gaussian_diameter_um=None,
        pixel_size_um=None,
        return_target=False,
    ):
        return _points_to_dpm(
            points,
            shape=self.shape,
            layer=self.dpm_layer,
            threshold=self.dpm_threshold,
            gaussian_diameter_um=gaussian_diameter_um,
            pixel_size_um=pixel_size_um,
            return_target=return_target,
        )

    def prepare_inference(self, input_channels=None, use_amp=False, channels_last=False):
        """Load the model once and configure its low-overhead inference path."""
        if input_channels is not None:
            self._set_input_channels(int(input_channels), "inference preload")
        self._load_weights_if_available()
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self._inference_amp = bool(use_amp and self.device.type == "cuda")
        self._inference_channels_last = bool(channels_last and self.device.type == "cuda")
        if self._inference_channels_last:
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()
        self._inference_ready = True
        return self

    def get_hologram(self, inputs, return_unet_time=False, warmup=0, repeats=1):
        arr = np.asarray(inputs, dtype=np.float32)
        if arr.ndim in (3, 4) and int(arr.shape[-1]) == int(self.shape[-1]):
            raise ValueError(
                "get_hologram expects 2-channel DPM input. "
                "Use SLM_phase(point_targets) or show_reconstruction(point_targets) "
                "to convert coordinates directly."
            )

        x = torch.from_numpy(arr)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError("inputs must have shape [N,H,W,2] or [H,W,2]")
        if int(x.shape[-1]) != 2:
            raise ValueError("inputs must be 2-channel DPM with shape [N,H,W,2] or [H,W,2]")
        self._set_input_channels(int(x.shape[-1]), "inference input")
        if not self._inference_ready:
            self._load_weights_if_available()
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        x = x.permute(0, 3, 1, 2).to(self.device)
        if self._inference_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        self.model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self._inference_amp,
        ):
            unet_time = None
            if return_unet_time:
                for _ in range(max(int(warmup), 0)):
                    _ = self.model(x)
                repeats = max(int(repeats), 1)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                start_time = time.perf_counter()
                pred = None
                for _ in range(repeats):
                    pred = self.model(x)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                unet_time = (time.perf_counter() - start_time) / repeats
            else:
                pred = self.model(x)

        pred = pred.permute(0, 2, 3, 1).cpu().numpy()
        if return_unet_time:
            return pred, unet_time
        return pred


def save_slm_phase(phases, path="SLM_phase.mat"):
    """Save one inferred SLM phase immediately, without plotting or compression."""
    if isinstance(phases, torch.Tensor):
        phase = phases.detach().cpu().numpy()
    else:
        phase = np.asarray(phases)

    if phase.ndim == 4:
        if phase.shape[0] != 1:
            raise ValueError("save_slm_phase expects a single phase map")
        phase = phase[0]
    if phase.ndim == 3 and phase.shape[-1] == 1:
        phase = phase[..., 0]
    if phase.ndim not in (2, 3):
        raise ValueError("phase must have shape [H,W], [H,W,1], or [1,H,W,1]")

    savemat(
        path,
        {"SLM_phase": np.asarray(phase, dtype=np.float32, order="C")},
        appendmat=False,
        do_compression=False,
    )
    return path


@lru_cache(maxsize=4)
def _load_grey_coeff(grey_coeff_path):
    """Load phase-to-graylevel polynomial coefficients from a .npy file."""
    path = Path(grey_coeff_path).expanduser()
    if not path.is_file() and path.suffix == "":
        path = path.with_suffix(".npy")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Grey coefficient file not found: {path}")

    # float32 is enough for final uint8 SLM greylevel output and avoids
    # casting the cached coefficients on every realtime hologram call.
    coeff = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
    if coeff.ndim != 1 or coeff.size < 2:
        raise ValueError("Grey coefficient file must contain a 1D coefficient array")
    return coeff


def save_slm_pattern(pattern, path):
    """Save final SLM-ready uint8 graylevel pattern.

    Supported by PIL using file extension, for example .tif, .tiff, .bmp, .png.
    This saves the graylevel SLM pattern, not the phase map.
    """
    if path is None:
        return None

    arr = np.asarray(pattern)
    if arr.ndim != 2:
        raise ValueError("pattern must have shape [H,W]")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    path = Path(path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".tif")
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    Image.fromarray(arr, mode="L").save(str(path))
    return str(path)


def _repeat_nearest_uint8(src, scale=4, out=None):
    """Nearest-neighbor repeat scale for uint8 2D image.

    This avoids two temporary arrays from np.repeat(np.repeat(...)).
    If out is provided, it must have shape [H*scale, W*scale].
    """
    src = np.asarray(src, dtype=np.uint8)
    if src.ndim != 2:
        raise ValueError("src must have shape [H,W]")

    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be a positive integer")
    if scale == 1:
        if out is None:
            return src.copy()
        if out.shape != src.shape:
            raise ValueError(f"out shape must be {src.shape}, got {out.shape}")
        out[...] = src
        return out

    h, w = src.shape
    out_shape = (h * scale, w * scale)
    if out is None:
        out = np.empty(out_shape, dtype=np.uint8)
    else:
        if out.shape != out_shape:
            raise ValueError(f"out shape must be {out_shape}, got {out.shape}")
        if out.dtype != np.uint8:
            raise ValueError("out dtype must be uint8")

    out.reshape(h, scale, w, scale)[:] = src[:, None, :, None]
    return out


def phase_to_greylevel(
    phase,
    save=None,
    output_path=None,
    grey_coeff_path=_DEFAULT_GREY_COEFF,
    scale=4,
    vertical_shift_pixels=0,
    out=None,
):
    """Convert a 2D phase map into an SLM-ready 8-bit graylevel pattern.

    Exact polynomial version:
        phase -> wrapped phase -> calibrated graylevel -> nearest-neighbor scale.

    This keeps the original polynomial conversion, but evaluates it on the
    256x256 phase map before scaling to 1024x1024.
    """
    if output_path is not None:
        if save is not None and str(save) != str(output_path):
            raise ValueError("Use either save or output_path, not both with different paths")
        save = output_path

    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim != 2:
        raise ValueError("phase must have shape [H,W]")

    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be a positive integer")

    coeff = _load_grey_coeff(str(Path(grey_coeff_path).resolve()))

    # Original logic: wrapped = mod(phase + 2*pi, 2*pi), then polyval(coeff, wrapped/pi).
    x = np.mod(phase + np.float32(2.0 * np.pi), np.float32(2.0 * np.pi)) * np.float32(1.0 / np.pi)

    # Same polynomial as np.polyval, written as Horner form to reduce overhead.
    y = np.empty_like(x, dtype=np.float32)
    y.fill(float(coeff[0]))
    for c in coeff[1:]:
        y *= x
        y += np.float32(c)

    greylevel = np.abs(y)
    greylevel = np.floor(greylevel + np.float32(0.5))
    greylevel = np.clip(greylevel, 0, 255).astype(np.uint8)

    if vertical_shift_pixels == 0:
        pattern = _repeat_nearest_uint8(greylevel, scale=scale, out=out)
    else:
        shift = int(vertical_shift_pixels) * scale
        if out is None:
            pattern = np.zeros((greylevel.shape[0] * scale, greylevel.shape[1] * scale), dtype=np.uint8)
        else:
            pattern = out
            pattern.fill(0)

        tmp = _repeat_nearest_uint8(greylevel, scale=scale, out=None)
        if shift <= 0:
            pattern[...] = tmp
        elif shift < pattern.shape[0]:
            pattern[:-shift, :] = tmp[shift:, :]

    if save is not None:
        save_slm_pattern(pattern, save)

    return pattern

def _z_um_to_indices(z_um):
    layer_offset = np.asarray(z_um, dtype=np.float64) / _Z_LAYER_SPACING_UM
    abs_offset = np.abs(layer_offset)
    lower_offset = np.floor(abs_offset)
    fractional = abs_offset - lower_offset

    rounded_abs = np.floor(abs_offset + 0.5)
    halfway = np.isclose(fractional, 0.5, rtol=0.0, atol=1e-7)
    rounded_abs = np.where(halfway, lower_offset, rounded_abs)

    return _Z_ZERO_INDEX - np.sign(layer_offset) * rounded_abs


def points_um_to_indices(point_targets, pixel_size_um=_XY_PIXEL_SIZE_UM):
    """Convert centered [x_um, y_um, z_um, intensity] into model voxel indices.

    Coordinate convention:
        x_um = 0, y_um = 0 is the center of CGH field.
        x_um > 0 moves right.
        y_um > 0 moves down.
        z_um = 0 is focal plane.

    Valid x/y range is approximately:
        -82.66 um to +82.66 um
    for 256 x 256 and pixel_size_um = 0.6483078.
    """
    points = np.asarray(point_targets, dtype=np.float32)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(
            "point_targets must have shape [N,4] with columns "
            "[x_um,y_um,z_um,intensity]"
        )

    if not np.all(np.isfinite(points)):
        raise ValueError("point_targets must contain only finite values")

    pixel_size_um = float(pixel_size_um)

    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")

    indexed = points.copy()

    # Model size is 256 x 256, valid index 0..255.
    center_index = 255.0 / 2.0   # 127.5

    # centered um -> voxel index
    indexed[:, 0] = np.rint(_X_INPUT_SIGN * indexed[:, 0] / pixel_size_um + center_index)
    indexed[:, 1] = np.rint(_Y_INPUT_SIGN * indexed[:, 1] / pixel_size_um + center_index)

    # z_um -> z index
    Z_WORLD_TO_DEEPCGH_SIGN = -1.0

    # z_um -> z index
    indexed[:, 2] = _z_um_to_indices(indexed[:, 2] * Z_WORLD_TO_DEEPCGH_SIGN)

    half_range_um = center_index * pixel_size_um

    if np.any(indexed[:, 0] < 0) or np.any(indexed[:, 0] > 255):
        raise ValueError(
            f"x_um is outside the valid centered range "
            f"{-half_range_um:.6f}..{half_range_um:.6f} um"
        )

    if np.any(indexed[:, 1] < 0) or np.any(indexed[:, 1] > 255):
        raise ValueError(
            f"y_um is outside the valid centered range "
            f"{-half_range_um:.6f}..{half_range_um:.6f} um"
        )

    if np.any(indexed[:, 2] < 0) or np.any(indexed[:, 2] > 20):
        raise ValueError("z_um is outside the valid range -20..20 um")

    return indexed

def _default_data_model():
    data = {
        "shape": (256, 256, 21),
        "object_size": 8,
        "object_type": "Gaussian",
        "centralized": False,
        "layer": 2,
        "dpm_threshold": 0.2,
    }
    model = {
        "path": str(_MODEL_DIR),
        "filename": "21layer_2000_2ch",
        "plane_distance": 0.000002,
        "wavelength": 1030e-9,
        "pixel_size": 0.0000256,
        "int_factor": 2,
        "n_kernels": [32, 64, 128],
        "input_name": "target",
        "output_name": "phi_slm",
        "lr": 1e-3,
        "focal_length": 0.004125,
        "batch_size": 5,
        "epochs": 5,
        "shuffle": 8,
        "max_steps": None,
        "refraction_index": 1.33,
    }

    runtime = globals().get("_RUNTIME_OPTICS", {})
    if runtime:
        model["wavelength"] = float(runtime.get("wavelength_m", model["wavelength"]))
        model["pixel_size"] = float(runtime.get("slm_pixel_pitch_m", model["pixel_size"]))
        model["focal_length"] = float(runtime.get("objective_focal_length_m", model["focal_length"]))
        model["refraction_index"] = float(runtime.get("refractive_index", model["refraction_index"]))

    return data, model


@lru_cache(maxsize=1)
def _get_model():
    optimize_torch_inference()
    data, model = _default_data_model()
    dcgh = DeepCGH(data, model)
    dcgh.prepare_inference(input_channels=2, use_amp=True, channels_last=True)
    return dcgh


def get_propagate(data, model):
    shape = data["shape"]
    plane_distance = model["plane_distance"]
    zs = np.linspace(0 - (shape[-1] // 2), shape[-1] // 2, shape[-1]) * plane_distance
    lambda_ = model["wavelength"]
    ps = model["pixel_size"]
    f = model["focal_length"]
    n = model["refraction_index"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __get_H(zs_, shape_, lambda_v, ps_v, f_v, n_v):
        hs = []
        k = np.pi * 2 / lambda_v * n_v
        x, y = np.meshgrid(
            np.linspace(-shape_[1] // 2 + 1, shape_[1] // 2, shape_[1]),
            np.linspace(-shape_[0] // 2 + 1, shape_[0] // 2, shape_[0]),
        )
        fx = x * ps_v / (lambda_v * f_v)
        fy = y * ps_v / (lambda_v * f_v)
        kz = np.sqrt(
            np.clip(k ** 2 - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2, 0, None)
        )
        kz = np.nan_to_num(kz)
        for z in zs_:
            hs.append(np.exp(1j * kz.real * z).astype(np.complex64))
        return hs

    def circle_mask(shape_):
        h, w, _ = shape_
        radius = 485
        cy = h // 2
        cx = w // 2
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
        return mask.astype(np.float32)

    def amplitude_SLM(shape_, centralwavelength, pixelsize_slm):
        focal_length_fourierlens = 750e-3
        beamsize = 4.24e-3 * 150 / 250
        c = 3e8
        centralfrequency = c / centralwavelength
        pulse_duration = 228e-15
        groove = 1 / 600 * 1e-3
        shape2 = (shape_[0] * 8, shape_[1] * 8)

        bandwidth = 0.441 / pulse_duration / c * np.power(centralwavelength, 2)
        sampling_frequency = 1.6e11
        spectral_bandwidth = c / np.power(centralwavelength, 2) * bandwidth
        incident_angle = np.arcsin(centralwavelength / groove)
        totalfrequency = np.arange(
            centralfrequency - spectral_bandwidth,
            centralfrequency + spectral_bandwidth,
            sampling_frequency,
        )
        if len(totalfrequency) > 32:
            idx = np.linspace(0, len(totalfrequency) - 1, 32).astype(int)
            totalfrequency = totalfrequency[idx]

        theta_m = np.zeros(len(totalfrequency), dtype=np.float64)
        for i in range(len(theta_m)):
            theta_m[i] = np.arcsin((c / totalfrequency[i]) / groove - np.sin(incident_angle))

        pixelsize_grating = (
            centralwavelength * focal_length_fourierlens / (shape2[0] * pixelsize_slm)
        )
        x = np.linspace((0 - shape2[0] // 2 + 1), shape2[0] - shape2[0] // 2, shape2[0])
        xx, yy = np.meshgrid(x, x)
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        mask_grating_intensity = np.exp(
            -2
            * (
                ((xx * pixelsize_grating) ** 2 + (yy * pixelsize_grating) ** 2)
                / (beamsize / 2) ** 2
            )
        )
        mask_grating = np.sqrt(mask_grating_intensity).astype(np.float32)

        mean = np.mean(totalfrequency)
        frequency_distribution = np.sqrt(np.pi / (2 * np.log(2))) * pulse_duration * np.exp(
            -1
            * np.power(pulse_duration, 2)
            * np.power((totalfrequency - mean) * 2 * np.pi, 2)
            / (8 * np.log(2))
        )
        frequency_distribution = frequency_distribution / max(frequency_distribution)

        result = np.zeros((shape_[0], shape_[1], len(theta_m)), dtype=np.complex64)
        for i in range(len(theta_m)):
            phasevariation = np.exp(
                1j
                * 2
                * np.pi
                / (c / totalfrequency[i])
                * np.sin(theta_m[i])
                * (xx * pixelsize_grating)
            )
            complex_amp = mask_grating * phasevariation
            complex_amp = np.fft.fftshift(np.fft.fft2(complex_amp) / shape2[1])
            result[:, :, i] = (
                complex_amp[
                    shape2[1] // 2 - shape_[1] // 2 : shape2[1] // 2 + shape_[1] // 2,
                    shape2[1] // 2 - shape_[1] // 2 : shape2[1] // 2 + shape_[1] // 2,
                ]
                * frequency_distribution[i]
            )

        return np.abs(result)

    def TF_reconstruction(temporal_focusing):
        n_freq = temporal_focusing.shape[-1]
        fft_num = 1024
        if n_freq < fft_num:
            pad = torch.zeros(
                temporal_focusing.shape[0],
                temporal_focusing.shape[1],
                temporal_focusing.shape[2],
                fft_num - n_freq,
                dtype=temporal_focusing.dtype,
                device=temporal_focusing.device,
            )
            result = torch.cat([temporal_focusing, pad], dim=-1)
        else:
            result = temporal_focusing
        fft_result = torch.fft.fftshift(torch.fft.fft(result, dim=-1) / 32.0, dim=-1)
        return torch.sum(torch.abs(fft_result) ** 4, dim=-1, keepdim=True).float()

    def __prop__(cf_slm, h):
        cf_slm = cf_slm * h.unsqueeze(0)
        complex_amp_focal_plane = torch.fft.fftshift(torch.fft.fft2(cf_slm), dim=(-2, -1))
        return complex_amp_focal_plane

    hs = [torch.from_numpy(x).to(device) for x in __get_H(zs, shape, lambda_, ps, f, n)]
    back_aperture = torch.from_numpy(circle_mask(shape)).to(device).to(torch.complex64)
    mask_amplitude = torch.from_numpy(amplitude_SLM(shape, lambda_, ps).astype(np.complex64)).to(device)

    def propagate(phi_slm):
        if isinstance(phi_slm, np.ndarray):
            phi_t = torch.from_numpy(phi_slm).float().to(device)
        else:
            phi_t = phi_slm.float().to(device)

        if phi_t.ndim == 3:
            phi_t = phi_t.unsqueeze(-1)
        if phi_t.ndim == 4 and phi_t.shape[-1] == 1:
            phi_t = phi_t.permute(0, 3, 1, 2)
        if phi_t.ndim != 4 or phi_t.shape[1] != 1:
            raise ValueError("phi_slm must be [N,H,W,1] or [N,1,H,W]")

        cf_slm = torch.exp(1j * phi_t.squeeze(1).to(torch.complex64))
        cf_slm = cf_slm * back_aperture.unsqueeze(0)

        frames = []
        for h in hs:
            single_layer = []
            for i in range(mask_amplitude.shape[2]):
                tfr = mask_amplitude[:, :, i].unsqueeze(0)
                buffer = cf_slm * tfr
                single_layer.append(__prop__(buffer, h).unsqueeze(-1))
            single_layer_multiwavelength = torch.cat(single_layer, dim=-1)
            temporal_focusing = TF_reconstruction(single_layer_multiwavelength)
            frames.append(temporal_focusing)

        y_pred = torch.cat(frames, dim=-1)
        y_pred = y_pred / (torch.amax(y_pred, dim=(1, 2, 3), keepdim=True) + 1e-8)

        gc.collect()
        return y_pred

    return propagate


@lru_cache(maxsize=1)
def _get_propagator():
    data, model = _default_data_model()
    return get_propagate(data, model)


def _infer_phase_from_points(
    point_targets,
    gaussian_diameter_um=10.0,
    pixel_size_um=_XY_PIXEL_SIZE_UM,
    return_target=False,
):
    dcgh = _get_model()
    indexed_points = points_um_to_indices(point_targets, pixel_size_um)
    result = dcgh.points_to_dpm(
        indexed_points,
        gaussian_diameter_um=gaussian_diameter_um,
        pixel_size_um=pixel_size_um,
        return_target=return_target,
    )
    if return_target:
        dpm, target = result
    else:
        dpm = result
        target = None

    phase = dcgh.get_hologram(dpm)[0, :, :, 0]
    phase = np.asarray(phase, dtype=np.float32)
    if return_target:
        return phase, target
    return phase


def SLM_phase(
    point_targets,
    output_path=None,
    gaussian_diameter_um=10.0,
    pixel_size_um=_XY_PIXEL_SIZE_UM,
    display=False,
):
    """Convert [x_um, y_um, z_um, intensity] points into one 2D SLM phase map.

    The model is loaded only on the first call. Set output_path=None to return
    the phase without writing a MAT file.
    """
    phase = _infer_phase_from_points(
        point_targets,
        gaussian_diameter_um=gaussian_diameter_um,
        pixel_size_um=pixel_size_um,
    )

    if output_path is not None:
        save_slm_phase(phase, output_path)

    if display:
        display_phase(phase)

    return phase


def display_phase(
    phase,
    title="SLM Phase",
    cmap="gray",
    colorbar=True,
    ax=None,
):
    """Display one UNet-predicted SLM phase map."""
    import matplotlib.pyplot as plt

    phase = np.asarray(phase)
    if phase.ndim == 4:
        if phase.shape[-1] == 1:
            phase = phase[0, :, :, 0]
        elif phase.shape[1] == 1:
            phase = phase[0, 0, :, :]
        else:
            raise ValueError("phase must contain a single channel")
    elif phase.ndim == 3:
        phase = np.squeeze(phase)
    if phase.ndim != 2:
        raise ValueError("phase must be [H,W], [H,W,1], [1,H,W,1], or [1,1,H,W]")

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(phase, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    if colorbar:
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.show()
    return ax


def show_phase(
    point_targets,
    output_path=None,
    gaussian_diameter_um=10.0,
    pixel_size_um=_XY_PIXEL_SIZE_UM,
    title="SLM Phase",
    cmap="gray",
    colorbar=True,
):
    """Run inference from micron coordinates and display the predicted phase."""
    phase = SLM_phase(
        point_targets,
        output_path=output_path,
        gaussian_diameter_um=gaussian_diameter_um,
        pixel_size_um=pixel_size_um,
        display=False,
    )
    display_phase(phase, title=title, cmap=cmap, colorbar=colorbar)
    return phase


def reconstruct_phase(phase):
    """Propagate one predicted SLM phase map into a 21-layer reconstruction."""
    phase = np.asarray(phase, dtype=np.float32)
    if phase.ndim == 2:
        phase = phase[None, :, :, None]
    elif phase.ndim == 3:
        phase = np.squeeze(phase)
        if phase.ndim == 2:
            phase = phase[None, :, :, None]
    reconstruction = _get_propagator()(phase)
    if isinstance(reconstruction, torch.Tensor):
        reconstruction = reconstruction.detach().cpu().numpy()
    reconstruction = np.asarray(reconstruction)
    if reconstruction.ndim == 4 and reconstruction.shape[0] == 1:
        reconstruction = reconstruction[0]
    return reconstruction


def display_reconstruction(
    reconstruction,
    target=None,
    phase=None,
    title="Reconstruction",
    cmap="gray",
    columns=7,
    z_layer_spacing_um=_Z_LAYER_SPACING_UM,
):
    """Display reconstructed intensity as z-layer images."""
    import matplotlib.pyplot as plt

    recon = np.asarray(reconstruction)
    if recon.ndim == 4:
        if recon.shape[0] != 1:
            raise ValueError("display_reconstruction expects one reconstruction batch")
        recon = recon[0]
    if recon.ndim != 3:
        raise ValueError("reconstruction must have shape [H,W,Z] or [1,H,W,Z]")

    layers = recon.shape[-1]
    target_vol = None
    if target is not None:
        target_vol = np.asarray(target)
        if target_vol.ndim == 4:
            if target_vol.shape[0] != 1:
                raise ValueError("display_reconstruction expects one target batch")
            target_vol = target_vol[0]
        if target_vol.ndim != 3:
            raise ValueError("target must have shape [H,W,Z] or [1,H,W,Z]")
        if target_vol.shape != recon.shape:
            raise ValueError(
                f"target shape {target_vol.shape} must match reconstruction shape {recon.shape}"
            )

    columns = int(columns)
    if columns <= 0:
        raise ValueError("columns must be a positive integer")
    z_zero = (layers - 1) / 2.0
    z_um = (np.arange(layers) - z_zero) * float(z_layer_spacing_um)

    if target_vol is not None:
        phase_img = None
        if phase is not None:
            phase_img = np.asarray(phase)
            if phase_img.ndim == 4:
                if phase_img.shape[-1] == 1:
                    phase_img = phase_img[0, :, :, 0]
                elif phase_img.shape[1] == 1:
                    phase_img = phase_img[0, 0, :, :]
            elif phase_img.ndim == 3:
                phase_img = np.squeeze(phase_img)
            if phase_img.ndim != 2:
                raise ValueError("phase must be [H,W], [H,W,1], [1,H,W,1], or [1,1,H,W]")

        plot_columns = layers + (1 if phase_img is not None else 0)
        fig, axes = plt.subplots(
            2,
            plot_columns,
            figsize=(2.2 * plot_columns, 4.8),
            sharex=True,
            sharey=True,
        )
        axes = np.asarray(axes).reshape(2, plot_columns)
        for layer in range(layers):
            axes[0, layer].imshow(target_vol[:, :, layer], cmap=cmap)
            axes[0, layer].set_title(f"Target @ {z_um[layer]:g} um", fontsize=8)
            axes[1, layer].imshow(recon[:, :, layer], cmap=cmap)
            axes[1, layer].set_title("Reconstruction", fontsize=8)
            axes[0, layer].axis("off")
            axes[1, layer].axis("off")

        if phase_img is not None:
            axes[0, -1].imshow(phase_img, cmap=cmap)
            axes[0, -1].set_title("SLM Phase", fontsize=8)
            axes[0, -1].axis("off")
            axes[1, -1].axis("off")

        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        plt.show()
        return fig

    rows = int(np.ceil(layers / columns))
    vmin = float(np.min(recon))
    vmax = float(np.max(recon))
    if np.isclose(vmin, vmax):
        vmin, vmax = None, None

    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    axes = np.asarray(axes).reshape(rows, columns)
    for layer in range(rows * columns):
        ax = axes.flat[layer]
        if layer < layers:
            ax.imshow(recon[:, :, layer], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Z {z_um[layer]:g} um", fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()
    return fig


def show_reconstruction(
    point_targets,
    output_path=None,
    gaussian_diameter_um=10.0,
    pixel_size_um=_XY_PIXEL_SIZE_UM,
    title="21-layer Reconstruction",
    cmap="gray",
    columns=7,
):
    """Run inference, propagate the predicted phase, and display all z layers."""
    phase, target = _infer_phase_from_points(
        point_targets,
        gaussian_diameter_um=gaussian_diameter_um,
        pixel_size_um=pixel_size_um,
        return_target=True,
    )
    if output_path is not None:
        save_slm_phase(phase, output_path)

    reconstruction = reconstruct_phase(phase)
    display_reconstruction(
        reconstruction,
        target=target,
        phase=phase,
        title=title,
        cmap=cmap,
        columns=columns,
        z_layer_spacing_um=_Z_LAYER_SPACING_UM,
    )
    return phase, reconstruction


def hologram(
    point_targets,
    display=False,
    show_reconstruction=False,
    save=None,
    output_path=None,
    grey_coeff_path=_DEFAULT_GREY_COEFF,
    gaussian_diameter_um=10.0,
    pixel_size_um=_XY_PIXEL_SIZE_UM,
    cmap="gray",
    columns=7,
    out=None,
):
    """Convert micron coordinates directly into an SLM-ready uint8 pattern.

    Realtime:
        pattern = hologram(points)

    Realtime with reused output buffer:
        buf = np.empty((1024, 1024), dtype=np.uint8)
        pattern = hologram(points, out=buf)

    Debug phase:
        pattern = hologram(points, display=True)

    Debug reconstruction:
        pattern = hologram(points, show_reconstruction=True)

    Save final SLM gray pattern:
        pattern = hologram(points, save="path.tif")

    Notes:
        - save/output_path saves the final SLM graylevel pattern, not phase.
        - output_path is kept as a backward-compatible alias for save.
        - returned value is always the final uint8 SLM pattern.
    """
    if output_path is not None:
        if save is not None and str(save) != str(output_path):
            raise ValueError("Use either save or output_path, not both with different paths")
        save = output_path

    if show_reconstruction:
        phase, target = _infer_phase_from_points(
            point_targets,
            gaussian_diameter_um=gaussian_diameter_um,
            pixel_size_um=pixel_size_um,
            return_target=True,
        )
    else:
        # Realtime path: avoid the SLM_phase() wrapper and call the shared
        # internal inference helper directly.
        phase = _infer_phase_from_points(
            point_targets,
            gaussian_diameter_um=gaussian_diameter_um,
            pixel_size_um=pixel_size_um,
            return_target=False,
        )
        target = None

    pattern = phase_to_greylevel(
        phase,
        save=save,
        grey_coeff_path=grey_coeff_path,
        out=out,
    )

    if show_reconstruction:
        reconstruction = reconstruct_phase(phase)
        display_reconstruction(
            reconstruction,
            target=target,
            phase=phase,
            title="21-layer Reconstruction",
            cmap=cmap,
            columns=columns,
            z_layer_spacing_um=_Z_LAYER_SPACING_UM,
        )
    elif display:
        display_phase(phase, title="SLM Phase", cmap=cmap, colorbar=True)

    return pattern



def make_pattern_buffer(shape=(1024, 1024)):
    """Create a reusable uint8 output buffer for hologram(..., out=buffer)."""
    return np.empty(tuple(shape), dtype=np.uint8)


def warmup_hologram(point_targets=None, n=20, out=None):
    """Warm up model inference and phase-to-greylevel conversion.

    Use once before realtime control:
        buf = make_pattern_buffer()
        warmup_hologram(points, n=20, out=buf)
    """
    if point_targets is None:
        point_targets = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    if out is None:
        out = make_pattern_buffer()

    result = None
    for _ in range(max(int(n), 1)):
        result = hologram(point_targets, out=out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result


def benchmark_hologram(point_targets=None, n=20, sync_cuda=True, reuse_out=True):
    """Benchmark hologram(points) without display/save/reconstruction."""
    if point_targets is None:
        point_targets = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    out = np.empty((1024, 1024), dtype=np.uint8) if reuse_out else None

    # warmup
    for _ in range(5):
        _ = hologram(point_targets, out=out)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(int(n)):
        t0 = time.perf_counter()
        _ = hologram(point_targets, out=out)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.asarray(times, dtype=np.float64)
    return {
        "n": int(n),
        "mean_ms": float(times.mean() * 1000),
        "median_ms": float(np.median(times) * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
        "fps_mean": float(1.0 / times.mean()),
        "reuse_out": bool(reuse_out),
        "sync_cuda": bool(sync_cuda),
    }


def profile_hologram(point_targets, n=20):
    """Profile index, DPM, UNet, and phase_to_greylevel separately."""
    dcgh = _get_model()
    out = np.empty((1024, 1024), dtype=np.uint8)

    for _ in range(3):
        _ = hologram(point_targets, out=out)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = {
        "coord_to_index_ms": [],
        "points_to_dpm_ms": [],
        "unet_gpu_cpu_ms": [],
        "phase_to_greylevel_ms": [],
        "total_ms": [],
    }

    for _ in range(int(n)):
        t0 = time.perf_counter()

        indexed_points = points_um_to_indices(point_targets)
        t1 = time.perf_counter()

        dpm = dcgh.points_to_dpm(
            indexed_points,
            gaussian_diameter_um=10.0,
            pixel_size_um=_XY_PIXEL_SIZE_UM,
        )
        t2 = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        phase = dcgh.get_hologram(dpm)[0, :, :, 0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        _ = phase_to_greylevel(phase, out=out)
        t4 = time.perf_counter()

        times["coord_to_index_ms"].append((t1 - t0) * 1000)
        times["points_to_dpm_ms"].append((t2 - t1) * 1000)
        times["unet_gpu_cpu_ms"].append((t3 - t2) * 1000)
        times["phase_to_greylevel_ms"].append((t4 - t3) * 1000)
        times["total_ms"].append((t4 - t0) * 1000)

    return {
        k: {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
        for k, v in times.items()
    }
