import torch


def lf_2d_to_views(
    lf: torch.Tensor,
    n_lens_y: int = 21,
    n_lens_x: int = 21,
    n_pix_y: int = 11,
    n_pix_x: int = 11,
) -> torch.Tensor:
    """
    Convert a 2D light-field image into sub-aperture views.

    Input
    -----
    lf: [B, 1, H, W]
        H = n_lens_y * n_pix_y
        W = n_lens_x * n_pix_x

    Output
    ------
    views: [B, V, n_lens_y, n_lens_x]
        V = n_pix_y * n_pix_x

    Convention
    ----------
    Raw LF layout is interpreted as:
        [lenslet_y, pixel_y, lenslet_x, pixel_x]

    View channel index:
        view_id = pixel_y * n_pix_x + pixel_x
    """
    if lf.ndim != 4:
        raise ValueError(f"Expected lf shape [B, 1, H, W], got {tuple(lf.shape)}")

    b, c, h, w = lf.shape

    if c != 1:
        raise ValueError(f"Expected one channel, got C={c}")

    expected_h = n_lens_y * n_pix_y
    expected_w = n_lens_x * n_pix_x

    if h != expected_h or w != expected_w:
        raise ValueError(
            f"Expected H,W=({expected_h},{expected_w}), got ({h},{w})"
        )

    x = lf.reshape(b, 1, n_lens_y, n_pix_y, n_lens_x, n_pix_x)
    x = x.permute(0, 3, 5, 2, 4, 1)
    views = x.reshape(b, n_pix_y * n_pix_x, n_lens_y, n_lens_x)

    return views


def views_to_lf_2d(
    views: torch.Tensor,
    n_lens_y: int = 21,
    n_lens_x: int = 21,
    n_pix_y: int = 11,
    n_pix_x: int = 11,
) -> torch.Tensor:
    """
    Convert sub-aperture views back to a 2D light-field image.

    Input
    -----
    views: [B, V, n_lens_y, n_lens_x]

    Output
    ------
    lf: [B, 1, H, W]
    """
    if views.ndim != 4:
        raise ValueError(
            f"Expected views shape [B, V, n_lens_y, n_lens_x], got {tuple(views.shape)}"
        )

    b, v, ly, lx = views.shape
    expected_v = n_pix_y * n_pix_x

    if v != expected_v:
        raise ValueError(f"Expected V={expected_v}, got V={v}")

    if ly != n_lens_y or lx != n_lens_x:
        raise ValueError(
            f"Expected lenslet grid ({n_lens_y},{n_lens_x}), got ({ly},{lx})"
        )

    x = views.reshape(b, n_pix_y, n_pix_x, n_lens_y, n_lens_x, 1)
    x = x.permute(0, 5, 3, 1, 4, 2)
    lf = x.reshape(b, 1, n_lens_y * n_pix_y, n_lens_x * n_pix_x)

    return lf