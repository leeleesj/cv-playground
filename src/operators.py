from typing import Any, Dict, Tuple, Optional

import numpy as np
import cv2


# ----------------------------
# Common utilities
# ----------------------------
def _ensure_rgb_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """
    Gradio Image(type="numpy") 입력을 가정: HWC, uint8, RGB.
    예외 케이스(그레이/float)도 방어적으로 처리.
    """
    if image_rgb is None:
        return None

    img = image_rgb

    # float -> uint8 (0~1 or 0~255 가정)
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    # Gray (H,W) -> RGB
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # (H,W,4) RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    return img


def _rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def _to_uint8_vis(x: np.ndarray) -> np.ndarray:
    """
    임의 범위 float/정수 배열을 0~255 uint8로 정규화(시각화 용).
    """
    x = x.astype(np.float32)
    x -= float(x.min())
    denom = float(x.max() - x.min()) + 1e-8
    x = x / denom
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def _gray_to_rgb(gray_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


def _to_uint8_vis(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """float array -> 0..255 uint8 for visualization."""
    x = x.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.uint8)
    x_norm = (x - x_min) / (x_max - x_min)
    return (x_norm * 255.0).astype(np.uint8)


# ----------------------------
# Laplacian
# ----------------------------
def laplacian(
    image_rgb: np.ndarray,
    ksize: int = 3,
    use_abs: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Laplacian 시각화 + blur 지표로 자주 쓰는 Laplacian variance 계산.

    Returns:
        (output_rgb_uint8, meta)
    """
    if image_rgb is None:
        return None, {"status": "no_image"}

    img = _ensure_rgb_uint8(image_rgb)
    gray = _rgb_to_gray(img)

    # CV_32F로 계산 (정밀도)
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=ksize)

    # blur score로 자주 쓰는 variance (원본 lap 기반)
    lap_var = float(lap.var())

    if use_abs:
        lap = np.abs(lap)

    vis = _gray_to_rgb(_to_uint8_vis(lap))

    meta = {
        "status": "ok",
        "op": "laplacian",
        "ksize": int(ksize),
        "laplacian_var": lap_var,
        "input_shape_hwc": list(img.shape),
        "dtype": str(img.dtype),
    }
    return vis, meta


# ----------------------------
# Sobel
# ----------------------------
def sobel(
    image_rgb: np.ndarray,
    ksize: int = 3,
    mode: str = "magnitude",  # "magnitude" | "x" | "y"
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Sobel 에지 시각화.
    - mode="x": x-방향 기울기
    - mode="y": y-방향 기울기
    - mode="magnitude": sqrt(gx^2 + gy^2)

    Returns:
        (output_rgb_uint8, meta)
    """
    if image_rgb is None:
        return None, {"status": "no_image"}

    img = _ensure_rgb_uint8(image_rgb)
    gray = _rgb_to_gray(img).astype(np.float32)

    gx = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gy = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)

    if mode == "x":
        arr = np.abs(gx)
    elif mode == "y":
        arr = np.abs(gy)
    else:
        # magnitude
        arr = cv2.magnitude(gx, gy)

    # 간단 지표(에지 강도 요약)
    mean_abs_gx = float(np.mean(np.abs(gx)))
    mean_abs_gy = float(np.mean(np.abs(gy)))
    mean_mag = float(np.mean(arr))

    vis = _gray_to_rgb(_to_uint8_vis(arr))

    meta = {
        "status": "ok",
        "op": "sobel",
        "ksize": int(ksize),
        "mode": mode,
        "mean_abs_gx": mean_abs_gx,
        "mean_abs_gy": mean_abs_gy,
        "mean_edge_strength": mean_mag,
        "input_shape_hwc": list(img.shape),
        "dtype": str(img.dtype),
    }
    return vis, meta


# ----------------------------
# FFT
# ----------------------------
def fft_spectrum(
    image_rgb: np.ndarray,
    log_scale: bool = True,
    center: bool = True,
    hf_radius_ratio: float = 0.25,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """
    FFT magnitude spectrum visualization (log magnitude).
    Returns:
      - spectrum_rgb: RGB uint8 image for display
      - meta: dict with hf_ratio, etc.
    """
    if image_rgb is None:
        return None, {"status": "no_image"}

    img = _ensure_rgb_uint8(image_rgb)
    gray = _rgb_to_gray(img).astype(np.float32) / 255.0
    h, w = gray.shape[:2]

    # FFT (complex)
    F = np.fft.fft2(gray)
    if center:
        F = np.fft.fftshift(F)

    mag = np.abs(F)

    # Log magnitude for visualization
    spec = np.log1p(mag) if log_scale else mag
    spec_vis = _to_uint8_vis(spec)
    spectrum_rgb = _gray_to_rgb(spec_vis)

    # High-frequency energy ratio (very simple metric)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    r0 = hf_radius_ratio * min(h, w)
    hf_energy = float(mag[rr >= r0].sum())
    total_energy = float(mag.sum()) + eps
    hf_ratio = hf_energy / total_energy

    meta = {
        "status": "ok",
        "op": "fft_spectrum",
        "shape_hw": [int(h), int(w)],
        "log_scale": bool(log_scale),
        "centered": bool(center),
        "hf_radius_ratio": float(hf_radius_ratio),
        "hf_ratio": float(hf_ratio),
    }
    return spectrum_rgb, meta
