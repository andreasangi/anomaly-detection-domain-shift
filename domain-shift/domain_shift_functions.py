import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

CLASS_FOLDERS = ["bent", "color", "flip", "good", "scratch"]
VALID_EXTS = {".png", ".jpg", ".jpeg"}

# Relative to this script's location (projectRoot/domain-shift/)
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data" / "metal_nut" / "test"
OUTPUT_ROOT = SCRIPT_DIR / "augmented_test"


def _clip(img: np.ndarray) -> np.ndarray:
    """Clip float image to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)

def _scaled_range(lo: float, hi: float, scale: float) -> tuple[float, float]:
    """
    Shrink or expand a [lo, hi] range around its midpoint.

    - scale = 1.0 returns the range unchanged
    - scale < 1.0 shrinks it (ex. 0.6 = 60% of the original width)
    - scale > 1.0 expands it
    """
    mid  = (lo + hi) / 2
    half = (hi - lo) / 2 * scale
    return mid - half, mid + half

def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img

def _save(img: np.ndarray, path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    sidecar = path.with_name(path.stem + "_params.json")
    sidecar.write_text(json.dumps(params, indent=2))

def apply_exposure(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Simulate exposure variation via linear scaling.
    NEW_PIXEL = alpha * OLD_PIXEL + beta

    alpha < 1  -> under-exposure (e.g. insufficient light, fast shutter)
    alpha > 1  -> over-exposure (e.g. blown highlights, long exposure)
    beta       -> additive offset (dark current / ambient offset)

    Industrial case: line lighting intensity varies with voltage
    fluctuations, lamp aging, and controller settings.
    Plausible range: alpha ∈ [0.5, 1.7], beta ∈ [-30, 30]
    """
    alpha = rng.uniform(*_scaled_range(0.5, 1.7, scale))
    beta  = rng.uniform(*_scaled_range(-30, 30, scale))
    out   = _clip(img.astype(np.float32) * alpha + beta)
    return out, {"alpha": round(alpha, 3), "beta": round(beta, 3)}

def apply_gamma(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Gamma correction to simulate different sensor response curves.

    gamma < 1  -> image brightened (as if sensor is more sensitive)      -- lifts shadows/midtones
    gamma > 1  -> image darkened         -- compresses shadows/midtones

    Industrial case: different camera models (or firmware versions)
    apply different gamma tables in-sensor. 
    Mild miscalibration between camera units (around 1 gamma) or 
    gamma correction accidentally enabled/disabled (gamma  0.45 or 2.2).
    Plausible range for fluctations: gamma ∈ [0.45, 2.2]
    """
    gamma     = rng.uniform(*_scaled_range(0.45, 2.2, scale))
    lut       = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    out       = cv2.LUT(img, lut)
    return out, {"gamma": round(gamma, 3)}

def apply_white_balance(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Per-channel multiplicative drift to simulate white-balance miscalibration.

    R and B are anti-correlated to model colour temperature shift along the
    warm/cool axis (warm = more R, less B; cool = more B, less R).
    G stays nearly stable as cameras are designed around the green channel.

    Industrial case: switching between fluorescent, LED, halogen and sodium-
    vapour lighting changes the illuminant spectrum; a fixed white-balance
    preset introduces a colour cast.
    Plausible range: per-channel scale ∈ [0.75, 1.25]
    """
    warm = rng.random() > 0.5          # True -> warm cast, False -> cool cast

    scale_G = rng.uniform(*_scaled_range(0.90, 1.10, scale))
    scale_R = rng.uniform(*_scaled_range(1.10, 1.40, scale)) if warm else rng.uniform(*_scaled_range(0.70, 0.90, scale))
    scale_B = rng.uniform(*_scaled_range(0.70, 0.90, scale)) if warm else rng.uniform(*_scaled_range(1.10, 1.40, scale))

    out = img.astype(np.float32).copy()
    for c, s in enumerate([scale_B, scale_G, scale_R]):
        out[:, :, c] *= s
    out = _clip(out)
    return out, {"scale_B": round(scale_B, 3),
                 "scale_G": round(scale_G, 3),
                 "scale_R": round(scale_R, 3),
                 "cast":    "warm" if warm else "cool"}

def apply_noise(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Gaussian read-noise + optional salt-and-pepper dead/hot pixels.

    Gaussian sigma models thermal (read) noise (low-light or high-gain).
    Salt-and-pepper fraction models sensor defects, hot/dead pixels.

    Industrial case: industrial cameras operating at high gain
    (low-light) show significant read noise, and older sensors develop dead pixels.
    Plausible range: sigma ∈ [5, 40], sp_fraction ∈ [0, 0.005]
    """
    sigma      = rng.uniform(*_scaled_range(5, 40, scale))
    sp_frac    = rng.uniform(*_scaled_range(0.0, 0.005, scale))

    # Gaussian
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out   = _clip(img.astype(np.float32) + noise)

    # Salt-and-pepper
    n_pixels = int(sp_frac * img.shape[0] * img.shape[1])
    if n_pixels > 0:
        # Salt (white)
        ys = np.random.randint(0, img.shape[0], n_pixels // 2)
        xs = np.random.randint(0, img.shape[1], n_pixels // 2)
        out[ys, xs] = 255
        # Pepper (black)
        ys = np.random.randint(0, img.shape[0], n_pixels // 2)
        xs = np.random.randint(0, img.shape[1], n_pixels // 2)
        out[ys, xs] = 0

    return out, {"gaussian_sigma": round(sigma, 2),
                 "sp_fraction":    round(sp_frac, 5)}

def apply_jpeg(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Simulate JPEG compression artifacts by encoding and re-decoding.

    Lower quality -> stronger block artifacts (8×8 DCT blocks visible).

    Industrial case: images transmitted over network links (GigE
    Vision with software compression, or IP cameras) are often JPEG-encoded.
    Plausible range: quality ∈ [20, 60]
    """
    q_lo, q_hi = _scaled_range(20, 60, scale)
    quality = rng.randint(round(q_lo), round(q_hi))
    _, buf  = cv2.imencode('.jpg', img,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    out     = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out, {"jpeg_quality": quality}

def apply_blur(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Motion blur (directional) OR defocus blur (isotropic), chosen randomly.

    Motion blur:   linear kernel at a random angle -
                   models conveyor, camera vibration or object not perfectly still during exposure.
    Defocus blur:  Gaussian kernel
                   Object not on focal plane,
                   models depth-of-field variation when objects have different 
                   heights on the inspection tapis.

    Plausible kernel sizes: motion length ∈ [5, 25 px], angle ∈ [0°, 180°]
                            defocus sigma ∈ [1.5, 6.0]
    """
    kind = rng.choice(["motion", "defocus"])
    params: dict = {"kind": kind}

    if kind == "motion":
        len_lo, len_hi = _scaled_range(5, 25, scale)
        length = rng.randint(round(len_lo), round(len_hi))
        angle  = rng.uniform(0, 180)   # orientation, left unscaled
        params.update({"length": length, "angle_deg": round(angle, 1)})

        # Build a line at the desired angle to create the kernel
        kernel = np.zeros((length, length), dtype=np.float32)
        cx, cy = length // 2, length // 2
        angle_rad = np.deg2rad(angle)
        x1 = int(cx - (length // 2) * np.cos(angle_rad))
        y1 = int(cy - (length // 2) * np.sin(angle_rad))
        x2 = int(cx + (length // 2) * np.cos(angle_rad))
        y2 = int(cy + (length // 2) * np.sin(angle_rad))
        cv2.line(kernel, (x1, y1), (x2, y2), 1.0, 1)

        kernel  = kernel / kernel.sum()             # re-normalise after rotation
            # warpAffine uses bilinear interpolation when rotating, which distributes energy 
            # across neighbouring pixels and can reduce the kernel sum below 1, causing 
            # the blurred image to darken slightly

        out     = cv2.filter2D(img, -1, kernel)

    else:  # defocus
        sigma = rng.uniform(*_scaled_range(1.5, 6.0, scale))
        ksize = int(sigma * 6) | 1                 # must be odd
        params.update({"sigma": round(sigma, 2), "ksize": ksize})
        out   = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    return out, params

def apply_vignette(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Apply a smooth radial brightness falloff (vignetting) that darkens toward the image corners.

    Implemented as a 2D Gaussian mask centred at (cx, cy) with spread sigma.
    Strength parameter controls how dark the corners get.

    Industrial case: all lenses exhibit natural vignetting;
    many real camera+lens systems exhibit relative-illumination roll-off toward the
    periphery due to optical and mechanical vignetting; severity depends on 
    lens design, aperture, and sensor/lens matching.
    Plausible range: strength ∈ [0.3, 0.8], sigma_frac ∈ [0.5, 0.9]
    """
    h, w   = img.shape[:2]
    strength    = rng.uniform(*_scaled_range(0.3, 0.8, scale))   # how dark corners get (0 = no effect)
    sigma_frac  = rng.uniform(*_scaled_range(0.5, 0.9, scale))   # Gaussian spread as fraction of image size

    sigma_x = w * sigma_frac
    sigma_y = h * sigma_frac
    cx, cy  = w / 2, h / 2

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # 2D Gaussian mask with centre=1 and corners approaching (1 - strength)
    mask = np.exp(-((X - cx) ** 2 / (2 * sigma_x ** 2) +
                    (Y - cy) ** 2 / (2 * sigma_y ** 2)))

    # Scale mask so centre = 1 and corners = (1 - strength)
    mask = 1.0 - strength * (1.0 - mask)
    mask = mask[:, :, np.newaxis]            # broadcast over channels

    out = _clip(img.astype(np.float32) * mask)
    return out, {"strength":   round(strength, 3),
                 "sigma_frac": round(sigma_frac, 3)}


def apply_shadow(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Directional gradient shadow: a linear brightness ramp across the image.

    Models uneven scene illumination from a single off-axis light source. 

    Industrial case: lamp repositioning or replacement, single-sided ring light failure,
                    conveyor edge shadow, factory window light...

    direction: 'horizontal' | 'vertical' | 'diagonal'
    dark_side:  which edge is darkened (0 = left/top, 1 = right/bottom)
    intensity:  fraction of brightness lost at the dark edge [0.2, 0.6]
    """
    direction  = rng.choice(["horizontal", "vertical", "diagonal"])
    dark_side  = rng.randint(0, 1)
    intensity  = rng.uniform(*_scaled_range(0.2, 0.6, scale))

    h, w = img.shape[:2]

    if direction == "horizontal":
        ramp = np.linspace(0, 1, w, dtype=np.float32)
        mask = np.tile(ramp, (h, 1))
    elif direction == "vertical":
        ramp = np.linspace(0, 1, h, dtype=np.float32)
        mask = np.tile(ramp[:, np.newaxis], (1, w))
    else:  
        ramp_x = np.linspace(0, 1, w, dtype=np.float32)
        ramp_y = np.linspace(0, 1, h, dtype=np.float32)
        mask   = np.outer(ramp_y, ramp_x)

    if dark_side == 0:
        mask = 1.0 - mask

    # bright side = no change, dark side = multiply by (1 - intensity)
    scale = 1.0 - intensity * (1.0 - mask)
    scale = scale[:, :, np.newaxis]

    out = _clip(img.astype(np.float32) * scale)
    return out, {"direction": direction, "dark_side": dark_side,
                 "intensity": round(intensity, 3)}


def apply_contrast(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Local contrast variation via CLAHE (Contrast Limited Adaptive Histogram Equalization), 
    simulating spatially non-uniform contrast response differences between 
    industrial camera units.

    Low clip_limit = subtle local contrast enhancement.
    High clip_limit = aggressive local boosting, creates blocky appearance.

    Applied in LAB colour space (L channel only) to avoid hue shifts.

    Industrial case: different AGC/AES histogram modes between cameras (mean vs peak-white),
        Flat-field correction drift, Firmware-level tone curve differences between camera models or firmware versions
    Plausible range: clip_limit [2, 6.0], tile_grid [8, 16, 32]
    """
    clip_limit = rng.uniform(*_scaled_range(2.0, 6.0, scale))
    tile_size  = rng.choice([8, 16, 32])

    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    L_eq  = clahe.apply(L)

    lab_eq = cv2.merge([L_eq, a, b])
    out    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return out, {"clip_limit": round(clip_limit, 2), "tile_size": tile_size}


def apply_affine(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Small affine transformation: translation + rotation + mild shear.

    Industrial case: imperfect part placement on the inspection fixture, a
    camera that shifted slightly between calibration and deployment or got changed/remouted.

    Plausible ranges (industrial fixture tolerances):
        translation: ±40 px 
        rotation:    ±25°
        shear:       ±0.08 (±4.6° camera tilt)
    """
    h, w = img.shape[:2]
    tx    = rng.uniform(*_scaled_range(-40, 40, scale))     # translation x
    ty    = rng.uniform(*_scaled_range(-40, 40, scale))     # translation y
    angle = rng.uniform(*_scaled_range(-25, 25, scale))
    shear = rng.uniform(*_scaled_range(-0.08, 0.08, scale))

    # Build affine matrix (rotation + shear) around center, then translate
    cx, cy = w / 2, h / 2

    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    a = cos_a - shear * sin_a
    b = -sin_a + shear * cos_a
    c = sin_a
    d = cos_a

    rot_mat = np.array([
        [a, b, (1 - a) * cx - b * cy + tx],
        [c, d, (1 - d) * cy - c * cx + ty],
    ], dtype=np.float32)

    out = cv2.warpAffine(img, rot_mat, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) # reflect border to avoid black edges
    return out, {"tx": round(tx, 2), "ty": round(ty, 2),
                 "angle_deg": round(angle, 2), "shear": round(shear, 4)}


def apply_perspective(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Perspective warp derived from physical camera tilt angles.

    Builds the homography from explicit pitch and roll angles, this guarantees that circles
    remain ellipses, straight lines remain straight, and the warp corresponds
    to a real camera position.

    Physical model: pinhole camera, planar object, orthographic approximation
    for small angles. Focal length f = image width, corresponding to ca.53° horizontal
    FOV which is slightly wider than a typical macro lens but chosen to produce clearly
    visible perspective distortion per degree of tilt, appropriate for a robustness test.

    Industrial case: camera remounted after maintenance with slight tilt,
    fixture not perfectly level, imperfect mounting, Scheimpflug tilt for depth of field control.

    Plausible range:
        pitch (x-tilt): ±15°   camera nodding forward/backward
        roll  (y-tilt): ±15°   camera tilting left/right
        Both angles independently sampled.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    f = w   # ~53° horizontal FOV

    pitch_deg = rng.uniform(*_scaled_range(-15, 15, scale))
    roll_deg  = rng.uniform(*_scaled_range(-15, 15, scale))
    pitch = np.deg2rad(pitch_deg)
    roll  = np.deg2rad(roll_deg)

    Rx = np.array([[1,           0,            0],
                   [0,  np.cos(pitch), -np.sin(pitch)],
                   [0,  np.sin(pitch),  np.cos(pitch)]], dtype=np.float64)

    Ry = np.array([[ np.cos(roll), 0, np.sin(roll)],
                   [0,             1,           0  ],
                   [-np.sin(roll), 0, np.cos(roll)]], dtype=np.float64)

    K = np.array([[f,  0, cx],
                  [0,  f, cy],
                  [0,  0,  1]], dtype=np.float64)

    H = K @ (Ry @ Rx) @ np.linalg.inv(K)
    H /= H[2, 2]

    # Recentre: find where image centre maps to and translate it back
    centre_mapped = H @ np.array([cx, cy, 1.0])
    centre_mapped /= centre_mapped[2]
    T = np.array([[1, 0, cx - centre_mapped[0]],
                  [0, 1, cy - centre_mapped[1]],
                  [0, 0, 1                    ]], dtype=np.float64)
    H = T @ H
    H /= H[2, 2]

    # Conditional scaling: measure actual overflow on each side independently,
    # only scale if something genuinely clips outside the frame
    corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=np.float64)
    mapped  = (H @ corners.T).T
    mapped  = mapped[:, :2] / mapped[:, 2:3]

    overflow = max(
        max(-mapped[:,0].min(), 0) / w,      # left side
        max(mapped[:,0].max() - w, 0) / w,   # right side
        max(-mapped[:,1].min(), 0) / h,      # top
        max(mapped[:,1].max() - h, 0) / h    # bottom
    )

    if overflow > 0.01:   # threshold: only act if overflow exceeds 1% of image
        scale = 1.0 - overflow * 1.05        # pull in by just enough + 5% margin
        S = np.array([[scale, 0,     cx * (1 - scale)],
                      [0,     scale, cy * (1 - scale)],
                      [0,     0,     1               ]], dtype=np.float64)
        H = S @ H
        H /= H[2, 2]
    else:
        scale = 1.0

    out = cv2.warpPerspective(img, H, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    return out, {
        "pitch_deg":     round(pitch_deg, 2),
        "roll_deg":      round(roll_deg,  2),
        "scale_applied": round(scale, 4),
    }


def apply_specular(img: np.ndarray, rng: random.Random, scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """
    Simulates a specular / glare hotspot on reflective surfaces.

    A bright, soft ellipse is blended additively onto the image, with
    intensity falling off as a 2D Gaussian.

    Industrial case: metallic and polished surfaces (many MVTec
    categories: metal_nut, screw, transistor) produce strong specular
    reflections when illumination angle changes slightly.

    Plausible range: hotspot covers 5–25% of image area, brightness ∈ [80, 220]
    """
    h, w   = img.shape[:2]
    cx     = rng.uniform(0.2, 0.8) * w          # center coords (inner 60% of image, position: left unscaled)
    cy     = rng.uniform(0.2, 0.8) * h
    rx     = rng.uniform(*_scaled_range(0.05, 0.25, scale)) * w   # radius in x/y (ellipse axes)
    ry     = rng.uniform(*_scaled_range(0.05, 0.25, scale)) * h
    bright = rng.uniform(*_scaled_range(80, 220, scale))          # peak intensity, added at the center

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # Gaussian hotspot (sigma = radius / 2)
    hotspot = bright * np.exp(-(((X - cx) / (rx / 2)) ** 2 +
                                ((Y - cy) / (ry / 2)) ** 2) / 2)
    hotspot = hotspot[:, :, np.newaxis]

    # Reduce hotspot where surface is already bright, we use exponential control not linear
    headroom = ((255 - img.astype(np.float32)) / 255.0) ** 0.3   # 0 where saturated, 1 where dark

    out = _clip(img.astype(np.float32) + hotspot * headroom)
    return out, {"cx": round(cx, 1), "cy": round(cy, 1),
                 "rx": round(rx, 1), "ry": round(ry, 1),
                 "brightness": round(bright, 1)}


TRANSFORMS = {
    "exposure":    apply_exposure,
    "gamma":       apply_gamma,
    "wb":          apply_white_balance,
    "noise":       apply_noise,
    "jpeg":        apply_jpeg,
    "blur":        apply_blur,
    "vignette":    apply_vignette,
    "shadow":      apply_shadow,
    "contrast":    apply_contrast,
    "affine":      apply_affine,
    "perspective": apply_perspective,
    "specular":    apply_specular,
}

def sample_images(class_dir: Path, n, rng: random.Random) -> list[Path]:
    """
    Return a list of image paths from class_dir.
    n = 'all' : every image in the directory.
    n = int   : random sample of n (available if n > available).
    """
    all_imgs = sorted([p for p in class_dir.iterdir()
                       if p.suffix.lower() in VALID_EXTS])
    if not all_imgs:
        return []
    if n == "all":
        return all_imgs
    n = int(n)
    if n >= len(all_imgs):
        print(f"    [!] Requested {n} but only {len(all_imgs)} available "
              f"in {class_dir.name} - using all.")
        return all_imgs
    return rng.sample(all_imgs, n)