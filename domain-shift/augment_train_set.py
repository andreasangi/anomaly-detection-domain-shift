"""
Builds augmented copies of the normal (good) training images, using the
same transform functions used to build the domain-shift test sets.

The input is a single flat folder of good images (train/good)

For each input image, one random enabled transform is selected by default.
Set AUGMENTATIONS_PER_IMAGE > 1 to generate more augmented copies per image.

Edit the CONFIGURATION section below to control:
  - which transforms to run
  - how many images to sample (None = all)
  - where input and output live
  - the random seed

Each run produces one directory under OUTPUT_ROOT:
    domain-shift/
    '-- metal_nut_augmented/
        '-- train_good/
            |-- good_exposure/
            │   ├-- 000.png                 <- same filename as source
            │   ├-- 001.png
            │   └-- augmentation_log.json    <- one file documenting all params
            |-- good_contrast/
            '-- ...
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

# Import all transform functions from the file
from domain_shift_functions import (
    apply_blur,
    apply_contrast,
    apply_exposure,
    apply_gamma,
    apply_jpeg,
    apply_noise,
    apply_perspective,
    apply_affine,
    apply_shadow,
    apply_specular,
    apply_vignette,
    apply_white_balance,
    VALID_EXTS,
)

# ------------------------------------------------------------------------------
# CONFIGURATION - edit this section
# ------------------------------------------------------------------------------

# Root of the project (parent of domain-shift/)
ROOT = Path(__file__).parent.parent

# Input: folder of normal images to augment.
# Point this at train/good to build the memory bank pool, or at the
# validation split's good folder to build the threshold pool. Run the
# script once per folder, changing INPUT_DIR and OUTPUT_ROOT below.
INPUT_DIR = ROOT / "data" / "metal_nut" / "train" / "good"

# Output: all augmented sets go here, original data is never touched
OUTPUT_DIR = Path(__file__).parent / "metal_nut_augmented" / "train" / "good"

# Number of augmented copies to generate for each source image.
# 1 = one random transform per image, 2 = two random transforms per image, ecc...
AUGMENTATIONS_PER_IMAGE = 1

# Number of images to sample. Set to None to augment all available images,
# which is the usual choice here since every normal image should get an
# augmented counterpart in the memory bank / threshold pool.
N_IMAGES = None

# Random seed: controls which images are sampled, not the augmentation params
# (augmentation params have their own seed = SEED + 1)
SEED = 42


# Transforms available for random selection.
# Kept aligned with the five transforms used in the domain-shift evaluation.

TRANSFORMS = {
    "exposure":    [apply_exposure],
    "wb":          [apply_white_balance],
    "noise":       [apply_noise],
    "contrast":    [apply_contrast],
    "perspective": [apply_perspective],

    # Not evaluated in the domain-shift table, kept here for completeness
    # if the pool needs to be extended later.
    #"gamma":       [apply_gamma],
    #"jpeg":        [apply_jpeg],
    #"blur":        [apply_blur],
    #"vignette":    [apply_vignette],
    #"shadow":      [apply_shadow],
    #"affine":      [apply_affine],
    #"specular":    [apply_specular],
}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


def _save_img(img: np.ndarray, path: Path) -> None:
    """Save image only - params are collected into the run-level JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _sample(image_dir: Path, n, rng: random.Random) -> list[Path]:
    """Return n randomly sampled image paths from image_dir."""
    imgs = sorted(p for p in image_dir.iterdir()
                  if p.suffix.lower() in VALID_EXTS)
    if not imgs:
        return []
    if n is None or n >= len(imgs):
        return imgs
    return rng.sample(imgs, n)


def _apply_chain(img: np.ndarray,
                 fns: list,
                 aug_rng: random.Random) -> tuple[np.ndarray, dict]:
    """Apply a list of transform functions in sequence, collecting all params."""
    out        = img.copy()
    fn_names   = [fn.__name__.replace("apply_", "") for fn in fns]
    all_params = {"transforms_applied": fn_names}

    for fn in fns:
        out, p = fn(out, aug_rng)
        name   = fn.__name__.replace("apply_", "")
        all_params[name] = p

    return out, all_params



# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            f"Check the ROOT and INPUT_DIR paths in the CONFIGURATION section."
        )

    if AUGMENTATIONS_PER_IMAGE < 1:
        raise ValueError("AUGMENTATIONS_PER_IMAGE must be >= 1")
    if not TRANSFORMS:
        raise ValueError("TRANSFORMS must contain at least one enabled transform")

    # Separate seeded RNGs keep image sampling, transform selection and
    # augmentation parameters independent and reproducible.
    sample_rng = random.Random(SEED)
    choice_rng = random.Random(SEED + 1)
    aug_rng    = random.Random(SEED + 2)
    np.random.seed(SEED)

    print(f"\nInput  : {INPUT_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Seed   : {SEED}   n_images: {N_IMAGES or 'all'}")
    print(f"Augmentations per image: {AUGMENTATIONS_PER_IMAGE}\n")

    images = _sample(INPUT_DIR, N_IMAGES, sample_rng)
    if not images:
        raise FileNotFoundError(f"No valid images found in: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transform_names = list(TRANSFORMS.keys())
    run_log = {}

    for src in images:
        img = _load(src)

        # Avoid repeating the same transform for one source image until all
        # available transforms have been used at least once.
        selected = []
        while len(selected) < AUGMENTATIONS_PER_IMAGE:
            cycle = transform_names.copy()
            choice_rng.shuffle(cycle)
            selected.extend(cycle)
        selected = selected[:AUGMENTATIONS_PER_IMAGE]

        for aug_idx, transform_name in enumerate(selected, start=1):
            aug, params = _apply_chain(img, TRANSFORMS[transform_name], aug_rng)

            output_name = (
                f"{src.stem}_aug{aug_idx:02d}_{transform_name}{src.suffix.lower()}"
            )
            _save_img(aug, OUTPUT_DIR / output_name)

            run_log[output_name] = {
                "source_image": src.name,
                "transform": transform_name,
                **params,
            }

    # One JSON documenting every generated image and its exact parameters.
    log_entry = {
        "n_source_images": len(images),
        "augmentations_per_image": AUGMENTATIONS_PER_IMAGE,
        "n_augmented_images": len(images) * AUGMENTATIONS_PER_IMAGE,
        "available_transforms": transform_names,
        "seed": SEED,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "images": run_log,
    }
    (OUTPUT_DIR / "augmentation_log.json").write_text(
        json.dumps(log_entry, indent=2)
    )

    print(
        f"Generated {len(images) * AUGMENTATIONS_PER_IMAGE} augmented images "
        f"from {len(images)} source images -> {OUTPUT_DIR}"
    )
    print("    log      -> augmentation_log.json\n")
    print("Done.")


if __name__ == "__main__":
    main()