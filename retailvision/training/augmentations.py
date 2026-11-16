import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from retailvision.image_ops import mask_bbox, mask_crop, render_mask_crop


def training_transform():
    # Preserve printed labels and package orientation: no horizontal flip.
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(.15, .15, .1, .02),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])


def perturbed_mask_crop(
    image: Image.Image,
    mask: np.ndarray,
    rng: np.random.Generator,
    *,
    clean_prob: float = 0.35,
    max_pad_out: float = 0.16,
    max_pad_in: float = 0.04,
    erode_range: tuple[float, float] = (0.01, 0.04),
    dilate_range: tuple[float, float] = (0.015, 0.06),
    edge_cut_prob: float = 0.35,
    edge_cut_range: tuple[float, float] = (0.03, 0.11),
    bg_residue_prob: float = 0.35,
    min_fg_retention: float = 0.72,
    min_box_retention: float = 0.80,
    min_fg_share: float = 0.40,
) -> Image.Image:
    """Apply bounded crop shift, mask boundary error, and background residue.

    Keeps ``clean_prob`` exact ``mask_crop`` samples and enforces minimum GT
    foreground retention so perturbations never drop the main object or flip
    SKU label identity.
    """
    if mask.shape != (image.height, image.width):
        raise ValueError('invalid or empty instance mask')
    if rng.random() < clean_prob:
        return mask_crop(image, mask, padding=0.05)

    x1, y1, x2, y2 = mask_bbox(mask)
    w, h = x2 - x1, y2 - y1
    scale = max(w, h)
    min_dim = min(w, h)
    gt_area = int(mask.sum())

    pads = rng.uniform(-max_pad_in, max_pad_out, size=4)
    left = max(0, x1 - int(round(scale * float(pads[0]))))
    top = max(0, y1 - int(round(scale * float(pads[1]))))
    right = min(image.width, x2 + int(round(scale * float(pads[2]))))
    bottom = min(image.height, y2 + int(round(scale * float(pads[3]))))
    if right <= left or bottom <= top:
        return mask_crop(image, mask, padding=0.05)

    sub_gt = mask[top:bottom, left:right]
    in_box_fg = int(sub_gt.sum())
    if in_box_fg < min_box_retention * gt_area:
        return mask_crop(image, mask, padding=0.05)

    sub_pert = sub_gt.astype(np.uint8)
    # 1. Mask boundary under/over-segmentation via elliptical morphology.
    if rng.random() < 0.85:
        if rng.random() < 0.45:
            radius = max(1, int(round(min_dim * float(rng.uniform(*erode_range)))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            sub_pert = cv2.erode(sub_pert, kernel)
        else:
            radius = max(1, int(round(min_dim * float(rng.uniform(*dilate_range)))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            sub_pert = cv2.dilate(sub_pert, kernel)

    # 2. Partial missing boundary / occlusion along one random side.
    if rng.random() < edge_cut_prob:
        frac = float(rng.uniform(*edge_cut_range))
        side = int(rng.integers(0, 4))
        gt_sub_box = np.flatnonzero(sub_gt.any(axis=0))
        gt_sub_rows = np.flatnonzero(sub_gt.any(axis=1))
        if len(gt_sub_box) and len(gt_sub_rows):
            gx1, gx2 = int(gt_sub_box.min()), int(gt_sub_box.max()) + 1
            gy1, gy2 = int(gt_sub_rows.min()), int(gt_sub_rows.max()) + 1
            if side == 0:
                cut = gx1 + max(1, int(round((gx2 - gx1) * frac)))
                sub_pert[:, :cut] = 0
            elif side == 1:
                cut = gx2 - max(1, int(round((gx2 - gx1) * frac)))
                sub_pert[:, cut:] = 0
            elif side == 2:
                cut = gy1 + max(1, int(round((gy2 - gy1) * frac)))
                sub_pert[:cut, :] = 0
            else:
                cut = gy2 - max(1, int(round((gy2 - gy1) * frac)))
                sub_pert[cut:, :] = 0

    pert_bool = sub_pert.astype(bool)
    box_area = (right - left) * (bottom - top)
    unmasked_bg = rng.random() < bg_residue_prob and (in_box_fg >= min_fg_share * box_area)
    visible_fg = in_box_fg if unmasked_bg else int((pert_bool & sub_gt).sum())
    if not pert_bool.any() or visible_fg < min_fg_retention * gt_area:
        return mask_crop(image, mask, padding=0.05)

    local_mask = np.zeros_like(mask, dtype=bool)
    local_mask[top:bottom, left:right] = pert_bool
    return render_mask_crop(image, local_mask, (left, top, right, bottom), apply_mask=not unmasked_bg)
