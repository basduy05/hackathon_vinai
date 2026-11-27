"""Data-centric boosters for yolo26n-seg (imgsz=640, min_gt_area_px=64):
  1. ``crop-zoom``: Crops high-res images around small/rare targets (Sign, Traffic light, Other vehicle)
     with exact Shapely polygon clipping so small signs stay well above 64 px^2 at 640 scale.
  2. ``copy-paste``: Horizon-aware instance copy-paste from rare classes (3, 6, 7, 9, 10) onto
     diverse street backgrounds without heavy overlap collisions.
  3. ``weather``: Deterministic (TEAM_ID=39 seeded) night/rain/sun-glare augmentations preserving exact labels.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import box as shapely_box

from src.seg_utils import (
    SegInstance,
    extract_polygons_from_shapely,
    mask_to_polygon,
    rasterize_norm_polygon,
    read_image_bgr,
    read_yolo_seg_file,
    write_data_yaml,
    write_image,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, bbox_iou, load_competition_config, polygon_area


def _clip_instances_to_crop(
    instances: list[SegInstance],
    img_w: int,
    img_h: int,
    cx1: int,
    cy1: int,
    cx2: int,
    cy2: int,
    min_area_640: float = 48.0,
    min_retain_ratio: float = 0.35,
    imgsz: int = 640,
) -> list[tuple[int, np.ndarray]]:
    cw, ch = float(cx2 - cx1), float(cy2 - cy1)
    crop_norm = shapely_box(cx1 / img_w, cy1 / img_h, cx2 / img_w, cy2 / img_h)
    scale_640 = (imgsz / max(cw, ch)) ** 2
    kept: list[tuple[int, np.ndarray]] = []

    for inst in instances:
        if inst.geom is None or inst.geom.area <= 0:
            continue
        inter = inst.geom.intersection(crop_norm)
        if inter.is_empty:
            continue
        if (inter.area / inst.geom.area) < min_retain_ratio:
            continue
        for poly in extract_polygons_from_shapely(inter):
            local = poly.copy()
            local[:, 0] = np.clip((local[:, 0] * img_w - cx1) / cw, 0.0, 1.0)
            local[:, 1] = np.clip((local[:, 1] * img_h - cy1) / ch, 0.0, 1.0)
            area_px = polygon_area(local) * cw * ch
            if area_px * scale_640 >= min_area_640 and len(np.unique(np.round(local, 6), axis=0)) >= 3:
                kept.append((inst.cls, local))
    return kept


def generate_crop_zoom_dataset(
    src_dir: Path,
    out_dir: Path,
    split: str = "train",
    target_classes: tuple[int, ...] = (3, 6, 7, 9, 10),
    crop_size: int = 640,
    max_crops_per_image: int = 2,
    min_target_area_640: float = 16.0,
    max_target_area_640: float = 500.0,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    """Create zoomed crop patches around small/medium target instances (preserving full polygon accuracy)."""
    cfg = load_competition_config(competition_path)
    imgsz = int(cfg["competition"]["imgsz"])
    rng = random.Random(seed)

    in_img_dir = src_dir / "images" / split if (src_dir / "images" / split).is_dir() else (src_dir / "images")
    in_lbl_dir = src_dir / "labels" / split if (src_dir / "labels" / split).is_dir() else (src_dir / "labels")
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    crops_written = 0
    for img_path in sorted(p for p in in_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
        lbl_path = in_lbl_dir / f"{img_path.stem}.txt"
        insts = read_yolo_seg_file(lbl_path)
        if not insts:
            continue
        img_bgr = read_image_bgr(img_path)
        h, w = img_bgr.shape[:2]
        if min(w, h) < crop_size + 40:
            continue

        anchors = [
            it for it in insts
            if it.cls in target_classes and min_target_area_640 <= it.area_640(w, h, imgsz) <= max_target_area_640
        ]
        if not anchors:
            continue

        rng.shuffle(anchors)
        chosen_boxes: list[tuple[int, int, int, int]] = []
        for anc in anchors:
            if len(chosen_boxes) >= max_crops_per_image:
                break
            ax1, ay1, ax2, ay2 = anc.bbox
            acx = int(round(0.5 * (ax1 + ax2) * w)) + rng.randint(-crop_size // 6, crop_size // 6)
            acy = int(round(0.5 * (ay1 + ay2) * h)) + rng.randint(-crop_size // 6, crop_size // 6)
            x1 = max(0, min(w - crop_size, acx - crop_size // 2))
            y1 = max(0, min(h - crop_size, acy - crop_size // 2))
            x2, y2 = x1 + crop_size, y1 + crop_size
            # Avoid overlapping crops on same image
            if any(abs(x1 - px1) < crop_size * 0.4 and abs(y1 - py1) < crop_size * 0.4 for px1, py1, _, _ in chosen_boxes):
                continue
            clipped_insts = _clip_instances_to_crop(insts, w, h, x1, y1, x2, y2, imgsz=imgsz)
            if not clipped_insts:
                continue
            chosen_boxes.append((x1, y1, x2, y2))
            crop_img = img_bgr[y1:y2, x1:x2]
            stem = f"cropzoom_{img_path.stem}_{len(chosen_boxes)}"
            write_image(out_img_dir / f"{stem}.jpg", crop_img)
            write_yolo_seg_label(out_lbl_dir / f"{stem}.txt", clipped_insts)
            crops_written += 1

    write_data_yaml(out_dir, cfg["classes"], has_val=(out_dir / "images" / "val").is_dir())
    return {"mode": "crop-zoom", "crops_written": crops_written, "crop_size": crop_size}


def apply_weather_effect(img_bgr: np.ndarray, effect: str, rng: random.Random) -> np.ndarray:
    """Synthesize realistic night, rain streaks, or harsh sunlight glare for domain robustness."""
    out = img_bgr.astype(np.float32)
    h, w = out.shape[:2]

    if effect == "night":
        gamma = rng.uniform(1.8, 2.4)
        gain = rng.uniform(0.45, 0.65)
        out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma) * gain
        # Subtle warm/cool headlight halo glow in upper-middle band
        cx, cy = rng.randint(w // 5, 4 * w // 5), rng.randint(h // 4, 2 * h // 3)
        yy, xx = np.ogrid[:h, :w]
        dist2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / float((0.18 * w) ** 2)
        halo = np.exp(-dist2)[..., None] * rng.uniform(35.0, 75.0)
        out = np.clip(out + halo, 0, 255)
    elif effect == "rain":
        # Slight overcast darkening + diagonal rain streaks + gentle motion blur
        out = out * rng.uniform(0.75, 0.88) + 12.0
        rain_layer = np.zeros((h, w), dtype=np.uint8)
        n_drops = rng.randint(250, 500)
        dx = rng.randint(-6, 6)
        for _ in range(n_drops):
            rx = rng.randint(0, w - 1)
            ry = rng.randint(0, h - 1)
            length = rng.randint(12, 28)
            cv2.line(rain_layer, (rx, ry), (rx + dx, ry + length), 180, 1, cv2.LINE_AA)
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)[..., None].astype(np.float32)
        out = np.clip(out + 0.35 * rain_layer, 0, 255)
        out = cv2.GaussianBlur(out, (3, 3), 0.6)
    elif effect == "glare":
        # Harsh Vietnamese midday sun glare + slight contrast lift
        cx, cy = rng.randint(w // 4, 3 * w // 4), rng.randint(0, h // 3)
        yy, xx = np.ogrid[:h, :w]
        dist2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / float((0.28 * w) ** 2)
        flare = np.exp(-dist2)[..., None] * rng.uniform(70.0, 115.0)
        out = np.clip(out * rng.uniform(1.02, 1.12) + flare, 0, 255)
    else:
        raise ValueError(f"Unknown weather effect: {effect}")
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_weather_dataset(
    src_dir: Path,
    out_dir: Path,
    split: str = "train",
    effects: tuple[str, ...] = ("night", "rain", "glare"),
    max_images: int = 300,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    cfg = load_competition_config(competition_path)
    rng = random.Random(seed)
    in_img_dir = src_dir / "images" / split if (src_dir / "images" / split).is_dir() else (src_dir / "images")
    in_lbl_dir = src_dir / "labels" / split if (src_dir / "labels" / split).is_dir() else (src_dir / "labels")
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(p for p in in_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    rng.shuffle(candidates)
    written = 0
    for img_path in candidates[:max_images]:
        lbl_path = in_lbl_dir / f"{img_path.stem}.txt"
        insts = read_yolo_seg_file(lbl_path)
        img_bgr = read_image_bgr(img_path)
        eff = rng.choice(effects)
        aug_bgr = apply_weather_effect(img_bgr, eff, rng)
        stem = f"weather_{eff}_{img_path.stem}"
        write_image(out_img_dir / f"{stem}.jpg", aug_bgr)
        write_yolo_seg_label(out_lbl_dir / f"{stem}.txt", [(it.cls, it.pts) for it in insts])
        written += 1

    write_data_yaml(out_dir, cfg["classes"], has_val=(out_dir / "images" / "val").is_dir())
    return {"mode": "weather", "images_written": written, "effects": list(effects)}


def generate_copypaste_dataset(
    bank_dir: Path | list[Path],
    bg_dir: Path,
    out_dir: Path,
    split: str = "train",
    rare_classes: tuple[int, ...] = (3, 6, 7, 9, 10),
    n_output: int = 200,
    pastes_per_image: tuple[int, int] = (2, 4),
    max_bg_overlap_iou: float = 0.15,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    """Horizon-aware instance Copy-Paste for rare classes onto street backgrounds."""
    cfg = load_competition_config(competition_path)
    imgsz = int(cfg["competition"]["imgsz"])
    rng = random.Random(seed)

    # Combine all eligible banks without staging additional image copies.
    bank_dirs = [bank_dir] if isinstance(bank_dir, Path) else list(bank_dir)
    bank_items: list[tuple[Path, int, np.ndarray]] = []
    for root in bank_dirs:
        bank_img_dir = root / "images" / split if (root / "images" / split).is_dir() else root / "images"
        bank_lbl_dir = root / "labels" / split if (root / "labels" / split).is_dir() else root / "labels"
        for img_p in sorted(p for p in bank_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
            for it in read_yolo_seg_file(bank_lbl_dir / f"{img_p.stem}.txt"):
                if it.cls in rare_classes:
                    bank_items.append((img_p, it.cls, it.pts))
    if not bank_items:
        return {"mode": "copy-paste", "images_written": 0, "pasted_instances": 0, "reason": "empty_instance_bank"}

    bg_img_dir = bg_dir / "images" / split if (bg_dir / "images" / split).is_dir() else (bg_dir / "images")
    bg_lbl_dir = bg_dir / "labels" / split if (bg_dir / "labels" / split).is_dir() else (bg_dir / "labels")
    bg_imgs = sorted(p for p in bg_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not bg_imgs:
        return {"mode": "copy-paste", "images_written": 0, "pasted_instances": 0, "reason": "empty_backgrounds"}

    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Sample classes uniformly first so large Sign banks cannot overwhelm Cyclist.
    bank_by_class = {}
    for item in bank_items:
        bank_by_class.setdefault(item[1], []).append(item)
    available_classes = sorted(bank_by_class)
    images_written = total_pasted = 0
    for out_idx in range(n_output):
        bg_path = rng.choice(bg_imgs)
        bg_bgr = read_image_bgr(bg_path).copy()
        bh, bw = bg_bgr.shape[:2]
        bg_insts = read_yolo_seg_file(bg_lbl_dir / f"{bg_path.stem}.txt")
        final_insts: list[tuple[int, np.ndarray]] = [(it.cls, it.pts) for it in bg_insts]
        occupied_boxes = [it.bbox for it in bg_insts]

        k_pastes = rng.randint(pastes_per_image[0], pastes_per_image[1])
        pasted_here = 0
        for _ in range(k_pastes * 3):
            if pasted_here >= k_pastes:
                break
            src_img_p, cls_id, src_pts = rng.choice(bank_by_class[rng.choice(available_classes)])
            src_bgr = read_image_bgr(src_img_p)
            sh, sw = src_bgr.shape[:2]
            src_mask = rasterize_norm_polygon(src_pts, sw, sh)
            ys, xs = np.nonzero(src_mask)
            if len(xs) < 20:
                continue
            sx1, sx2 = int(xs.min()), int(xs.max()) + 1
            sy1, sy2 = int(ys.min()), int(ys.max()) + 1
            patch_bgr = src_bgr[sy1:sy2, sx1:sx2]
            patch_mask = src_mask[sy1:sy2, sx1:sx2]
            ph, pw = patch_bgr.shape[:2]
            if ph < 8 or pw < 8:
                continue

            # Signs/Traffic lights belong higher in frame; ground vehicles belong lower half
            if cls_id in (9, 10):
                ty_center = rng.uniform(0.18, 0.50)
            else:
                ty_center = rng.uniform(0.52, 0.85)
            tx_center = rng.uniform(0.10, 0.90)

            # Horizon-aware scale: objects closer to bottom can be slightly larger
            horizon_scale = (0.7 + 0.6 * ty_center) * rng.uniform(0.85, 1.15)
            new_w = max(12, min(int(bw * 0.45), int(round(pw * (bw / float(sw)) * horizon_scale))))
            new_h = max(12, min(int(bh * 0.45), int(round(ph * (bh / float(sh)) * horizon_scale))))
            if (new_w * new_h) * ((imgsz / max(bw, bh)) ** 2) < 80.0:
                continue

            tx1 = max(0, min(bw - new_w, int(round(tx_center * bw - 0.5 * new_w))))
            ty1 = max(0, min(bh - new_h, int(round(ty_center * bh - 0.5 * new_h))))
            cand_box = (tx1 / bw, ty1 / bh, (tx1 + new_w) / bw, (ty1 + new_h) / bh)
            if any(bbox_iou(cand_box, ob) >= max_bg_overlap_iou for ob in occupied_boxes):
                continue

            resized_patch = cv2.resize(patch_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Soft boundary feathering + subtle luminance matching
            alpha = cv2.GaussianBlur(resized_mask.astype(np.float32), (3, 3), 0)[..., None]
            roi = bg_bgr[ty1:ty1 + new_h, tx1:tx1 + new_w].astype(np.float32)
            patch_f = resized_patch.astype(np.float32)
            bg_mean = float(roi.mean()) + 1e-3
            fg_mean = float(patch_f[resized_mask > 0].mean()) + 1e-3 if np.any(resized_mask > 0) else bg_mean
            gain = np.clip(bg_mean / fg_mean, 0.72, 1.35)
            blended = np.clip(alpha * (patch_f * gain) + (1.0 - alpha) * roi, 0, 255).astype(np.uint8)
            bg_bgr[ty1:ty1 + new_h, tx1:tx1 + new_w] = blended

            full_mask = np.zeros((bh, bw), dtype=np.uint8)
            full_mask[ty1:ty1 + new_h, tx1:tx1 + new_w] = resized_mask
            new_poly = mask_to_polygon(full_mask, bw, bh)
            if new_poly is None:
                continue
            final_insts.append((cls_id, new_poly))
            occupied_boxes.append(cand_box)
            pasted_here += 1
            total_pasted += 1

        if pasted_here > 0:
            stem = f"copypaste_{out_idx:05d}_{bg_path.stem}"
            write_image(out_img_dir / f"{stem}.jpg", bg_bgr)
            write_yolo_seg_label(out_lbl_dir / f"{stem}.txt", final_insts)
            images_written += 1

    write_data_yaml(out_dir, cfg["classes"], has_val=(out_dir / "images" / "val").is_dir())
    return {"mode": "copy-paste", "images_written": images_written, "pasted_instances": total_pasted,
            "bank_instances_per_class": {str(c): len(items) for c, items in bank_by_class.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run data-centric augmentations (crop-zoom, copy-paste, weather).")
    parser.add_argument("--mode", required=True, choices=["crop-zoom", "copy-paste", "weather"])
    parser.add_argument("--src-dir", type=Path, required=True, help="Source YOLO-seg directory (or instance bank for copy-paste)")
    parser.add_argument("--bg-dir", type=Path, default=None, help="Background dataset directory (defaults to --src-dir)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination augmented dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seed", type=int, default=39, help="TEAM_ID seed (default: 39)")
    parser.add_argument("--n-output", type=int, default=200, help="Output image count for copy-paste / weather")
    parser.add_argument("--crop-size", type=int, default=640, help="Crop square side for crop-zoom")
    args = parser.parse_args()

    if args.mode == "crop-zoom":
        res = generate_crop_zoom_dataset(args.src_dir, args.out_dir, split=args.split, crop_size=args.crop_size, seed=args.seed)
    elif args.mode == "weather":
        res = generate_weather_dataset(args.src_dir, args.out_dir, split=args.split, max_images=args.n_output, seed=args.seed)
    else:
        res = generate_copypaste_dataset(
            bank_dir=args.src_dir,
            bg_dir=args.bg_dir or args.src_dir,
            out_dir=args.out_dir,
            split=args.split,
            n_output=args.n_output,
            seed=args.seed,
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
