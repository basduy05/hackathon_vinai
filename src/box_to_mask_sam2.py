"""Upgrade BBox-only YOLO datasets (e.g. VNTS, Zalo Traffic Sign, Roboflow VN) to YOLO-seg polygons via SAM 2.

Prevents pole bleed on Sign (9) and Traffic light (10) by evaluating mask containment inside
each prompt bounding box and optionally clipping excess bleed outside a small margin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.seg_utils import (
    mask_to_polygon,
    read_image_bgr,
    register_dataset_usage,
    transfer_image,
    write_data_yaml,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, load_competition_config


BoxPredictor = Callable[[np.ndarray, list[list[float]]], list[np.ndarray]]


class UltralyticsSAM2Predictor:
    """Thin wrapper around ``ultralytics.SAM`` returning one ``(H, W)`` uint8 mask per prompt box."""

    def __init__(self, model_name: str = "sam2_b.pt", device: str | None = None) -> None:
        from ultralytics import SAM  # noqa: PLC0415

        self.model = SAM(model_name)
        self.device = device

    def __call__(self, img_bgr: np.ndarray, bboxes_xyxy: list[list[float]]) -> list[np.ndarray]:
        if not bboxes_xyxy:
            return []
        import cv2  # noqa: PLC0415

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        kwargs: dict[str, Any] = {"bboxes": bboxes_xyxy, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(img_rgb, **kwargs)
        h, w = img_bgr.shape[:2]
        masks_out: list[np.ndarray] = []
        if results and results[0].masks is not None:
            raw_masks = results[0].masks.data.cpu().numpy()
            for i in range(min(len(bboxes_xyxy), len(raw_masks))):
                masks_out.append((raw_masks[i] > 0.5).astype(np.uint8))
        while len(masks_out) < len(bboxes_xyxy):
            masks_out.append(np.zeros((h, w), dtype=np.uint8))
        return masks_out


def parse_yolo_bbox_file(
    label_path: Path,
    width: int,
    height: int,
    class_map: dict[int, int] | None = None,
    default_class: int | None = None,
) -> list[tuple[int, list[float]]]:
    """Read YOLO 5-column bbox lines into ``[(target_cls, [x1, y1, x2, y2] px), ...]``."""
    if not label_path.is_file():
        return []
    out: list[tuple[int, list[float]]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        cols = raw.split()
        if len(cols) != 5:
            continue
        raw_cls = int(float(cols[0]))
        if class_map is not None:
            if raw_cls not in class_map:
                continue
            target_cls = int(class_map[raw_cls])
        elif default_class is not None:
            target_cls = int(default_class)
        else:
            target_cls = raw_cls

        xc, yc, bw, bh = [float(x) for x in cols[1:]]
        x1 = max(0.0, (xc - 0.5 * bw) * width)
        y1 = max(0.0, (yc - 0.5 * bh) * height)
        x2 = min(float(width), (xc + 0.5 * bw) * width)
        y2 = min(float(height), (yc + 0.5 * bh) * height)
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        out.append((target_cls, [x1, y1, x2, y2]))
    return out


def clip_mask_to_box(mask: np.ndarray, box_xyxy: list[float], pad_ratio: float = 0.03) -> np.ndarray:
    """Zero out SAM2 bleed outside the slightly padded prompt bounding box."""
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    px1 = max(0, int(np.floor(x1 - pad_ratio * bw)))
    py1 = max(0, int(np.floor(y1 - pad_ratio * bh)))
    px2 = min(w, int(np.ceil(x2 + pad_ratio * bw)))
    py2 = min(h, int(np.ceil(y2 + pad_ratio * bh)))
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[py1:py2, px1:px2] = mask[py1:py2, px1:px2]
    return clipped


def box_to_rect_polygon(box_xyxy: list[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    return np.array(
        [
            [x1 / width, y1 / height],
            [x2 / width, y1 / height],
            [x2 / width, y2 / height],
            [x1 / width, y2 / height],
        ],
        dtype=np.float64,
    )


def convert_bbox_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    predictor: BoxPredictor,
    split: str = "train",
    class_map: dict[int, int] | None = None,
    default_class: int | None = None,
    clip_pad_ratio: float = 0.03,
    min_inside_ratio: float = 0.85,
    min_box_fill_ratio: float = 0.25,
    min_area_640: float = 40.0,
    fallback_rect_for_signs: bool = True,
    competition_path: Path = Path("configs/competition.yaml"),
    transfer_mode: str = "copy",
    registry_path: Path | None = None,
    source_name: str = "vietnam_bbox_sam2",
    dataset_url: str = "",
    license_name: str = "",
) -> dict[str, Any]:
    """Convert all YOLO bbox labels under ``labels_dir`` to YOLO-seg masks using ``predictor``."""
    comp_cfg = load_competition_config(competition_path)
    classes: list[str] = comp_cfg["classes"]
    imgsz = int(comp_cfg["competition"]["imgsz"])

    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    review_queue: list[dict[str, Any]] = []
    n_images = n_instances = n_fallback = n_bleed_clipped = 0

    for img_path in image_paths:
        rel = img_path.relative_to(images_dir)
        lbl_path = labels_dir / rel.with_suffix(".txt")
        if not lbl_path.is_file():
            lbl_path = labels_dir / f"{img_path.stem}.txt"

        img_bgr = read_image_bgr(img_path)
        h, w = img_bgr.shape[:2]
        scale_640 = (imgsz / max(w, h)) ** 2
        boxes_info = parse_yolo_bbox_file(lbl_path, w, h, class_map=class_map, default_class=default_class)

        if not boxes_info:
            transfer_image(img_path, out_img_dir / img_path.name, mode=transfer_mode)
            write_yolo_seg_label(out_lbl_dir / f"{img_path.stem}.txt", [])
            n_images += 1
            continue

        bboxes = [b for _, b in boxes_info]
        raw_masks = predictor(img_bgr, bboxes)
        seg_instances: list[tuple[int, np.ndarray]] = []

        for idx, ((cls_id, box), raw_mask) in enumerate(zip(boxes_info, raw_masks)):
            x1, y1, x2, y2 = box
            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            if box_area * scale_640 < min_area_640:
                continue

            mask_bin = (np.asarray(raw_mask) > 0).astype(np.uint8)
            raw_area = float(mask_bin.sum())
            clipped = clip_mask_to_box(mask_bin, box, pad_ratio=clip_pad_ratio)
            clipped_area = float(clipped.sum())
            inside_ratio = (clipped_area / raw_area) if raw_area > 0 else 0.0
            fill_ratio = clipped_area / box_area

            if raw_area > 0 and inside_ratio < min_inside_ratio:
                n_bleed_clipped += 1
                review_queue.append({
                    "flag": "SAM2_BLEED_CLIPPED",
                    "image": img_path.name,
                    "box_idx": idx,
                    "cls": cls_id,
                    "inside_ratio": round(inside_ratio, 3),
                    "detail": "SAM2 mask spilled outside bbox (e.g., pole/background); clipped to box margin",
                })

            poly = mask_to_polygon(clipped, w, h) if clipped_area >= 12.0 else None
            if poly is None or fill_ratio < min_box_fill_ratio:
                if fallback_rect_for_signs and cls_id in (9, 10):
                    poly = box_to_rect_polygon(box, w, h)
                    n_fallback += 1
                    review_queue.append({
                        "flag": "SAM2_RECT_FALLBACK",
                        "image": img_path.name,
                        "box_idx": idx,
                        "cls": cls_id,
                        "fill_ratio": round(fill_ratio, 3),
                        "detail": "Used tight bbox rectangle fallback for rectangular Sign/Traffic light",
                    })
                elif poly is None:
                    review_queue.append({
                        "flag": "SAM2_EMPTY_MASK",
                        "image": img_path.name,
                        "box_idx": idx,
                        "cls": cls_id,
                        "detail": "SAM2 failed to extract valid polygon for bbox",
                    })
                    continue

            seg_instances.append((cls_id, poly))
            n_instances += 1

        transfer_image(img_path, out_img_dir / img_path.name, mode=transfer_mode)
        write_yolo_seg_label(out_lbl_dir / f"{img_path.stem}.txt", seg_instances)
        n_images += 1

    write_data_yaml(out_dir, classes, has_val=(out_dir / "images" / "val").is_dir())
    queue_path = out_dir / f"sam2_review_queue_{split}.json"
    queue_path.write_text(json.dumps(review_queue, indent=2, ensure_ascii=False), encoding="utf-8")

    if registry_path is not None:
        register_dataset_usage(
            registry_path=registry_path,
            name=source_name,
            url=dataset_url or "(VN BBox -> SAM2)",
            license_name=license_name or "Verify upstream license",
            raw_format="YOLO BBox -> SAM2 Seg",
            train_count=n_images if split == "train" else 0,
            val_count=n_images if split == "val" else 0,
            mapping_note=f"default_cls={default_class}, map={class_map}",
            status="Converted (SAM2)",
        )

    return {
        "images_processed": n_images,
        "instances_written": n_instances,
        "bleed_clipped": n_bleed_clipped,
        "rect_fallbacks": n_fallback,
        "review_queue_size": len(review_queue),
        "review_queue_path": str(queue_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO BBox datasets to YOLO-seg polygons using SAM 2.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder of source images")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Folder of YOLO BBox .txt labels")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output YOLO-seg dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--sam-model", default="sam2_b.pt", help="Ultralytics SAM2 checkpoint (e.g. sam2_b.pt, sam2_l.pt)")
    parser.add_argument("--device", default=None, help="Device string, e.g. '0' or 'cpu'")
    parser.add_argument("--default-class", type=int, default=None, help="Map all source boxes to this competition class ID (e.g. 9 for Sign)")
    parser.add_argument("--class-map", default=None, help="JSON string or path mapping raw_cls_id -> target_cls_id")
    parser.add_argument("--source-name", default="vietnam_bbox_sam2", help="Name to log into datasets_registry.md")
    parser.add_argument("--url", default="", help="Source URL for registry")
    parser.add_argument("--license", default="", help="License string for registry")
    args = parser.parse_args()

    cmap: dict[int, int] | None = None
    if args.class_map:
        p = Path(args.class_map)
        raw_obj = json.loads(p.read_text(encoding="utf-8") if p.is_file() else args.class_map)
        cmap = {int(k): int(v) for k, v in raw_obj.items()}

    predictor = UltralyticsSAM2Predictor(model_name=args.sam_model, device=args.device)
    summary = convert_bbox_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        out_dir=args.out_dir,
        predictor=predictor,
        split=args.split,
        class_map=cmap,
        default_class=args.default_class,
        source_name=args.source_name,
        dataset_url=args.url,
        license_name=args.license,
        registry_path=Path("datasets_registry.md"),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
