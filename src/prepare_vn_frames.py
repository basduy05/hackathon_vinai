"""Prepare Vietnamese dashcam frames & teacher pseudo-labels with deduplication and leakage protection.

Extracts non-duplicate keyframes from dashcam videos or image folders, rejects blurry frames,
prevents leakage against ``data/golden_val/``, and optionally runs a strong Ultralytics seg
teacher + Geometric Mask Merger (converting ``person + 2-wheeler`` into ``Motorcyclist`` / ``Cyclist``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from src.map_and_merge_public import apply_geometric_mask_merger, load_class_mapping
from src.seg_utils import (
    SegInstance,
    collect_exclusion_hashes,
    is_near_duplicate,
    read_image_bgr,
    register_dataset_usage,
    write_image,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, bbox_iou, dhash64, load_competition_config


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class RawDetection:
    label_name: str
    conf: float
    pts: np.ndarray  # normalized (N, 2)


TeacherPredictor = Callable[[np.ndarray], list[RawDetection]]


class UltralyticsTeacherPredictor:
    """Runs a strong Ultralytics segmentation teacher (e.g. ``yolo11x-seg.pt``)."""

    def __init__(self, weights: str = "yolo11x-seg.pt", conf: float = 0.25, imgsz: int = 1280, device: str | None = None) -> None:
        from ultralytics import YOLO  # noqa: PLC0415

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def __call__(self, img_bgr: np.ndarray) -> list[RawDetection]:
        kwargs: dict[str, Any] = {"conf": self.conf, "imgsz": self.imgsz, "retina_masks": True, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        res = self.model.predict(img_bgr, **kwargs)[0]
        if res.masks is None or res.boxes is None:
            return []
        h, w = img_bgr.shape[:2]
        out: list[RawDetection] = []
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)
        conf_arr = res.boxes.conf.cpu().numpy().astype(float)
        for idx, poly_xy in enumerate(res.masks.xy):
            if len(poly_xy) < 3:
                continue
            pts = np.asarray(poly_xy, dtype=np.float64).copy()
            pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
            pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
            cname = str(self.model.names[int(cls_arr[idx])]).strip().lower()
            out.append(RawDetection(label_name=cname, conf=float(conf_arr[idx]), pts=pts))
        return out


def laplacian_sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _dhash_from_bgr(img_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_LINEAR).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


def _is_dhash_near(dh: int, existing: list[int], max_hamming: int) -> bool:
    if not existing:
        return False
    arr = np.array(existing, dtype=np.uint64)
    dist = np.unpackbits((arr ^ np.uint64(dh)).view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
    return int(dist.min()) <= max_hamming


def extract_frames_from_sources(
    input_path: Path,
    out_img_dir: Path,
    exclude_dir: Path | None = None,
    stride_sec: float = 1.5,
    min_sharpness: float = 45.0,
    dup_hamming_max: int = 4,
    max_frames: int | None = None,
    prefix: str = "vn_dashcam",
) -> dict[str, Any]:
    """Collect sharp, deduplicated frames from a folder of videos and/or raw images."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    seen_shas, seen_dhashes = collect_exclusion_hashes(exclude_dir)
    kept: list[Path] = []
    skipped_blur = skipped_dup = 0

    files = [input_path] if input_path.is_file() else sorted(p for p in input_path.rglob("*") if p.is_file())
    for fp in files:
        if max_frames is not None and len(kept) >= max_frames:
            break
        ext = fp.suffix.lower()
        if ext in IMG_EXTS:
            is_dup, sha, dh = is_near_duplicate(fp, seen_shas, seen_dhashes, max_hamming=dup_hamming_max)
            if is_dup:
                skipped_dup += 1
                continue
            img = read_image_bgr(fp)
            if laplacian_sharpness(img) < min_sharpness:
                skipped_blur += 1
                continue
            seen_shas.add(sha)
            seen_dhashes.append(dh)
            dst = out_img_dir / f"{prefix}_{fp.stem}.jpg"
            write_image(dst, img)
            kept.append(dst)
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(fp))
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(fps * stride_sec)))
            frame_idx = 0
            try:
                while True:
                    if max_frames is not None and len(kept) >= max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % step == 0:
                        dh = _dhash_from_bgr(frame)
                        if _is_dhash_near(dh, seen_dhashes, dup_hamming_max):
                            skipped_dup += 1
                        elif laplacian_sharpness(frame) < min_sharpness:
                            skipped_blur += 1
                        else:
                            seen_dhashes.append(dh)
                            dst = out_img_dir / f"{prefix}_{fp.stem}_f{frame_idx:06d}.jpg"
                            write_image(dst, frame)
                            kept.append(dst)
                    frame_idx += 1
            finally:
                cap.release()

    return {
        "frames_kept": len(kept),
        "skipped_blur": skipped_blur,
        "skipped_duplicates": skipped_dup,
        "image_paths": [str(p) for p in kept],
    }


def _is_ego_hood_car(seg: SegInstance) -> bool:
    """Detect dashcam ego-vehicle hood/dashboard false positives at bottom image edge."""
    if seg.cls != 0:
        return False
    x1, y1, x2, y2 = seg.bbox
    bw = x2 - x1
    bh = y2 - y1
    yc = 0.5 * (y1 + y2)
    return y2 >= 0.94 and bw >= 0.32 and (yc >= 0.72 or (y1 >= 0.65 and bh <= 0.38))


def _suppress_ped_overlapping_rider(instances: list[SegInstance]) -> list[SegInstance]:
    """Drop spurious Pedestrian(8) labels sitting directly inside a merged Motorcyclist(5)/Cyclist(7)."""
    riders = [it for it in instances if it.cls in (5, 7)]
    if not riders:
        return instances
    out: list[SegInstance] = []
    for it in instances:
        if it.cls != 8:
            out.append(it)
            continue
        rx1, ry1, rx2, ry2 = it.bbox
        p_area = it.geom.area if it.geom is not None else max(1e-8, (rx2 - rx1) * (ry2 - ry1))
        overlap_rider = False
        for rd in riders:
            if bbox_iou(it.bbox, rd.bbox) < 0.08:
                continue
            if it.geom is not None and rd.geom is not None:
                inter = float(it.geom.intersection(rd.geom).area)
            else:
                dx1, dy1, dx2, dy2 = rd.bbox
                ix = max(0.0, min(rx2, dx2) - max(rx1, dx1))
                iy = max(0.0, min(ry2, dy2) - max(ry1, dy1))
                inter = ix * iy
            if inter / max(1e-8, p_area) >= 0.55:
                overlap_rider = True
                break
        if not overlap_rider:
            out.append(it)
    return out


def pseudo_label_folder(
    images_dir: Path,
    labels_out_dir: Path,
    predictor: TeacherPredictor,
    source_mapping_name: str = "coco",
    mapping_path: Path = Path("configs/class_mapping.yaml"),
    competition_path: Path = Path("configs/competition.yaml"),
    min_area_640: float = 20.0,
    low_conf_threshold: float = 0.45,
    filter_ego_hood: bool = True,
    filter_rider_ped_overlap: bool = True,
    registry_path: Path | None = None,
    dataset_name: str = "vn_dashcam_pseudolabel",
    dataset_url: str = "",
    license_name: str = "Open-source / CC",
    post_merge_min_area_640: float = 36.0,
    aux_sign_predictor: TeacherPredictor | None = None,
) -> dict[str, Any]:
    """Run teacher pseudo-labeling + Geometric Mask Merger and generate human review queue."""
    comp_cfg = load_competition_config(competition_path)
    map_cfg = load_class_mapping(mapping_path)
    imgsz = int(comp_cfg["competition"]["imgsz"])

    if source_mapping_name not in map_cfg["sources"]:
        # Fall back to competition direct class names
        direct_map = {name.lower(): idx for idx, name in enumerate(comp_cfg["classes"])}
        needs_merge = False
        rider_labels: set[str] = set()
    else:
        src_cfg = map_cfg["sources"][source_mapping_name]
        direct_map = {str(k).strip().lower(): int(v) for k, v in src_cfg["direct_map"].items()}
        needs_merge = bool(src_cfg.get("needs_rider_merge", False))
        rider_labels = {str(x).strip().lower() for x in src_cfg.get("rider_labels", [])}

    merge_rules = map_cfg["geometric_merge_rules"]
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    review_queue: list[dict[str, Any]] = []
    n_images = n_instances = n_merged = n_hood_dropped = 0

    for img_path in sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
        img_bgr = read_image_bgr(img_path)
        h, w = img_bgr.shape[:2]
        dets = list(predictor(img_bgr))
        if aux_sign_predictor is not None:
            dets.extend(d for d in aux_sign_predictor(img_bgr)
                        if direct_map.get(d.label_name.strip().lower()) == 9)
        insts: list[SegInstance] = []
        low_conf_count = 0

        for d in dets:
            key = d.label_name.strip().lower()
            if key not in direct_map:
                continue
            seg = SegInstance(cls=direct_map[key], pts=d.pts, source_label=key)
            if filter_ego_hood and _is_ego_hood_car(seg):
                n_hood_dropped += 1
                continue
            if seg.area_640(w, h, imgsz) < min_area_640:
                continue
            insts.append(seg)
            if d.conf < low_conf_threshold:
                low_conf_count += 1

        if needs_merge and insts:
            insts, ambig, mstats = apply_geometric_mask_merger(insts, merge_rules, rider_labels, img_path.name)
            n_merged += mstats["auto_merged"]
            review_queue.extend(ambig)
            if filter_rider_ped_overlap:
                insts = _suppress_ped_overlapping_rider(insts)

        insts = [it for it in insts if it.area_640(w, h, imgsz) >= post_merge_min_area_640]

        if low_conf_count > 0:
            review_queue.append({
                "flag": "PSEUDO_LOW_CONF",
                "image": img_path.name,
                "low_conf_instances": low_conf_count,
                "total_instances": len(insts),
                "detail": "Contains low-confidence teacher predictions; inspect VN edge cases (Bus/Truck/Other vehicle/Sign)",
            })

        write_yolo_seg_label(labels_out_dir / f"{img_path.stem}.txt", [(it.cls, it.pts) for it in insts])
        n_images += 1
        n_instances += len(insts)

    queue_path = labels_out_dir.parent / "pseudo_review_queue.json"
    queue_path.write_text(json.dumps(review_queue, indent=2, ensure_ascii=False), encoding="utf-8")

    if registry_path is not None:
        register_dataset_usage(
            registry_path=registry_path,
            name=dataset_name,
            url=dataset_url or "(VN dashcam frames)",
            license_name=license_name,
            raw_format="Video/Images -> Teacher Pseudo + Human QC",
            train_count=n_images,
            val_count=0,
            mapping_note=f"source_map={source_mapping_name}, merged={n_merged}",
            status="Needs Human QC",
        )

    return {
        "images_labeled": n_images,
        "instances_written": n_instances,
        "auto_merged_riders": n_merged,
        "hood_cars_dropped": n_hood_dropped,
        "review_queue_size": len(review_queue),
        "review_queue_path": str(queue_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract deduplicated VN dashcam frames and run teacher pseudo-labeling.")
    parser.add_argument("--input", type=Path, required=True, help="Video file, folder of videos, or folder of raw images")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output staged folder (creates images/ and labels/)")
    parser.add_argument("--exclude-dir", type=Path, default=Path("data/golden_val"), help="Exclude duplicates/leakage against this dir")
    parser.add_argument("--stride-sec", type=float, default=1.5, help="Seconds between extracted video frames")
    parser.add_argument("--min-sharpness", type=float, default=45.0, help="Minimum Laplacian variance sharpness")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap extracted frames")
    parser.add_argument("--pseudo-label", action="store_true", help="Run Ultralytics seg teacher to draft YOLO-seg labels")
    parser.add_argument("--teacher-weights", default="yolo11x-seg.pt", help="Teacher segmentation model path")
    parser.add_argument("--teacher-map", default="coco", help="Mapping key in configs/class_mapping.yaml for teacher output")
    parser.add_argument("--device", default=None, help="Device string for teacher inference")
    args = parser.parse_args()

    img_out = args.out_dir / "images"
    ext_summary = extract_frames_from_sources(
        input_path=args.input,
        out_img_dir=img_out,
        exclude_dir=args.exclude_dir if args.exclude_dir.exists() else None,
        stride_sec=args.stride_sec,
        min_sharpness=args.min_sharpness,
        max_frames=args.max_frames,
    )
    result: dict[str, Any] = {"extract": {k: v for k, v in ext_summary.items() if k != "image_paths"}}
    if args.pseudo_label and ext_summary["frames_kept"] > 0:
        teacher = UltralyticsTeacherPredictor(weights=args.teacher_weights, device=args.device)
        result["pseudo_label"] = pseudo_label_folder(
            images_dir=img_out,
            labels_out_dir=args.out_dir / "labels",
            predictor=teacher,
            source_mapping_name=args.teacher_map,
            registry_path=Path("datasets_registry.md"),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
