"""V5-only quality helpers. The existing V4 notebook and its embedded code stay unchanged."""
from __future__ import annotations

import json
import hashlib
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.box_to_mask_sam2 import UltralyticsSAM2Predictor
from src.map_and_merge_public import apply_geometric_mask_merger, load_class_mapping
from src.prepare_vn_frames import RawDetection, _is_ego_hood_car, _suppress_ped_overlapping_rider
from src.seg_utils import SegInstance, read_image_bgr, read_yolo_seg_file, write_yolo_seg_label
from src.validate_dataset import IMG_EXTS, bbox_iou, geom_iou, image_size, load_competition_config


class CorrectedSAM2Predictor(UltralyticsSAM2Predictor):
    """Ultralytics NumPy input is BGR; do not swap it to RGB a second time."""

    def __call__(self, img_bgr: np.ndarray, bboxes_xyxy: list[list[float]]) -> list[np.ndarray]:
        if not bboxes_xyxy:
            return []
        kwargs = {"bboxes": bboxes_xyxy, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model(img_bgr, **kwargs)
        h, w = img_bgr.shape[:2]
        masks = []
        if results and results[0].masks is not None:
            raw = results[0].masks.data.cpu().numpy()
            for mask in raw[:len(bboxes_xyxy)]:
                binary = (mask > .5).astype(np.uint8)
                if binary.shape != (h, w):
                    binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
                masks.append(binary)
        while len(masks) < len(bboxes_xyxy):
            masks.append(np.zeros((h, w), dtype=np.uint8))
        return masks


def _mapped_view(detections, mapping, merge_rules, rider_labels, needs_merge, shape, min_conf, thresholds):
    h, w = shape[:2]
    insts = []
    for det in detections:
        key = det.label_name.strip().lower()
        if key not in mapping:
            continue
        cid = mapping[key]
        if det.conf < thresholds.get(cid, min_conf):
            continue
        inst = SegInstance(cid, det.pts, source_label=key)
        if not _is_ego_hood_car(inst) and inst.area_640(w, h) >= 20.:
            insts.append(inst)
    if needs_merge:
        insts, _, _ = apply_geometric_mask_merger(insts, merge_rules, rider_labels, "teacher_view")
        insts = _suppress_ped_overlapping_rider(insts)
    return [it for it in insts if it.area_640(w, h) >= 36.]


def complete_labels_with_consensus(
    dataset_dir: Path, predictor, preserved_classes: set[int], fill_classes: set[int],
    source_mapping_name: str = "coco", min_conf: float = .50,
    per_class_min_conf: dict[int, float] | None = None,
    agreement_iou: float = .55, split: str = "train",
    mapping_path: Path = Path("configs/class_mapping.yaml"),
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    """Add confident labels only when original and mirrored predictions agree.

    This is a conservative, offline consistency heuristic, not a FixMatch implementation
    or a guarantee of annotation completeness. Existing label polygons are retained.
    """
    cfg = load_competition_config(competition_path)
    mappings = load_class_mapping(mapping_path)
    if source_mapping_name == "competition_direct":
        mapping = {name.lower(): i for i, name in enumerate(cfg["classes"])}
        needs_merge, rider_labels = False, set()
    else:
        source = mappings["sources"][source_mapping_name]
        mapping = {str(k).lower(): int(v) for k, v in source["direct_map"].items()}
        needs_merge = source.get("needs_rider_merge", False)
        rider_labels = set(source.get("rider_labels", []))
    allowed = set(fill_classes) - set(preserved_classes)
    thresholds = per_class_min_conf or {}
    img_dir = dataset_dir / "images" / split
    lbl_dir = dataset_dir / "labels" / split
    added: Counter[int] = Counter()
    records = []
    for image in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS):
        relative = image.relative_to(img_dir)
        label = lbl_dir / relative.with_suffix(".txt")
        existing = read_yolo_seg_file(label)
        bgr = read_image_bgr(image)
        original = list(predictor(bgr))
        mirror = []
        for det in predictor(np.ascontiguousarray(bgr[:, ::-1])):
            pts = det.pts.copy()
            pts[:, 0] = 1. - pts[:, 0]
            mirror.append(RawDetection(det.label_name, det.conf, pts))
        view_a = _mapped_view(original, mapping, mappings["geometric_merge_rules"], rider_labels,
                              needs_merge, bgr.shape, min_conf, thresholds)
        view_b = _mapped_view(mirror, mapping, mappings["geometric_merge_rules"], rider_labels,
                              needs_merge, bgr.shape, min_conf, thresholds)
        accepted = []
        used_b = set()
        unmatched = 0
        for candidate in view_a:
            if candidate.cls not in allowed:
                continue
            # Protect existing GT across classes, including person fragments inside a rider mask.
            if _conflicts_with_gt(candidate, existing):
                continue
            matches = [(bbox_iou(candidate.bbox, other.bbox), j) for j, other in enumerate(view_b)
                       if j not in used_b and other.cls == candidate.cls]
            score, idx = max(matches, default=(0., -1))
            if score < agreement_iou:
                unmatched += 1
                continue
            if any(_conflicts_with_gt(candidate, [other]) for other in accepted):
                continue
            used_b.add(idx)
            accepted.append(candidate)
            added[candidate.cls] += 1
        # Count mirrored-only confident objects too: a missing detection is uncertainty, not evidence of background.
        unmatched += sum(1 for j, other in enumerate(view_b)
                         if j not in used_b and other.cls in allowed and not _conflicts_with_gt(other, existing + accepted))
        write_yolo_seg_label(label, [(it.cls, it.pts) for it in existing + accepted])
        records.append({"image": relative.as_posix(), "gt_kept": len(existing), "added": len(accepted),
                        "unresolved_confident_objects": unmatched})
    result = {"images_processed": len(records), "teacher_added_total": sum(added.values()),
              "teacher_added_by_class": {cfg["classes"][c]: n for c, n in added.items()},
              "method": "confidence + horizontal-flip agreement", "records": records}
    (dataset_dir / f"consensus_{source_mapping_name}_{split}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def _conflicts_with_gt(candidate: SegInstance, instances: list[SegInstance]) -> bool:
    for other in instances:
        if bbox_iou(candidate.bbox, other.bbox) >= .45 or geom_iou(candidate, other) >= .30:
            return True
        if candidate.geom is not None and other.geom is not None:
            intersection = candidate.geom.intersection(other.geom).area
            if intersection / max(candidate.geom.area, 1e-10) >= .70:
                return True
    return False


def quarantine_uncertain(dataset_dir: Path, reports: list[dict], quarantine_dir: Path) -> dict:
    """Quarantine disposable staged partial-label images with confident view disagreement."""
    rejected = {rec["image"] for report in reports for rec in report["records"]
                if rec["unresolved_confident_objects"] > 0}
    moved = 0
    for relative in sorted(rejected):
        source = dataset_dir / "images" / "train" / relative
        if not source.is_file():
            continue
        destination = quarantine_dir / dataset_dir.name / "images" / "train" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        label = dataset_dir / "labels" / "train" / Path(relative).with_suffix(".txt")
        if label.is_file():
            dest_label = quarantine_dir / dataset_dir.name / "labels" / "train" / Path(relative).with_suffix(".txt")
            dest_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label), str(dest_label))
        moved += 1
    return {"source": str(dataset_dir), "quarantined_images": moved}


def make_repeat_factor_dataset(
    master_dir: Path, out_yaml: Path, rare_classes=(3, 5, 6, 7), max_repeat: int = 3,
    max_extra_fraction: float = .35, clean_only: bool = False, seed: int = 39,
) -> dict[str, Any]:
    """Keep every GT class; repeat rare-class *images* only from human full-label sources.

    Paths in the train list may repeat; validation is never resampled. Sampling is
    capped to limit overfitting and majority-class labels within the same image remain intact.
    """
    cfg = yaml.safe_load((master_dir / "data.yaml").read_text(encoding="utf-8"))
    all_images = sorted(p for p in (master_dir / "images" / "train").glob("*") if p.suffix.lower() in IMG_EXTS)
    clean = lambda p: any(marker in p.stem.lower() for marker in ("mapillary", "bdd100k", "cityscapes")) and "aug_" not in p.stem
    images = [p for p in all_images if clean(p)] if clean_only else all_images
    if not images:
        raise ValueError("No fully annotated original images for the requested dataset")
    cls_by_image = {}
    frequencies: Counter[int] = Counter()
    instances: Counter[int] = Counter()
    for image in images:
        w, h = image_size(image)
        labels = read_yolo_seg_file(master_dir / "labels" / "train" / f"{image.stem}.txt")
        cls_set = {it.cls for it in labels if it.area_640(w, h) >= 64.}
        cls_by_image[image] = cls_set
        if clean(image):
            frequencies.update(cls_set)
        instances.update(it.cls for it in labels if it.area_640(w, h) >= 64.)
    if clean_only and any(instances[c] < 10 for c in range(len(cfg["names"]))):
        raise ValueError("Clean refinement subset is missing sufficient evaluable GT for one or more classes")
    extras = []
    clean_count = sum(clean(p) for p in images)
    for image in images:
        factor = 1
        if clean(image):
            for cid in cls_by_image[image] & set(rare_classes):
                # Target at least 10% of the fully annotated image pool.
                factor = max(factor, min(max_repeat, math.ceil(math.sqrt(.10 * clean_count / max(1, frequencies[cid])))))
        extras.extend([image] * (factor - 1))
    random.Random(seed).shuffle(extras)
    selected = images + extras[:int(len(images) * max_extra_fraction)]
    train_list = out_yaml.with_suffix(".train.txt")
    train_list.parent.mkdir(parents=True, exist_ok=True)
    train_list.write_text("\n".join(p.resolve().as_posix() for p in selected) + "\n", encoding="utf-8")
    cfg.update(path=master_dir.resolve().as_posix(), train=train_list.resolve().as_posix(), val="images/val")
    out_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = {"unique_images": len(images), "training_entries": len(selected), "clean_only": clean_only,
              "max_repeat": max_repeat, "evaluable_instances": dict(instances),
              "original_image_frequency": dict(frequencies)}
    out_yaml.with_suffix(".sampling.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def choose_official_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject fallback metrics, NaNs, and missing-class validation before choosing a checkpoint."""
    if not candidates:
        raise ValueError("No checkpoint candidates")
    expected_gt = None
    for candidate in candidates:
        summary = candidate["score_summary"]
        if summary.get("evaluator") != "official_detscore":
            raise ValueError("Checkpoint selection requires official_detscore, not a fallback score")
        score = float(summary["score"])
        if not math.isfinite(score):
            raise ValueError("Non-finite checkpoint score")
        if summary.get("classes_without_gt"):
            raise ValueError("Checkpoint validation lacks GT classes")
        gt = summary.get("n_gt_per_class")
        if not gt or len(gt) != 11 or any(n < 25 for n in gt.values()):
            raise ValueError("Checkpoint validation must contain >=25 evaluable GT for all 11 classes")
        if not candidate.get("holdout_sha256"):
            raise ValueError("Missing holdout content fingerprint")
        signature = (candidate["holdout_sha256"], candidate["n_images"], tuple(sorted(gt.items())))
        if expected_gt is not None and signature != expected_gt:
            raise ValueError("Checkpoint candidates were evaluated on different holdouts")
        expected_gt = signature
    return max(candidates, key=lambda candidate: float(candidate["score_summary"]["score"]))


def holdout_fingerprint(master_dir: Path) -> str:
    digest = hashlib.sha256()
    for sub in ("images/val", "labels/val"):
        for path in sorted(p for p in (master_dir / sub).rglob("*") if p.is_file()):
            digest.update(path.relative_to(master_dir).as_posix().encode("utf-8"))
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def cap_generated_images(dataset_dir: Path, max_images: int, seed: int = 39) -> int:
    """Cap disposable generated images; keep image/label pairs together."""
    images = sorted(p for p in (dataset_dir / "images/train").glob("*") if p.suffix.lower() in IMG_EXTS)
    random.Random(seed).shuffle(images)
    for image in images[max_images:]:
        image.unlink()
        (dataset_dir / "labels/train" / f"{image.stem}.txt").unlink(missing_ok=True)
    return min(len(images), max_images)
