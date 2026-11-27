"""Quota-based holdout selection and pre-augmentation leakage guards for staged data."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.seg_utils import (
    collect_exclusion_hashes, is_near_duplicate, read_yolo_seg_file,
    transfer_image, write_data_yaml, write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, image_size, load_competition_config


def build_golden_holdout(
    val_sources: list[Path],
    human_train_sources: list[Path],
    out_dir: Path,
    min_per_class: int = 25,
    target_images: int = 320,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    """Select original validation first, then staged human-label holdouts.

    Callers must explicitly supply trusted sources; this cannot certify mask quality.
    Quotas count objects above the evaluator's area threshold. Source files are kept.
    """
    if min_per_class < 1 or target_images < 1:
        raise ValueError("Golden quotas and target_images must be positive")
    if any((out_dir / sub).exists() and any((out_dir / sub).rglob("*"))
           for sub in ("images", "labels")):
        raise ValueError("Golden output must be empty; do not overwrite a locked holdout")
    cfg = load_competition_config(competition_path)
    classes = cfg["classes"]
    comp = cfg["competition"]
    rng = random.Random(seed)
    counts: Counter[int] = Counter()
    seen_shas: set[str] = set()
    seen_dhashes: list[int] = []
    manifest: list[dict[str, Any]] = []
    skipped_duplicates = 0

    # Process each tier independently to preserve original-val preference.
    for split, sources in (("val", val_sources), ("train", human_train_sources)):
        candidates = []
        for si, src in enumerate(sources):
            for im in sorted((src / "images" / split).rglob("*")):
                if not im.is_file() or im.suffix.lower() not in IMG_EXTS:
                    continue
                rel = im.relative_to(src / "images" / split)
                label = src / "labels" / split / rel.with_suffix(".txt")
                insts = read_yolo_seg_file(label)
                w, h = image_size(im)
                valid = Counter(it.cls for it in insts
                                if 0 <= it.cls < len(classes)
                                and it.area_640(w, h, int(comp["imgsz"])) >= float(comp["min_gt_area_px"]))
                if valid:
                    candidates.append((si, src, im, insts, valid))
        rng.shuffle(candidates)
        while candidates:
            deficits = {c: max(0, min_per_class - counts[c]) for c in range(len(classes))}
            # Greedy, recomputed after every selection; do not waste quota on majority classes.
            scores = [sum(min(v, deficits[c]) for c, v in rec[4].items()) for rec in candidates]
            best = max(range(len(candidates)), key=lambda i: scores[i])
            if scores[best] == 0 and len(manifest) >= target_images:
                break
            si, src, im, insts, valid = candidates.pop(best)
            duplicate, sha, dh = is_near_duplicate(
                im, seen_shas, seen_dhashes, int(cfg["qc_thresholds"]["dup_hamming_max"]))
            if duplicate:
                skipped_duplicates += 1
                continue
            seen_shas.add(sha)
            seen_dhashes.append(dh)
            stem = f"{split}_{si:02d}_{len(manifest):05d}_{im.stem}"
            transfer_image(im, out_dir / "images" / "val" / f"{stem}{im.suffix.lower()}", mode="copy")
            write_yolo_seg_label(out_dir / "labels" / "val" / f"{stem}.txt",
                                 [(it.cls, it.pts) for it in insts])
            counts.update(valid)
            manifest.append({"source": str(src), "original_image": str(im), "original_split": split,
                             "output_stem": stem, "evaluable_instances": dict(valid)})
    missing = {classes[c]: min_per_class - counts[c] for c in range(len(classes)) if counts[c] < min_per_class}
    report = {"ok": not missing, "min_per_class": min_per_class, "images": len(manifest),
              "evaluable_per_class": {name: counts[c] for c, name in enumerate(classes)},
              "missing_instances": missing, "skipped_duplicates": skipped_duplicates,
              "mask_quality": "Upstream polygons or SAM2 masks from human boxes; visual audit required",
              "manifest": manifest}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_dir, classes, has_val=True)
    (out_dir / "holdout_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def exclude_golden_from_staged_train(
    train_sources: list[Path], golden_dir: Path,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, int]:
    """Remove exact/near matches from disposable staged copies BEFORE augmentation.

    Raw upstream datasets must never be passed as train_sources.
    """
    cfg = load_competition_config(competition_path)
    shas, dhashes = collect_exclusion_hashes(golden_dir / "images" / "val")
    if not shas:
        raise ValueError("Cannot exclude leakage against an empty Golden Val")
    removed: dict[str, int] = {}
    for src in train_sources:
        removed[str(src)] = 0
        for im in sorted((src / "images" / "train").rglob("*")):
            if im.is_file() and im.suffix.lower() in IMG_EXTS:
                dup, _, _ = is_near_duplicate(im, shas, dhashes, int(cfg["qc_thresholds"]["dup_hamming_max"]))
                if dup:
                    rel = im.relative_to(src / "images" / "train")
                    im.unlink()
                    (src / "labels" / "train" / rel.with_suffix(".txt")).unlink(missing_ok=True)
                    removed[str(src)] += 1
    return removed
