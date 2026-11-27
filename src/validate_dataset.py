"""QC Tier-1 dataset validator for AIInAction RoboTaxi (11-class YOLO-seg).

Implements both the starter notebook's data checks (polygon validity, area < 64 px^2 at 640 scale,
exact/near dhash duplicate & train/val leakage detection) AND the 5 edge-case geometric flags
specified in ``annotation_guideline_and_qc.md``:
  1. RIDER_LEAK: Pedestrian (8) overlapping/mounted over Motorcycle (4) or Bicycle (6)
  2. DUPLICATE_MASK: Same-class polygons with IoU > 0.85
  3. SIGN_POLE_SUSPECT: Sign (9) with tall aspect ratio h/w > 3.5 (pole likely included)
  4. LIGHT_BULB_ONLY: Traffic light (10) tiny near-square blobs clustered vertically
  5. TRAILER_MERGED: Truck (2) wide & elongated (likely tractor + semi-trailer merged)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps
from shapely.geometry import Polygon


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COORD_TOLERANCE = 1e-4


@dataclass
class PolyInstance:
    line_no: int
    cls: int
    pts: np.ndarray  # (N, 2) normalized [0, 1]
    area_640: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    geom: Polygon | None


def load_competition_config(config_path: str | Path = "configs/competition.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def image_size(path: Path) -> tuple[int, int]:
    """Return EXIF-corrected (width, height) without loading full pixel buffer."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(274)
        except Exception:  # noqa: BLE001
            orientation = None
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash64(path: Path) -> int:
    """64-bit difference hash matching ``train_and_export.ipynb``."""
    with Image.open(path) as im:
        try:
            im.draft("L", (96, 96))
        except Exception:  # noqa: BLE001
            pass
        im = ImageOps.exif_transpose(im).convert("L").resize((9, 8), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


def find_duplicate_pairs(hashes: list[int], max_distance: int) -> list[tuple[int, int, int]]:
    if len(hashes) < 2:
        return []
    arr = np.array(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int, int]] = []
    for i in range(len(arr) - 1):
        x = arr[i + 1 :] ^ arr[i]
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= max_distance)[0]:
            pairs.append((i, i + 1 + int(j), int(dist[j])))
    return pairs


def polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_self_intersects(pts: np.ndarray) -> bool:
    n = len(pts)
    if n < 4 or n > 400:
        return False
    a, b = pts, np.roll(pts, -1, axis=0)

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

    pa, pb, pc, pd = a[:, None, :], b[:, None, :], a[None, :, :], b[None, :, :]
    crossing = (orient(pa, pb, pc) * orient(pa, pb, pd) < 0) & (orient(pc, pd, pa) * orient(pc, pd, pb) < 0)
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    return bool((crossing & ~((gap <= 1) | (gap == n - 1))).any())


def make_shapely_polygon(pts: np.ndarray) -> Polygon | None:
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty and poly.area > 0 else None
    except Exception:  # noqa: BLE001
        return None


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def geom_iou(a: PolyInstance, b: PolyInstance) -> float:
    if bbox_iou(a.bbox, b.bbox) <= 0.0:
        return 0.0
    if a.geom is not None and b.geom is not None:
        try:
            inter = a.geom.intersection(b.geom).area
            union = a.geom.area + b.geom.area - inter
            return float(inter / union) if union > 0 else 0.0
        except Exception:  # noqa: BLE001
            pass
    return bbox_iou(a.bbox, b.bbox)


def discover_splits(data_dir: Path) -> dict[str, list[tuple[Path, Path | None]]]:
    """Discover images and matching label files across train/val or a flat pool."""
    splits: dict[str, list[tuple[Path, Path | None]]] = {}
    img_root = data_dir / "images"
    lbl_root = data_dir / "labels"

    if (img_root / "train").is_dir() or (img_root / "val").is_dir():
        for split in ("train", "val"):
            s_img_dir = img_root / split
            s_lbl_dir = lbl_root / split
            if not s_img_dir.is_dir():
                continue
            pairs = []
            for img in sorted(p for p in s_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
                rel = img.relative_to(s_img_dir)
                lbl = s_lbl_dir / rel.with_suffix(".txt")
                pairs.append((img, lbl if lbl.is_file() else None))
            splits[split] = pairs
        return splits

    # Flat layout (e.g., sample_data/images + sample_data/labels) or split/{images,labels}
    if any((data_dir / s / "images").is_dir() for s in ("train", "val")):
        for split in ("train", "val"):
            s_img_dir = data_dir / split / "images"
            s_lbl_dir = data_dir / split / "labels"
            if not s_img_dir.is_dir():
                continue
            pairs = []
            for img in sorted(p for p in s_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
                rel = img.relative_to(s_img_dir)
                lbl = s_lbl_dir / rel.with_suffix(".txt")
                pairs.append((img, lbl if lbl.is_file() else None))
            splits[split] = pairs
        return splits

    search_img = img_root if img_root.is_dir() else data_dir
    search_lbl = lbl_root if lbl_root.is_dir() else data_dir
    pairs = []
    for img in sorted(p for p in search_img.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
        rel = img.relative_to(search_img)
        lbl = search_lbl / rel.with_suffix(".txt")
        pairs.append((img, lbl if lbl.is_file() else None))
    if pairs:
        splits["pool"] = pairs
    return splits


def parse_and_validate_label(
    label_path: Path | None,
    width: int,
    height: int,
    nc: int,
    imgsz: int,
    shown_name: str,
) -> tuple[list[PolyInstance], list[str], list[str]]:
    """Parse one label file into ``PolyInstance`` objects and separate fatal vs polygon-quality issues."""
    instances: list[PolyInstance] = []
    fatal: list[str] = []
    poly_issues: list[str] = []
    if label_path is None:
        return instances, fatal, poly_issues

    text = label_path.read_text(encoding="utf-8", errors="replace")
    scale_640 = (imgsz / max(width, height)) ** 2
    for line_no, raw in enumerate(text.splitlines(), 1):
        cols = raw.split()
        if not cols:
            continue
        where = f"{shown_name}:{line_no}"
        if len(cols) == 5:
            fatal.append(f"{where}: 5 columns (bbox detection format), expected YOLO-seg polygon")
            continue
        if len(cols) < 7 or len(cols) % 2 == 0:
            fatal.append(f"{where}: {len(cols)} columns, expected odd count >= 7 (`cls x1 y1 ...`)")
            continue
        try:
            cls_f = float(cols[0])
            vals = np.array([float(x) for x in cols[1:]], dtype=np.float64)
        except ValueError:
            fatal.append(f"{where}: non-numeric token in label")
            continue
        if not np.isfinite(cls_f) or cls_f != int(cls_f):
            fatal.append(f"{where}: class id {cols[0]!r} is not an integer")
            continue
        cls = int(cls_f)
        if not 0 <= cls < nc:
            fatal.append(f"{where}: class id {cls} out of 0..{nc - 1}")
            continue
        if not np.isfinite(vals).all():
            fatal.append(f"{where}: contains NaN/Inf")
            continue
        if vals.min() < -COORD_TOLERANCE or vals.max() > 1.0 + COORD_TOLERANCE:
            fatal.append(f"{where}: coordinates out of [0, 1] (min={vals.min():.4f}, max={vals.max():.4f})")
            continue

        pts = np.clip(vals, 0.0, 1.0).reshape(-1, 2)
        if len(np.unique(np.round(pts, 6), axis=0)) < 3:
            poly_issues.append(f"{where} (cls={cls}): fewer than 3 distinct vertices")
            continue
        if polygon_self_intersects(pts):
            poly_issues.append(f"{where} (cls={cls}): self-intersecting polygon")
        area_px = polygon_area(pts) * width * height
        if area_px < 1.0:
            poly_issues.append(f"{where} (cls={cls}): degenerate area < 1 px^2 ({area_px:.2f})")
        bbox = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))
        instances.append(
            PolyInstance(
                line_no=line_no,
                cls=cls,
                pts=pts,
                area_640=area_px * scale_640,
                bbox=bbox,
                geom=make_shapely_polygon(pts),
            )
        )
    return instances, fatal, poly_issues


def detect_edge_case_flags(
    instances: list[PolyInstance],
    thresholds: dict[str, Any],
    image_rel: str,
) -> list[dict[str, Any]]:
    """Run the 5 heuristic geometric edge-case detectors on one image's instances."""
    flags: list[dict[str, Any]] = []
    dup_iou_thr = float(thresholds.get("duplicate_mask_iou", 0.85))
    rider_iou_thr = float(thresholds.get("rider_leak_iou", 0.10))
    sign_ar_thr = float(thresholds.get("sign_pole_aspect_ratio", 3.5))
    trailer_min_w = float(thresholds.get("trailer_merged_min_width_norm", 0.50))
    trailer_ar_thr = float(thresholds.get("trailer_merged_aspect_ratio", 2.5))
    bulb_tiny_px = float(thresholds.get("light_bulb_tiny_px_640", 144.0))

    # 1. RIDER_LEAK: Pedestrian (8) near/on Motorcycle (4) or Bicycle (6)
    peds = [it for it in instances if it.cls == 8]
    bikes = [it for it in instances if it.cls in (4, 6)]
    for ped in peds:
        px1, py1, px2, py2 = ped.bbox
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        for bk in bikes:
            bx1, by1, bx2, by2 = bk.bbox
            bw, bh = max(1e-6, bx2 - bx1), max(1e-6, by2 - by1)
            bcx = 0.5 * (bx1 + bx2)
            iou = geom_iou(ped, bk)
            above_mounted = (abs(pcx - bcx) <= 0.65 * bw) and (py2 >= by1 - 0.12 * bh) and (pcy <= by2)
            if iou >= rider_iou_thr or (bbox_iou(ped.bbox, bk.bbox) >= 0.05 and above_mounted):
                flags.append({
                    "flag": "RIDER_LEAK",
                    "image": image_rel,
                    "lines": [ped.line_no, bk.line_no],
                    "detail": f"Pedestrian(L{ped.line_no}) overlaps/mounts cls={bk.cls}(L{bk.line_no}) (IoU={iou:.2f}); verify if should be Motorcyclist(5)/Cyclist(7)",
                })

    # 2. DUPLICATE_MASK: same-class high IoU
    by_cls: dict[int, list[PolyInstance]] = defaultdict(list)
    for it in instances:
        by_cls[it.cls].append(it)
    for cls, group in by_cls.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                iou = geom_iou(group[i], group[j])
                if iou > dup_iou_thr:
                    flags.append({
                        "flag": "DUPLICATE_MASK",
                        "image": image_rel,
                        "lines": [group[i].line_no, group[j].line_no],
                        "detail": f"cls={cls} duplicate overlap IoU={iou:.2f}",
                    })

    # 3. SIGN_POLE_SUSPECT (9) & 5. TRAILER_MERGED (2)
    for it in instances:
        w = max(1e-6, it.bbox[2] - it.bbox[0])
        h = max(1e-6, it.bbox[3] - it.bbox[1])
        clipped_h = it.bbox[0] <= 0.005 or it.bbox[2] >= 0.995
        if it.cls == 9 and not clipped_h and (h / w) > sign_ar_thr:
            flags.append({
                "flag": "SIGN_POLE_SUSPECT",
                "image": image_rel,
                "lines": [it.line_no],
                "detail": f"Sign(L{it.line_no}) h/w={h/w:.2f} > {sign_ar_thr}; check if pole was included",
            })
        elif it.cls == 2 and w > trailer_min_w and (w / h) > trailer_ar_thr:
            flags.append({
                "flag": "TRAILER_MERGED",
                "image": image_rel,
                "lines": [it.line_no],
                "detail": f"Truck(L{it.line_no}) norm_w={w:.2f}, w/h={w/h:.2f}; check if semi-trailer should be split as Other vehicle(3)",
            })

    # 4. LIGHT_BULB_ONLY (10): tiny square-ish clusters of traffic lights stacked vertically
    lights = by_cls.get(10, [])
    for i, lt in enumerate(lights):
        w = max(1e-6, lt.bbox[2] - lt.bbox[0])
        h = max(1e-6, lt.bbox[3] - lt.bbox[1])
        ar = w / h
        if lt.area_640 < bulb_tiny_px and 0.65 <= ar <= 1.55:
            lcx, lcy = 0.5 * (lt.bbox[0] + lt.bbox[2]), 0.5 * (lt.bbox[1] + lt.bbox[3])
            clustered = any(
                j != i
                and abs(lcx - 0.5 * (o.bbox[0] + o.bbox[2])) <= 1.2 * w
                and abs(lcy - 0.5 * (o.bbox[1] + o.bbox[3])) <= 2.5 * h
                for j, o in enumerate(lights)
            )
            if clustered:
                flags.append({
                    "flag": "LIGHT_BULB_ONLY",
                    "image": image_rel,
                    "lines": [lt.line_no],
                    "detail": f"Traffic light(L{lt.line_no}) is tiny/square ({lt.area_640:.1f}px@640) next to another light; check if only lit bulb was boxed instead of full housing",
                })
    return flags


def run_qc(data_dir: Path, cfg: dict[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    classes: list[str] = cfg["classes"]
    nc = len(classes)
    comp = cfg["competition"]
    qc_thr = cfg["qc_thresholds"]
    imgsz = int(comp["imgsz"])
    min_area_px = float(comp["min_gt_area_px"])
    dup_hamming_max = int(qc_thr["dup_hamming_max"])

    splits_map = discover_splits(data_dir)
    if not splits_map:
        raise FileNotFoundError(f"No images found under {data_dir}")

    per_class = {
        c: {
            "instances": 0,
            "images": 0,
            "tiny_lt_64px": 0,
            "gt_half_frame": 0,
            "area_buckets": {"lt_64": 0, "64_256": 0, "256_1024": 0, "gt_1024": 0},
            "by_split": defaultdict(int),
        }
        for c in classes
    }
    split_stats: dict[str, dict[str, int]] = {}
    by_split_ambiguity: dict[str, dict[str, Any]] = {}
    fatal: list[str] = []
    poly_issues: list[str] = []
    ambiguity_queue: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing_txt_files: list[str] = []

    for split_name, pairs in splits_map.items():
        n_inst = n_empty = 0
        split_ambig: list[dict[str, Any]] = []
        for img_path, lbl_path in pairs:
            rel_str = f"{split_name}/{img_path.name}"
            if lbl_path is None:
                missing_txt_files.append(rel_str)
            w, h = image_size(img_path)
            insts, f_errs, p_errs = parse_and_validate_label(
                lbl_path, w, h, nc, imgsz, f"{split_name}/{lbl_path.name}" if lbl_path else rel_str
            )
            fatal.extend(f_errs)
            poly_issues.extend(p_errs)
            seen_cls: set[str] = set()
            for it in insts:
                cname = classes[it.cls]
                per_class[cname]["instances"] += 1
                per_class[cname]["by_split"][split_name] += 1
                a640 = it.area_640
                if a640 < min_area_px:
                    per_class[cname]["tiny_lt_64px"] += 1
                    per_class[cname]["area_buckets"]["lt_64"] += 1
                elif a640 < 256.0:
                    per_class[cname]["area_buckets"]["64_256"] += 1
                elif a640 < 1024.0:
                    per_class[cname]["area_buckets"]["256_1024"] += 1
                else:
                    per_class[cname]["area_buckets"]["gt_1024"] += 1
                bx1, by1, bx2, by2 = it.bbox
                if (bx2 - bx1) * (by2 - by1) > 0.45:
                    per_class[cname]["gt_half_frame"] += 1
                seen_cls.add(cname)
            for cname in seen_cls:
                per_class[cname]["images"] += 1
            if not insts:
                n_empty += 1
            n_inst += len(insts)
            img_flags = detect_edge_case_flags(insts, qc_thr, rel_str)
            split_ambig.extend(img_flags)
            ambiguity_queue.extend(img_flags)
            records.append({
                "split": split_name,
                "name": img_path.name,
                "sha256": sha256_file(img_path),
                "dhash": dhash64(img_path),
            })
        split_stats[split_name] = {"images": len(pairs), "empty_images": n_empty, "instances": n_inst}
        by_split_ambiguity[split_name] = {
            "count": len(split_ambig),
            "flags": dict(Counter(item["flag"] for item in split_ambig)),
        }

    # Duplicate & cross-split leakage
    by_sha: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_sha[rec["sha256"]].append(idx)
    exact_groups = [[f"{records[i]['split']}/{records[i]['name']}" for i in grp] for grp in by_sha.values() if len(grp) > 1]
    near_raw = find_duplicate_pairs([r["dhash"] for r in records], dup_hamming_max)
    near_pairs = [
        (f"{records[i]['split']}/{records[i]['name']}", f"{records[j]['split']}/{records[j]['name']}", dist)
        for i, j, dist in near_raw
        if records[i]["sha256"] != records[j]["sha256"]
    ]
    leak_exact = [
        [f"{records[i]['split']}/{records[i]['name']}" for i in grp]
        for grp in by_sha.values()
        if len({records[i]["split"] for i in grp}) > 1
    ]
    leak_near = [
        (f"{records[i]['split']}/{records[i]['name']}", f"{records[j]['split']}/{records[j]['name']}", dist)
        for i, j, dist in near_raw
        if records[i]["split"] != records[j]["split"] and records[i]["sha256"] != records[j]["sha256"]
    ]

    # Convert nested defaultdicts to plain dicts for JSON serialization
    per_class_clean = {
        c: {
            "instances": v["instances"],
            "images": v["images"],
            "tiny_lt_64px": v["tiny_lt_64px"],
            "gt_half_frame": v["gt_half_frame"],
            "area_buckets": dict(v["area_buckets"]),
            "by_split": dict(v["by_split"]),
        }
        for c, v in per_class.items()
    }
    flag_counts = dict(Counter(item["flag"] for item in ambiguity_queue))
    report = {
        "data_dir": str(data_dir),
        "ok": len(fatal) == 0,
        "splits": split_stats,
        "per_class": per_class_clean,
        "fatal_count": len(fatal),
        "fatal": fatal,
        "poly_issue_count": len(poly_issues),
        "poly_issues": poly_issues,
        "missing_txt_files": len(missing_txt_files),
        "duplicates": {
            "exact_groups": exact_groups,
            "near_pairs": near_pairs,
            "leak_exact": leak_exact,
            "leak_near": leak_near,
        },
        "edge_case_flag_counts": flag_counts,
        "by_split_ambiguity": by_split_ambiguity,
        "ambiguity_queue_size": len(ambiguity_queue),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "ambiguity_queue.json").write_text(json.dumps(ambiguity_queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_qc_summary(report: dict[str, Any], cfg: dict[str, Any]) -> None:
    classes = cfg["classes"]
    vi_map = cfg.get("class_display_vi", {})
    print("=" * 78)
    print(f"QC TIER-1 REPORT: {report['data_dir']}")
    print("=" * 78)
    for sp, st in report["splits"].items():
        print(f"  {sp:>6}: {st['images']:>5} images | {st['instances']:>6} instances | {st['empty_images']:>4} empty")
    print("")
    print(f"  {'id':>2}  {'class':<14} {'ten tieng Viet':<22} {'inst':>6} {'imgs':>5} {'<64px@640':>10}")
    for cid, name in enumerate(classes):
        d = report["per_class"][name]
        vi = vi_map.get(name, name)[:22]
        print(f"  {cid:>2}  {name:<14} {vi:<22} {d['instances']:>6} {d['images']:>5} {d['tiny_lt_64px']:>10}")
    print("")
    print(f"  Fatal format errors : {report['fatal_count']}")
    print(f"  Polygon issues      : {report['poly_issue_count']}")
    print(f"  Missing .txt files  : {report['missing_txt_files']}")
    dup = report["duplicates"]
    print(f"  Duplicates (exact/near): {len(dup['exact_groups'])} / {len(dup['near_pairs'])} | Leakage: {len(dup['leak_exact'])} / {len(dup['leak_near'])}")
    print(f"  Ambiguity flags     : {report['ambiguity_queue_size']} -> {report['edge_case_flag_counts']}")
    for err in report["fatal"][:10]:
        print(f"    [FATAL] {err}")
    for iss in report["poly_issues"][:10]:
        print(f"    [POLY ] {iss}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QC Tier-1 validation and 5 edge-case detectors on a YOLO-seg dataset.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--config", type=Path, default=Path("configs/competition.yaml"), help="Path to competition.yaml")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for qc_report.json and ambiguity_queue.json")
    args = parser.parse_args()

    cfg = load_competition_config(args.config)
    out_dir = args.out_dir or (Path("work/qc") / args.data_dir.name)
    report = run_qc(args.data_dir, cfg, out_dir=out_dir)
    print_qc_summary(report, cfg)
    print(f"Saved JSON reports to: {out_dir}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
