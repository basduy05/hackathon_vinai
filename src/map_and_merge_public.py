"""Convert open-source public datasets to 11-class YOLO-seg with Geometric Mask Merger.

Supports:
  - ``coco``: Standard COCO instance segmentation JSON (COCO 2017 traffic subset, Roboflow COCO, etc.)
  - ``idd``: India Driving Dataset Cityscapes-style ``*_gtFine_polygons.json``
  - ``bdd100k``: BDD100K frame JSON labels (with optional ``--filter-weather`` / ``--filter-timeofday``)
  - ``mapillary``: Mapillary Vistas JSON polygon annotations (e.g. converted vistas/COCO-like dicts)

Applies ``configs/class_mapping.yaml`` and resolves the taxonomy mismatch where public datasets
annotate ``person`` / ``rider`` separately from ``motorcycle`` (4) / ``bicycle`` (6):
  - High-confidence rider+bike pairs (including multi-rider pillions) -> merged single ``5`` or ``7`` mask
  - Gray-zone pairs -> logged to ``ambiguity_queue.json`` for human audit
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.seg_utils import (
    SegInstance,
    collect_exclusion_hashes,
    is_near_duplicate,
    mask_to_polygon,
    merge_polygons_union,
    register_dataset_usage,
    transfer_image,
    write_data_yaml,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, bbox_iou, geom_iou, load_competition_config, make_shapely_polygon


def load_class_mapping(path: str | Path = "configs/class_mapping.yaml") -> dict[str, Any]:
    return load_competition_config(path)


def _decode_coco_segmentation(seg: Any, width: int, height: int) -> list[np.ndarray]:
    """Extract normalized ``(N, 2)`` polygons from COCO polygon list or uncompressed/compressed RLE."""
    if isinstance(seg, list):
        polys: list[np.ndarray] = []
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
            arr[:, 0] = np.clip(arr[:, 0] / float(width), 0.0, 1.0)
            arr[:, 1] = np.clip(arr[:, 1] / float(height), 0.0, 1.0)
            if len(np.unique(np.round(arr, 6), axis=0)) >= 3:
                polys.append(arr)
        return polys
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        try:
            from pycocotools import mask as mask_utils  # noqa: PLC0415

            rle = seg
            if isinstance(rle["counts"], list):
                rle = mask_utils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            bin_mask = mask_utils.decode(rle)
            poly = mask_to_polygon(bin_mask, width, height)
            return [poly] if poly is not None else []
        except ImportError:
            return []
    return []


def _to_seg_poly(pts_list: list[np.ndarray], cls_id: int, source_label: str) -> SegInstance | None:
    if not pts_list:
        return None
    if len(pts_list) == 1:
        return SegInstance(cls=cls_id, pts=pts_list[0], source_label=source_label)
    merged = merge_polygons_union(pts_list)
    if merged is None:
        return None
    return SegInstance(cls=cls_id, pts=merged, source_label=source_label)


def apply_geometric_mask_merger(
    instances: list[SegInstance],
    merge_rules: dict[str, Any],
    rider_labels: set[str],
    image_name: str,
) -> tuple[list[SegInstance], list[dict[str, Any]], dict[str, int]]:
    """Merge ``rider/person`` + ``motorcycle/bicycle`` into single ``Motorcyclist (5)`` / ``Cyclist (7)`` masks.

    Supports multiple riders on a single two-wheeler (e.g. parent + pillion + child in Vietnam traffic)
    by grouping all aligned riders onto their matched vehicle and performing one Shapely union.
    """
    rule_specs = [
        merge_rules["motorcycle_to_motorcyclist"],
        merge_rules["bicycle_to_cyclist"],
    ]
    vehicle_to_rule = {int(r["vehicle_class"]): r for r in rule_specs}

    rider_indices = [
        i for i, inst in enumerate(instances)
        if inst.cls == 8 and (not rider_labels or inst.source_label.lower() in rider_labels)
    ]
    bike_indices = [i for i, inst in enumerate(instances) if inst.cls in vehicle_to_rule]

    if not rider_indices or not bike_indices:
        return instances, [], {"auto_merged": 0, "ambiguous": 0}

    # Assign each candidate rider to at most one best matching two-wheeler
    bike_groups: dict[int, list[int]] = defaultdict(list)
    consumed_riders: set[int] = set()
    ambiguity_items: list[dict[str, Any]] = []

    for ri in rider_indices:
        r_inst = instances[ri]
        rx1, ry1, rx2, ry2 = r_inst.bbox
        rcx, rcy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
        r_area = r_inst.geom.area if r_inst.geom is not None else max(1e-8, (rx2 - rx1) * (ry2 - ry1))

        best_bi: int | None = None
        best_score = -1.0
        best_status = "none"
        best_metrics: dict[str, float] = {}

        for bi in bike_indices:
            b_inst = instances[bi]
            rule = vehicle_to_rule[b_inst.cls]
            bx1, by1, bx2, by2 = b_inst.bbox
            bw = max(1e-6, bx2 - bx1)
            bh = max(1e-6, by2 - by1)
            bcx = 0.5 * (bx1 + bx2)

            if r_inst.geom is not None and b_inst.geom is not None:
                inter_area = float(r_inst.geom.intersection(b_inst.geom).area)
                union_area = float(r_inst.geom.area + b_inst.geom.area - inter_area)
                iou = inter_area / union_area if union_area > 0 else 0.0
            else:
                iou = bbox_iou(r_inst.bbox, b_inst.bbox)
                ix1, iy1 = max(rx1, bx1), max(ry1, by1)
                ix2, iy2 = min(rx2, bx2), min(ry2, by2)
                inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

            biou = bbox_iou(r_inst.bbox, b_inst.bbox)
            person_overlap = inter_area / max(1e-8, r_area)
            dx_ratio = abs(rcx - bcx) / bw
            above_or_inside = (ry2 >= by1 - 0.15 * bh) and (rcy <= by2 + 0.05 * bh)
            aligned = (dx_ratio <= float(rule["max_center_dx_ratio"])) and (
                above_or_inside or not bool(rule.get("require_rider_above_or_inside", True))
            )
            strict_dx = dx_ratio <= float(rule.get("auto_merge_max_dx_ratio", 0.60))

            auto_hit = aligned and (
                iou >= float(rule["auto_merge_min_iou"])
                or person_overlap >= float(rule["auto_merge_min_person_overlap"])
                or (strict_dx and biou >= float(rule.get("auto_merge_min_bbox_iou", 0.08)))
            )
            ambig_hit = (not auto_hit) and aligned and (
                person_overlap >= float(rule["ambiguity_min_person_overlap"]) or biou >= 0.05
            )
            score = iou + person_overlap + 0.5 * biou
            if auto_hit and score > best_score:
                best_score = score
                best_bi = bi
                best_status = "auto"
                best_metrics = {"iou": iou, "person_overlap": person_overlap, "bbox_iou": biou, "dx_ratio": dx_ratio}
            elif ambig_hit and best_status != "auto" and score > best_score:
                best_score = score
                best_bi = bi
                best_status = "ambiguous"
                best_metrics = {"iou": iou, "person_overlap": person_overlap, "bbox_iou": biou, "dx_ratio": dx_ratio}

        if best_bi is not None and best_status == "auto":
            bike_groups[best_bi].append(ri)
            consumed_riders.add(ri)
        elif best_bi is not None and best_status == "ambiguous":
            ambiguity_items.append({
                "flag": "RIDER_MERGE_GRAYZONE",
                "image": image_name,
                "rider_idx": ri,
                "vehicle_idx": best_bi,
                "vehicle_cls": instances[best_bi].cls,
                "metrics": {k: round(v, 4) for k, v in best_metrics.items()},
                "detail": "Borderline person-over-2wheeler overlap; verify manually whether Motorcyclist(5)/Cyclist(7) or Pedestrian(8)+Vehicle",
            })

    out: list[SegInstance] = []
    merged_bikes: set[int] = set()
    auto_merged_count = 0

    for bi, r_list in bike_groups.items():
        b_inst = instances[bi]
        merged_cls = int(vehicle_to_rule[b_inst.cls]["merged_class"])
        merged_poly = merge_polygons_union([b_inst.pts] + [instances[ri].pts for ri in r_list])
        if merged_poly is not None:
            out.append(SegInstance(cls=merged_cls, pts=merged_poly, source_label=f"{b_inst.source_label}+rider"))
            merged_bikes.add(bi)
            auto_merged_count += 1
        else:
            # Fallback: keep unmerged if union failed
            for ri in r_list:
                consumed_riders.discard(ri)

    for idx, inst in enumerate(instances):
        if idx in consumed_riders or idx in merged_bikes:
            continue
        out.append(inst)

    return out, ambiguity_items, {"auto_merged": auto_merged_count, "ambiguous": len(ambiguity_items)}


def _parse_coco_dataset(
    ann_json: Path,
    images_root: Path,
    direct_map: dict[str, int],
    max_images: int | None = None,
    prioritize_classes: tuple[int, ...] = (3, 6, 7, 9, 10, 4, 1, 2),
    seed: int = 39,
) -> list[dict[str, Any]]:
    t0 = time.monotonic()
    print(f"  [COCO] Đang nạp {ann_json.name}...", flush=True)
    data = json.loads(ann_json.read_text(encoding="utf-8"))
    cats = {int(c["id"]): str(c["name"]).strip().lower() for c in data.get("categories", [])}
    cat_to_cls = {cid: int(direct_map[nm]) for cid, nm in cats.items() if nm in direct_map}

    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    img_cls_set: dict[int, set[int]] = defaultdict(set)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        cid = int(ann["category_id"])
        if cid not in cat_to_cls:
            continue
        iid = int(ann["image_id"])
        anns_by_img[iid].append(ann)
        img_cls_set[iid].add(cat_to_cls[cid])

    im_candidates = [im for im in data.get("images", []) if int(im["id"]) in anns_by_img]
    # Pre-select prioritized COCO images on lightweight metadata before decoding 800k+ polygons.
    if max_images is not None and len(im_candidates) > max_images:
        prio_set = set(prioritize_classes)
        prio_ims, other_ims = [], []
        for im in im_candidates:
            clss = img_cls_set[int(im["id"])]
            has_rider_pair = 8 in clss and bool(clss & {4, 6})
            (prio_ims if ((clss & prio_set) or has_rider_pair) else other_ims).append(im)
        rng = random.Random(seed)
        rng.shuffle(prio_ims)
        rng.shuffle(other_ims)
        candidate_cap = max(int(max_images * 3), int(max_images) + 1000)
        im_candidates = (prio_ims + other_ims)[:candidate_cap]
    print(
        f"  [COCO] Lọc nhanh còn {len(im_candidates)} ảnh ứng viên ({time.monotonic() - t0:.1f}s); đang giải mã polygon...",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    total_cands = len(im_candidates)
    for idx, im in enumerate(im_candidates, start=1):
        img_id = int(im["id"])
        file_name = str(im["file_name"])
        img_path = images_root / file_name
        if not img_path.is_file():
            img_path = images_root / Path(file_name).name
        if not img_path.is_file():
            continue
        w, h = int(im["width"]), int(im["height"])
        insts: list[SegInstance] = []
        for ann in anns_by_img.get(img_id, []):
            raw_name = cats.get(int(ann["category_id"]), "")
            if raw_name not in direct_map:
                continue
            polys = _decode_coco_segmentation(ann.get("segmentation"), w, h)
            seg_inst = _to_seg_poly(polys, int(direct_map[raw_name]), raw_name)
            if seg_inst is not None:
                insts.append(seg_inst)
        records.append({"image_path": img_path, "width": w, "height": h, "instances": insts})
        if total_cands >= 500 and (idx % 500 == 0 or idx == total_cands):
            print(f"  [COCO] Đã giải mã {idx}/{total_cands} ảnh ({time.monotonic() - t0:.1f}s)", flush=True)
    return records


def _parse_idd_polygons_dataset(
    labels_root: Path,
    images_root: Path,
    direct_map: dict[str, int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    img_by_rel: dict[tuple[str, str], Path] = {}
    img_by_stem: dict[str, Path] = {}
    for p in sorted(images_root.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        base = p.stem.replace("_leftImg8bit", "")
        rel_parent = p.relative_to(images_root).parent.as_posix()
        img_by_rel[(rel_parent, base)] = p
        img_by_stem.setdefault(base, p)

    for jf in sorted(labels_root.rglob("*_polygons.json")):
        stem = jf.stem.replace("_gtFine_polygons", "").replace("_gtCoarse_polygons", "")
        rel_parent = jf.relative_to(labels_root).parent.as_posix()
        img_path = img_by_rel.get((rel_parent, stem))
        if img_path is None:
            # Also handle when one side omits/includes train/val prefix or folder-only match
            folder_match = [p for (par, b), p in img_by_rel.items() if b == stem and Path(par).name == Path(rel_parent).name]
            img_path = folder_match[0] if folder_match else img_by_stem.get(stem)
        if img_path is None:
            continue
        obj = json.loads(jf.read_text(encoding="utf-8"))
        w, h = int(obj["imgWidth"]), int(obj["imgHeight"])
        insts: list[SegInstance] = []
        for item in obj.get("objects", []):
            raw_name = str(item.get("label", "")).strip().lower()
            if raw_name not in direct_map:
                continue
            poly = np.asarray(item.get("polygon", []), dtype=np.float64)
            if poly.ndim != 2 or len(poly) < 3 or poly.shape[1] != 2:
                continue
            poly[:, 0] = np.clip(poly[:, 0] / float(w), 0.0, 1.0)
            poly[:, 1] = np.clip(poly[:, 1] / float(h), 0.0, 1.0)
            if len(np.unique(np.round(poly, 6), axis=0)) >= 3:
                insts.append(SegInstance(cls=int(direct_map[raw_name]), pts=poly, source_label=raw_name))
        records.append({"image_path": img_path, "width": w, "height": h, "instances": insts})
    return records


def _parse_bdd100k_dataset(
    labels_json: Path,
    images_root: Path,
    direct_map: dict[str, int],
    weather_filter: set[str] | None = None,
    time_filter: set[str] | None = None,
) -> list[dict[str, Any]]:
    raw = json.loads(labels_json.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("frames", [])
    records: list[dict[str, Any]] = []
    for item in items:
        attrs = item.get("attributes", {}) or {}
        if weather_filter and str(attrs.get("weather", "")).lower() not in weather_filter:
            continue
        if time_filter and str(attrs.get("timeofday", "")).lower() not in time_filter:
            continue
        name = str(item.get("name", ""))
        img_path = images_root / name
        if not img_path.is_file():
            img_path = images_root / Path(name).name
        if not img_path.is_file():
            continue
        from PIL import Image  # noqa: PLC0415

        with Image.open(img_path) as im:
            w, h = im.size
        insts: list[SegInstance] = []
        for lbl in item.get("labels", []):
            raw_name = str(lbl.get("category", "")).strip().lower()
            if raw_name not in direct_map:
                continue
            poly2d = lbl.get("poly2d") or []
            polys: list[np.ndarray] = []
            for p_entry in poly2d:
                verts = p_entry.get("vertices", []) if isinstance(p_entry, dict) else []
                if len(verts) < 3:
                    continue
                arr = np.asarray(verts, dtype=np.float64)
                arr[:, 0] = np.clip(arr[:, 0] / float(w), 0.0, 1.0)
                arr[:, 1] = np.clip(arr[:, 1] / float(h), 0.0, 1.0)
                if len(np.unique(np.round(arr, 6), axis=0)) >= 3:
                    polys.append(arr)
            seg_inst = _to_seg_poly(polys, int(direct_map[raw_name]), raw_name)
            if seg_inst is not None:
                insts.append(seg_inst)
        records.append({"image_path": img_path, "width": w, "height": h, "instances": insts})
    return records


def convert_public_source(
    source_name: str,
    fmt: str,
    images_dir: Path,
    labels_path: Path,
    out_dir: Path,
    split: str = "train",
    mapping_path: Path = Path("configs/class_mapping.yaml"),
    competition_path: Path = Path("configs/competition.yaml"),
    exclude_dir: Path | None = None,
    min_area_640: float = 32.0,
    keep_empty: bool = False,
    max_images: int | None = None,
    transfer_mode: str = "copy",
    weather_filter: set[str] | None = None,
    time_filter: set[str] | None = None,
    prioritize_classes: tuple[int, ...] = (3, 6, 7, 9, 10, 4, 1, 2),
    seed: int = 39,
    registry_path: Path | None = None,
    dataset_url: str = "",
    license_name: str = "",
) -> dict[str, Any]:
    """Convert and merge a public dataset into standardized 11-class YOLO-seg split layout."""
    comp_cfg = load_competition_config(competition_path)
    map_cfg = load_class_mapping(mapping_path)
    classes: list[str] = comp_cfg["classes"]
    imgsz = int(comp_cfg["competition"]["imgsz"])
    dup_hamming_max = int(comp_cfg["qc_thresholds"]["dup_hamming_max"])

    if source_name not in map_cfg["sources"]:
        raise ValueError(f"Unknown source {source_name!r}; expected one of {sorted(map_cfg['sources'])}")
    src_cfg = map_cfg["sources"][source_name]
    direct_map = {str(k).strip().lower(): int(v) for k, v in src_cfg["direct_map"].items()}
    needs_merge = bool(src_cfg.get("needs_rider_merge", False))
    rider_labels = {str(x).strip().lower() for x in src_cfg.get("rider_labels", [])}
    merge_rules = map_cfg["geometric_merge_rules"]

    if fmt == "coco":
        records = _parse_coco_dataset(
            labels_path, images_dir, direct_map,
            max_images=max_images, prioritize_classes=prioritize_classes, seed=seed,
        )
    elif fmt == "idd":
        records = _parse_idd_polygons_dataset(labels_path, images_dir, direct_map)
    elif fmt == "bdd100k":
        records = _parse_bdd100k_dataset(labels_path, images_dir, direct_map, weather_filter, time_filter)
    else:
        raise ValueError(f"Unsupported format {fmt!r} (supported: coco, idd, bdd100k)")

    if max_images is not None and len(records) > max_images:
        prio_set = set(prioritize_classes)
        prio_recs, other_recs = [], []
        for rec in records:
            cls_in_rec = {it.cls for it in rec["instances"]}
            has_rider_pair = 8 in cls_in_rec and bool(cls_in_rec & {4, 6})
            if (cls_in_rec & prio_set) or has_rider_pair:
                prio_recs.append(rec)
            else:
                other_recs.append(rec)
        rng = random.Random(seed)
        rng.shuffle(prio_recs)
        rng.shuffle(other_recs)
        records = prio_recs + other_recs

    seen_shas, seen_dhashes = collect_exclusion_hashes(exclude_dir)
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept_images = skipped_empty = skipped_dup = 0
    total_auto_merged = 0
    ambiguity_queue: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    for rec in records:
        if max_images is not None and kept_images >= max_images:
            break
        img_path: Path = rec["image_path"]
        w, h = int(rec["width"]), int(rec["height"])
        insts: list[SegInstance] = [
            it for it in rec["instances"] if it.area_640(w, h, imgsz) >= min_area_640
        ]
        if needs_merge:
            insts, ambig, mstats = apply_geometric_mask_merger(insts, merge_rules, rider_labels, img_path.name)
            total_auto_merged += mstats["auto_merged"]
            ambiguity_queue.extend(ambig)
        if not insts and not keep_empty:
            skipped_empty += 1
            continue

        is_dup, sha, dh = is_near_duplicate(img_path, seen_shas, seen_dhashes, max_hamming=dup_hamming_max)
        if is_dup:
            skipped_dup += 1
            continue
        seen_shas.add(sha)
        seen_dhashes.append(dh)

        try:
            rel_parts = img_path.relative_to(images_dir).with_suffix("").parts
            rel_stem = "_".join(rel_parts)
        except ValueError:
            rel_stem = img_path.stem
        out_stem = f"{source_name}_{rel_stem}"
        dst_img = out_img_dir / f"{out_stem}{img_path.suffix.lower()}"
        dst_lbl = out_lbl_dir / f"{out_stem}.txt"
        transfer_image(img_path, dst_img, mode=transfer_mode)
        write_yolo_seg_label(dst_lbl, [(it.cls, it.pts) for it in insts])

        kept_images += 1
        for it in insts:
            class_counts[classes[it.cls]] += 1

    write_data_yaml(out_dir, classes, has_val=(out_dir / "images" / "val").is_dir())
    ambig_path = out_dir / f"ambiguity_queue_{source_name}_{split}.json"
    ambig_path.write_text(json.dumps(ambiguity_queue, indent=2, ensure_ascii=False), encoding="utf-8")

    if registry_path is not None:
        register_dataset_usage(
            registry_path=registry_path,
            name=source_name,
            url=dataset_url or "(local/mirror)",
            license_name=license_name or "Verify upstream license",
            raw_format=fmt,
            train_count=kept_images if split == "train" else 0,
            val_count=kept_images if split == "val" else 0,
            mapping_note=f"configs/class_mapping.yaml#{source_name} (merged={total_auto_merged})",
            status="Converted",
        )

    return {
        "source": source_name,
        "format": fmt,
        "split": split,
        "records_parsed": len(records),
        "kept_images": kept_images,
        "skipped_empty": skipped_empty,
        "skipped_duplicates_or_leak": skipped_dup,
        "auto_merged_riders": total_auto_merged,
        "ambiguity_flagged": len(ambiguity_queue),
        "ambiguity_queue_path": str(ambig_path),
        "class_counts": dict(class_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert and merge public datasets to 11-class YOLO-seg.")
    parser.add_argument("--source", required=True, help="Key in configs/class_mapping.yaml (e.g. mapillary_vistas, idd, bdd100k, coco)")
    parser.add_argument("--format", required=True, choices=["coco", "idd", "bdd100k"], help="Annotation format")
    parser.add_argument("--images-dir", type=Path, required=True, help="Root folder containing source images")
    parser.add_argument("--labels", type=Path, required=True, help="JSON annotation file (coco/bdd100k) or polygons dir (idd)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination YOLO-seg dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Target split")
    parser.add_argument("--exclude-dir", type=Path, default=Path("data/golden_val"), help="Exclude duplicates/leakage against this dir")
    parser.add_argument("--min-area-640", type=float, default=32.0, help="Drop tiny fragments below this px^2 at 640 scale")
    parser.add_argument("--keep-empty", action="store_true", help="Keep background images with 0 instances")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on converted images")
    parser.add_argument("--transfer-mode", default="copy", choices=["copy", "hardlink", "symlink"])
    parser.add_argument("--filter-weather", default="", help="Comma-separated BDD100K weather values (e.g. rainy,foggy)")
    parser.add_argument("--filter-timeofday", default="", help="Comma-separated BDD100K timeofday values (e.g. night,dawn/dusk)")
    parser.add_argument("--registry", type=Path, default=Path("datasets_registry.md"), help="Registry markdown to update")
    parser.add_argument("--url", default="", help="Dataset source URL for registry")
    parser.add_argument("--license", default="", help="Dataset license for registry")
    args = parser.parse_args()

    weather_set = {x.strip().lower() for x in args.filter_weather.split(",") if x.strip()} or None
    time_set = {x.strip().lower() for x in args.filter_timeofday.split(",") if x.strip()} or None
    summary = convert_public_source(
        source_name=args.source,
        fmt=args.format,
        images_dir=args.images_dir,
        labels_path=args.labels,
        out_dir=args.out_dir,
        split=args.split,
        exclude_dir=args.exclude_dir if args.exclude_dir.exists() else None,
        min_area_640=args.min_area_640,
        keep_empty=args.keep_empty,
        max_images=args.max_images,
        transfer_mode=args.transfer_mode,
        weather_filter=weather_set,
        time_filter=time_set,
        registry_path=args.registry,
        dataset_url=args.url,
        license_name=args.license,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
