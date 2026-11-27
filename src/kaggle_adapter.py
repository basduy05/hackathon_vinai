"""Universal Kaggle & HuggingFace Dataset Adapter for 11-class YOLO-seg.

Auto-detects and converts any mounted Kaggle dataset or HuggingFace Parquet repo into clean,
deduplicated 11-class YOLO-seg:
  - ``yolo_seg``: Remaps ``data.yaml`` names via ``kaggle_generic_aliases`` + Geometric Mask Merger
  - ``yolo_bbox``: Remaps ``data.yaml`` names -> SAM 2 (``box_to_mask_sam2``) -> Geometric Mask Merger
  - ``pascal_voc_xml``: Parses ``.xml`` bndboxes -> SAM 2 -> Geometric Mask Merger
  - ``coco_json``, ``idd_gtfine``, ``bdd100k_json``: Delegates to ``map_and_merge_public`` (or SAM 2 for BDD box2d)
  - ``hf_parquet``: Loads HuggingFace Parquet datasets (e.g. ``star092304/Traffic-sign-detection-VietNam``)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.box_to_mask_sam2 import BoxPredictor, UltralyticsSAM2Predictor, convert_bbox_dataset
from src.map_and_merge_public import apply_geometric_mask_merger, convert_public_source, load_class_mapping
from src.seg_utils import (
    SegInstance,
    collect_exclusion_hashes,
    is_near_duplicate,
    read_yolo_seg_file,
    register_dataset_usage,
    transfer_image,
    write_data_yaml,
    write_image,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, image_size, load_competition_config


def unzip_archives_in_dir(root_dir: Path, remove_zip: bool = False) -> list[Path]:
    """Safely extract any ``.zip`` archives inside ``root_dir`` with zip-slip protection."""
    extracted_dirs: list[Path] = []
    root_res = root_dir.resolve()
    for zp in sorted(root_dir.rglob("*.zip")):
        out_sub = zp.parent / zp.stem
        out_sub.mkdir(parents=True, exist_ok=True)
        out_res = out_sub.resolve()
        with zipfile.ZipFile(zp) as zf:
            for member in zf.infolist():
                if member.filename.startswith("__MACOSX") or "/." in member.filename:
                    continue
                target = (out_res / member.filename).resolve()
                if out_res != target and out_res not in target.parents:
                    raise ValueError(f"Unsafe zip member path in {zp}: {member.filename}")
                zf.extract(member, out_res)
        extracted_dirs.append(out_sub)
        if remove_zip and out_res.exists():
            zp.unlink(missing_ok=True)
    return extracted_dirs


def _parse_yaml_names(yaml_path: Path) -> dict[int, str]:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = obj.get("names", {})
    if isinstance(names, list):
        return {i: str(n).strip().lower() for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v).strip().lower() for k, v in names.items()}
    return {}


def _parse_classes_txt(txt_path: Path) -> dict[int, str]:
    lines = [ln.strip().lower() for ln in txt_path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return {i: name for i, name in enumerate(lines)}


def _find_classes_txt(anchor: Path, root: Path) -> Path | None:
    for c_root in (anchor, anchor.parent, root):
        for fname in ("classes_en.txt", "classes.txt", "obj.names"):
            cand = c_root / fname
            if cand.is_file():
                return cand
    found = sorted(p for p in root.rglob("classes*.txt")) + sorted(p for p in root.rglob("obj.names"))
    return found[0] if found else None


def _detect_yolo_label_kind(labels_dir: Path) -> str:
    for tf in sorted(labels_dir.rglob("*.txt"))[:30]:
        for line in tf.read_text(encoding="utf-8", errors="replace").splitlines():
            cols = line.split()
            if not cols:
                continue
            if len(cols) == 5:
                return "yolo_bbox"
            if len(cols) >= 7 and len(cols) % 2 == 1:
                return "yolo_seg"
    return "yolo_bbox"


def _find_matching_image_dir(anchor: Path, root: Path) -> Path | None:
    candidates = [
        anchor / "images" / "train",
        anchor / "train" / "images",
        anchor / "train2017",
        anchor / "images",
        anchor.parent / "train2017",
        anchor.parent / "images" / "train",
        anchor.parent / "images",
        root / "images" / "train",
        root / "train" / "images",
        root / "images",
    ]
    for c in candidates:
        if c.is_dir() and any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in c.rglob("*")):
            return c
    for d in sorted(root.rglob("images")):
        if d.is_dir() and any(p.is_file() and p.suffix.lower() in IMG_EXTS for p in d.rglob("*")):
            return d
    return None


def detect_dataset_format(dataset_dir: Path) -> dict[str, Any] | None:
    """Inspect a mounted Kaggle dataset directory and detect its annotation format & roots."""
    if not dataset_dir.is_dir():
        return None

    # 0. Mapillary Vistas v1.2 / v2.0 (polygons/*.json or instances/*.png + config_v*.json)
    for m_cfg in list(dataset_dir.rglob("config_v2.0.json")) + list(dataset_dir.rglob("config_v1.2.json")):
        m_root = m_cfg.parent
        tr_imgs = m_root / "training" / "images"
        lbl_cands = [
            m_root / "training" / "v2.0" / "polygons",
            m_root / "training" / "v1.2" / "polygons",
            m_root / "training" / "v2.0" / "instances",
            m_root / "training" / "v1.2" / "instances",
        ]
        tr_lbls = next((d for d in lbl_cands if d.is_dir() and any(d.iterdir())), None)
        if tr_imgs.is_dir() and tr_lbls is not None:
            val_imgs = m_root / "validation" / "images"
            val_cands = [
                m_root / "validation" / "v2.0" / "polygons",
                m_root / "validation" / "v1.2" / "polygons",
                m_root / "validation" / "v2.0" / "instances",
                m_root / "validation" / "v1.2" / "instances",
            ]
            val_lbls = next((d for d in val_cands if d.is_dir() and any(d.iterdir())), None)
            out: dict[str, Any] = {
                "format": "mapillary_vistas",
                "name": dataset_dir.name,
                "images_dir": str(tr_imgs),
                "labels_path": str(tr_lbls),
            }
            if val_imgs.is_dir() and val_lbls is not None:
                out["val_images_dir"] = str(val_imgs)
                out["val_labels_path"] = str(val_lbls)
            return out

    # 1. IDD / Cityscapes gtFine polygons
    for gtfine in sorted(dataset_dir.rglob("gtFine")):
        if gtfine.is_dir() and any(gtfine.rglob("*_polygons.json")):
            imgs = gtfine.parent / "leftImg8bit"
            if not imgs.is_dir():
                alt = [p for p in dataset_dir.rglob("leftImg8bit") if p.is_dir()]
                imgs = alt[0] if alt else (_find_matching_image_dir(gtfine, dataset_dir) or dataset_dir)
            if imgs.is_dir():
                is_city = any(k in dataset_dir.as_posix().lower() for k in ("cityscapes", "cityspaces"))
                fmt_name = "cityscapes_gtfine" if is_city else "idd_gtfine"
                return {"format": fmt_name, "name": dataset_dir.name, "images_dir": str(imgs), "labels_path": str(gtfine)}

    # 2. BDD100K JSON
    for jf in sorted(dataset_dir.rglob("*.json")):
        if "bdd100k" in str(jf).lower() and any(k in jf.name.lower() for k in ("train", "ins_seg", "labels")):
            imgs = _find_matching_image_dir(jf.parent, dataset_dir)
            for sub in ("images/100k/train", "100k/train", "images/train", "train/images"):
                cand = jf.parent.parent / sub
                if cand.is_dir():
                    imgs = cand
                    break
            if imgs and imgs.is_dir():
                return {"format": "bdd100k_json", "name": dataset_dir.name, "images_dir": str(imgs), "labels_path": str(jf)}

    # 3. COCO JSON (skip captions/keypoints; check BOTH file head and tail because COCO puts categories near EOF!)
    json_cands = sorted(
        (
            p for p in dataset_dir.rglob("*.json")
            if not p.name.startswith(("captions_", "person_keypoints_", "config_"))
        ),
        key=lambda p: (0 if "train" in p.name.lower() else 1, 0 if p.name.startswith("instances_") else 1, str(p)),
    )
    for jf in json_cands:
        if jf.name.startswith("instances_") or ("train" in jf.name.lower() and jf.stat().st_size > 1000):
            try:
                fsize = jf.stat().st_size
                with jf.open("rb") as fh:
                    head = fh.read(8192).decode("utf-8", errors="ignore")
                    tail = ""
                    if fsize > 8192:
                        fh.seek(max(0, fsize - 32768))
                        tail = fh.read(32768).decode("utf-8", errors="ignore")
            except OSError:
                continue
            blob = head + "\n" + tail
            if '"categories"' in blob and ('"annotations"' in blob or '"images"' in blob):
                imgs = _find_matching_image_dir(jf.parent, dataset_dir)
                if imgs and imgs.is_dir():
                    return {"format": "coco_json", "name": dataset_dir.name, "images_dir": str(imgs), "labels_path": str(jf)}

    # 4. YOLO (data.yaml + images/labels, or classes.txt)
    yamls = [p for p in sorted(dataset_dir.rglob("*.yaml")) if p.name in ("data.yaml", "dataset.yaml") or "yolo" in p.name.lower()]
    if yamls:
        y_path = yamls[0]
        lbl_cands = [
            y_path.parent / "train" / "labels",
            y_path.parent / "labels" / "train",
            y_path.parent / "labels",
        ]
        lbl_cands.extend(sorted(p for p in dataset_dir.rglob("labels") if p.is_dir()))
        lbl_dir = next((d for d in lbl_cands if d.is_dir() and any(d.rglob("*.txt"))), None)
        img_dir = _find_matching_image_dir(y_path.parent, dataset_dir)
        cls_txt = _find_classes_txt(y_path.parent, dataset_dir)
        if lbl_dir and img_dir:
            kind = _detect_yolo_label_kind(lbl_dir)
            return {
                "format": kind,
                "name": dataset_dir.name,
                "images_dir": str(img_dir),
                "labels_path": str(lbl_dir),
                "data_yaml": str(y_path),
                "classes_txt": str(cls_txt) if cls_txt else None,
            }

    # 5. YOLO images/ + labels/ with classes.txt / obj.names
    lbl_dirs = [p for p in sorted(dataset_dir.rglob("labels")) if p.is_dir() and any(p.rglob("*.txt"))]
    if lbl_dirs:
        lbl_dir = next((p for p in lbl_dirs if "train" in str(p).lower()), lbl_dirs[0])
        img_dir = _find_matching_image_dir(lbl_dir.parent, dataset_dir)
        cls_txt = _find_classes_txt(lbl_dir.parent, dataset_dir)
        if img_dir:
            kind = _detect_yolo_label_kind(lbl_dir)
            return {
                "format": kind,
                "name": dataset_dir.name,
                "images_dir": str(img_dir),
                "labels_path": str(lbl_dir),
                "data_yaml": None,
                "classes_txt": str(cls_txt) if cls_txt else None,
            }

    # 6. Pascal VOC XML
    xmls = list(dataset_dir.rglob("*.xml"))
    if xmls:
        img_dir = _find_matching_image_dir(xmls[0].parent, dataset_dir) or xmls[0].parent
        return {
            "format": "pascal_voc_xml",
            "name": dataset_dir.name,
            "images_dir": str(img_dir),
            "labels_path": str(xmls[0].parent),
        }

    return None


def _looks_like_dataset_root(d: Path) -> bool:
    markers = ("data.yaml", "dataset.yaml", "classes.txt", "annotations", "gtFine", "images", "train", "training", "Mapillary Vistas", "coco2017")
    return any((d / m).exists() for m in markers)


def discover_all_kaggle_inputs(input_root: Path = Path("/kaggle/input"), skip_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Scan ``/kaggle/input/*`` (unwrapping Kaggle's ``datasets/<owner>/<slug>`` mount folders)."""
    if not input_root.exists():
        return []
    skip = {s.lower() for s in (skip_names or set())}
    found: list[dict[str, Any]] = []
    seen_imgs: set[str] = set()

    def walk_mounts(node: Path, depth: int = 0) -> None:
        if not node.is_dir() or node.name.lower() in skip:
            return
        is_wrapper = node.name.lower() in {"datasets", "competitions", "models", "notebooks"}
        if depth > 0 and not is_wrapper and (_looks_like_dataset_root(node) or depth >= 3):
            info = detect_dataset_format(node)
            if info is not None:
                res_img = str(Path(info["images_dir"]).resolve())
                if res_img not in seen_imgs:
                    seen_imgs.add(res_img)
                    found.append(info)
                return
        if depth < 3:
            for child in sorted(node.iterdir()):
                if child.is_dir():
                    walk_mounts(child, depth + 1)

    walk_mounts(input_root, 0)
    return found


def _build_class_id_map(
    raw_names: dict[int, str],
    aliases: dict[str, int],
    override_map: dict[int, int] | None = None,
    sign_code_prefixes: tuple[str, ...] = ("dp.", "p.", "r.", "s.", "w."),
) -> dict[int, int]:
    if override_map is not None:
        return dict(override_map)
    out: dict[int, int] = {}
    for cid, cname in raw_names.items():
        key_raw = cname.strip().lower()
        key = key_raw.replace("_", " ")
        if key_raw in aliases:
            out[cid] = int(aliases[key_raw])
        elif key in aliases:
            out[cid] = int(aliases[key])
        elif any(key_raw.startswith(pref) for pref in sign_code_prefixes):
            out[cid] = 9  # Vietnamese road traffic sign codes -> Sign (9)
    return out


def _fast_list_images(images_dir: Path) -> list[Path]:
    """List image files using os.walk to minimize stat syscalls on FUSE mounts."""
    found: list[Path] = []
    for root, _, filenames in os.walk(images_dir):
        rpath = Path(root)
        for fn in filenames:
            if Path(fn).suffix.lower() in IMG_EXTS:
                found.append(rpath / fn)
    found.sort()
    return found


def _select_prioritized_yolo_images(
    all_imgs: list[Path],
    images_dir: Path,
    labels_dir: Path,
    cls_map: dict[int, int],
    default_class: int | None,
    max_images: int | None,
    prioritize_classes: tuple[int, ...],
    seed: int,
    source_name: str = "kaggle_yolo",
) -> list[Path]:
    if max_images is None or len(all_imgs) <= max_images:
        return all_imgs
    prio_set = set(prioritize_classes)
    prio_imgs: list[Path] = []
    rest_imgs: list[Path] = []
    # Shuffle first and bound label reads so 70k+ file FUSE mounts don't stall for tens of minutes.
    pool = list(all_imgs)
    rng = random.Random(seed)
    rng.shuffle(pool)
    scan_budget = min(len(pool), max(max_images * 6, 4000))
    t_scan = time.monotonic()

    for checked, im_p in enumerate(pool[:scan_budget], start=1):
        rel = im_p.relative_to(images_dir)
        lbl_p = labels_dir / rel.with_suffix(".txt")
        if not lbl_p.is_file():
            lbl_p = labels_dir / f"{im_p.stem}.txt"
        has_prio = False
        if lbl_p.is_file():
            for raw in lbl_p.read_text(encoding="utf-8", errors="replace").splitlines():
                cols = raw.split()
                if not cols:
                    continue
                tc = cls_map.get(int(float(cols[0])), default_class)
                if tc is not None and tc in prio_set:
                    has_prio = True
                    break
        (prio_imgs if has_prio else rest_imgs).append(im_p)
        if checked % 500 == 0 or len(prio_imgs) >= max_images:
            print(
                f"  [SCAN:{source_name}] đã kiểm {checked}/{len(all_imgs)} nhãn "
                f"(ưu tiên={len(prio_imgs)}/{max_images}, {time.monotonic() - t_scan:.1f}s)",
                flush=True,
            )
        if len(prio_imgs) >= max_images:
            break

    if len(prio_imgs) >= max_images:
        return sorted(prio_imgs[:max_images])
    rem = max_images - len(prio_imgs)
    return sorted(prio_imgs + rest_imgs[: min(rem, len(rest_imgs))])


def convert_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    fmt: str,
    data_yaml: Path | None = None,
    classes_txt: Path | None = None,
    predictor: BoxPredictor | None = None,
    split: str = "train",
    source_name: str = "kaggle_yolo",
    default_class: int | None = None,
    override_map: dict[int, int] | None = None,
    exclude_dir: Path | None = None,
    max_images: int | None = None,
    seed: int = 39,
    min_area_640: float = 32.0,
    prioritize_classes: tuple[int, ...] = (3, 6, 7, 9, 10, 1, 5, 4, 8),
    competition_path: Path = Path("configs/competition.yaml"),
    mapping_path: Path = Path("configs/class_mapping.yaml"),
    registry_path: Path | None = None,
    dataset_url: str = "",
    license_name: str = "Verify Kaggle license",
) -> dict[str, Any]:
    """Convert any Kaggle ``yolo_bbox`` or ``yolo_seg`` dataset to clean 11-class YOLO-seg."""
    comp_cfg = load_competition_config(competition_path)
    map_cfg = load_class_mapping(mapping_path)
    classes: list[str] = comp_cfg["classes"]
    imgsz = int(comp_cfg["competition"]["imgsz"])
    dup_hamming_max = int(comp_cfg["qc_thresholds"]["dup_hamming_max"])

    alias_cfg = map_cfg["sources"]["kaggle_generic_aliases"]
    aliases = {str(k).strip().lower(): int(v) for k, v in alias_cfg["direct_map"].items()}
    sign_prefixes = tuple(str(x).strip().lower() for x in alias_cfg.get("sign_code_prefixes", ["dp.", "p.", "r.", "s.", "w."]))
    rider_labels = {str(x).strip().lower() for x in alias_cfg.get("rider_labels", [])}
    merge_rules = map_cfg["geometric_merge_rules"]

    raw_names: dict[int, str] = {}
    if data_yaml and data_yaml.is_file():
        raw_names = _parse_yaml_names(data_yaml)
    if not raw_names and classes_txt and classes_txt.is_file():
        raw_names = _parse_classes_txt(classes_txt)
    if not raw_names:
        auto_cls = _find_classes_txt(labels_dir.parent, images_dir.parent)
        if auto_cls is not None:
            raw_names = _parse_classes_txt(auto_cls)

    cls_map = _build_class_id_map(raw_names, aliases, override_map=override_map, sign_code_prefixes=sign_prefixes)
    if not cls_map and default_class is None:
        raise ValueError(
            f"Cannot map classes for '{source_name}' (raw_names={raw_names}). "
            "Refusing silent {0:0..10:10} identity fallback to prevent label corruption."
        )

    print(f"  [YOLO:{source_name}/{split}] Đang liệt kê ảnh trong {images_dir}...", flush=True)
    all_imgs = _fast_list_images(images_dir)
    print(f"  [YOLO:{source_name}/{split}] Tìm thấy {len(all_imgs)} ảnh gốc (cap={max_images}).", flush=True)
    all_imgs = _select_prioritized_yolo_images(
        all_imgs, images_dir, labels_dir, cls_map, default_class, max_images, prioritize_classes, seed, source_name=source_name
    )

    seen_shas, seen_dhashes = collect_exclusion_hashes(exclude_dir)

    if fmt == "yolo_bbox":
        if predictor is None:
            raise ValueError("BoxPredictor (e.g. UltralyticsSAM2Predictor) is required to convert yolo_bbox datasets.")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            tmp_imgs, tmp_lbls = tmp_root / "images", tmp_root / "labels"
            tmp_imgs.mkdir(parents=True)
            tmp_lbls.mkdir(parents=True)
            kept = 0
            t_stage = time.monotonic()
            for idx, im_p in enumerate(all_imgs):
                is_dup, sha, dh = is_near_duplicate(im_p, seen_shas, seen_dhashes, max_hamming=dup_hamming_max)
                if is_dup:
                    continue
                rel = im_p.relative_to(images_dir)
                lbl_p = labels_dir / rel.with_suffix(".txt")
                if not lbl_p.is_file():
                    lbl_p = labels_dir / f"{im_p.stem}.txt"
                if not lbl_p.is_file():
                    continue
                remapped_lines: list[str] = []
                for raw in lbl_p.read_text(encoding="utf-8", errors="replace").splitlines():
                    cols = raw.split()
                    if len(cols) != 5:
                        continue
                    rc = int(float(cols[0]))
                    tc = cls_map.get(rc, default_class)
                    if tc is None:
                        continue
                    remapped_lines.append(f"{tc} {' '.join(cols[1:])}")
                if not remapped_lines:
                    continue
                seen_shas.add(sha)
                seen_dhashes.append(dh)
                stem = f"{idx:05d}_{im_p.stem}"
                transfer_image(im_p, tmp_imgs / f"{stem}{im_p.suffix.lower()}", mode="copy")
                (tmp_lbls / f"{stem}.txt").write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
                kept += 1
                if len(all_imgs) >= 100 and (idx + 1) % 250 == 0:
                    print(f"  [STAGE:{source_name}/{split}] Đã chuẩn bị {idx + 1}/{len(all_imgs)} ảnh ({time.monotonic() - t_stage:.1f}s)...", flush=True)

            print(f"  [SAM2:{source_name}/{split}] Bắt đầu tách mask trên {kept} ảnh đã chọn...", flush=True)
            summary = convert_bbox_dataset(
                images_dir=tmp_imgs,
                labels_dir=tmp_lbls,
                out_dir=out_dir,
                predictor=predictor,
                split=split,
                competition_path=competition_path,
                registry_path=None,
                source_name=source_name,
            )
            # Run Geometric Mask Merger if any converted files contain class 8 + 4/6
            out_lbl_dir = out_dir / "labels" / split
            merged_total = 0
            for lf in sorted(out_lbl_dir.glob("*.txt")):
                insts = read_yolo_seg_file(lf)
                if any(it.cls == 8 for it in insts) and any(it.cls in (4, 6) for it in insts):
                    tagged = [SegInstance(cls=it.cls, pts=it.pts, source_label="person" if it.cls == 8 else "") for it in insts]
                    merged, _, ms = apply_geometric_mask_merger(tagged, merge_rules, rider_labels, lf.stem)
                    merged_total += ms["auto_merged"]
                    write_yolo_seg_label(lf, [(it.cls, it.pts) for it in merged])
            summary["auto_merged_riders"] = merged_total

            if registry_path is not None:
                register_dataset_usage(
                    registry_path=registry_path,
                    name=source_name,
                    url=dataset_url or f"kaggle:{source_name}",
                    license_name=license_name,
                    raw_format="YOLO BBox -> SAM2 + Merger",
                    train_count=summary["images_processed"] if split == "train" else 0,
                    val_count=summary["images_processed"] if split == "val" else 0,
                    mapping_note=f"mapped={cls_map}, merged={merged_total}",
                    status="Converted",
                )
            return {"format": fmt, "source": source_name, **summary}

    # yolo_seg direct remap + polygon repair + Geometric Mask Merger
    out_img_dir = out_dir / "images" / split
    out_lbl_dir = out_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    kept_images = merged_total = 0
    class_counts: Counter[str] = Counter()
    for idx, im_p in enumerate(all_imgs):
        is_dup, sha, dh = is_near_duplicate(im_p, seen_shas, seen_dhashes, max_hamming=dup_hamming_max)
        if is_dup:
            continue
        rel = im_p.relative_to(images_dir)
        lbl_p = labels_dir / rel.with_suffix(".txt")
        if not lbl_p.is_file():
            lbl_p = labels_dir / f"{im_p.stem}.txt"
        raw_insts = read_yolo_seg_file(lbl_p)
        w, h = image_size(im_p)
        remapped: list[SegInstance] = []
        for it in raw_insts:
            tc = cls_map.get(it.cls, default_class)
            if tc is None:
                continue
            raw_name = raw_names.get(it.cls, "person" if tc == 8 else "")
            seg = SegInstance(cls=tc, pts=it.pts, source_label=raw_name)
            if seg.area_640(w, h, imgsz) >= min_area_640:
                remapped.append(seg)
        if not remapped:
            continue
        remapped, _, ms = apply_geometric_mask_merger(remapped, merge_rules, rider_labels, im_p.name)
        merged_total += ms["auto_merged"]

        seen_shas.add(sha)
        seen_dhashes.append(dh)
        stem = f"{source_name}_{idx:05d}_{im_p.stem}"
        transfer_image(im_p, out_img_dir / f"{stem}{im_p.suffix.lower()}", mode="copy")
        write_yolo_seg_label(out_lbl_dir / f"{stem}.txt", [(it.cls, it.pts) for it in remapped])
        kept_images += 1
        for it in remapped:
            class_counts[classes[it.cls]] += 1
        if len(all_imgs) >= 100 and (idx + 1) % 250 == 0:
            print(f"  [YOLO_SEG:{source_name}/{split}] {idx + 1}/{len(all_imgs)} ảnh | giữ={kept_images}", flush=True)

    write_data_yaml(out_dir, classes, has_val=(out_dir / "images" / "val").is_dir())
    if registry_path is not None:
        register_dataset_usage(
            registry_path=registry_path,
            name=source_name,
            url=dataset_url or f"kaggle:{source_name}",
            license_name=license_name,
            raw_format="YOLO Seg -> Remapped + Merger",
            train_count=kept_images if split == "train" else 0,
            val_count=kept_images if split == "val" else 0,
            mapping_note=f"mapped={cls_map}, merged={merged_total}",
            status="Converted",
        )
    return {
        "format": fmt,
        "source": source_name,
        "kept_images": kept_images,
        "auto_merged_riders": merged_total,
        "class_counts": dict(class_counts),
    }


def convert_pascal_voc_dataset(
    images_dir: Path,
    xml_dir: Path,
    out_dir: Path,
    predictor: BoxPredictor,
    split: str = "train",
    source_name: str = "kaggle_voc",
    max_images: int | None = None,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
    mapping_path: Path = Path("configs/class_mapping.yaml"),
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Convert Pascal VOC XML bounding box annotations to YOLO-seg using SAM 2."""
    map_cfg = load_class_mapping(mapping_path)
    aliases = {str(k).strip().lower(): int(v) for k, v in map_cfg["sources"]["kaggle_generic_aliases"]["direct_map"].items()}
    img_by_stem = {p.stem: p for p in sorted(images_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS}
    xml_files = sorted(xml_dir.rglob("*.xml"))
    if max_images is not None and len(xml_files) > max_images:
        xml_files = sorted(random.Random(seed).sample(xml_files, max_images))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_imgs, tmp_lbls = tmp_root / "images", tmp_root / "labels"
        tmp_imgs.mkdir(parents=True)
        tmp_lbls.mkdir(parents=True)
        for xf in xml_files:
            root_el = ET.parse(xf).getroot()
            fname = root_el.findtext("filename") or xf.stem
            im_p = img_by_stem.get(Path(fname).stem) or img_by_stem.get(xf.stem)
            if im_p is None:
                continue
            w, h = image_size(im_p)
            lines: list[str] = []
            for obj in root_el.findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                tc = aliases.get(name, aliases.get(name.replace("_", " ")))
                if tc is None:
                    continue
                bb = obj.find("bndbox")
                if bb is None:
                    continue
                x1 = float(bb.findtext("xmin") or 0)
                y1 = float(bb.findtext("ymin") or 0)
                x2 = float(bb.findtext("xmax") or 0)
                y2 = float(bb.findtext("ymax") or 0)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc, yc = 0.5 * (x1 + x2) / w, 0.5 * (y1 + y2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{tc} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                transfer_image(im_p, tmp_imgs / im_p.name, mode="copy")
                (tmp_lbls / f"{im_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

        return convert_bbox_dataset(
            images_dir=tmp_imgs,
            labels_dir=tmp_lbls,
            out_dir=out_dir,
            predictor=predictor,
            split=split,
            competition_path=competition_path,
            registry_path=registry_path,
            source_name=source_name,
        )


def export_hf_parquet_to_yolo_bbox(
    repo_id: str,
    out_raw_dir: Path,
    max_images: int = 2000,
    seed: int = 39,
    default_class: int = 9,
) -> dict[str, Any]:
    """Load a HuggingFace Parquet dataset (e.g. ``star092304/Traffic-sign-detection-VietNam``) into YOLO BBox files."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(repo_id)
    split_name = "train" if "train" in ds else next(iter(ds.keys()))
    dset = ds[split_name]
    if max_images and len(dset) > max_images:
        dset = dset.shuffle(seed=seed).select(range(max_images))

    img_dir = out_raw_dir / "images"
    lbl_dir = out_raw_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for idx, row in enumerate(dset):
        im = row.get("image")
        if im is None:
            continue
        w, h = im.size
        stem = f"hf_{idx:05d}"
        im_path = img_dir / f"{stem}.jpg"
        im.convert("RGB").save(im_path, quality=95)

        lines: list[str] = []
        # Support common HF detection schemas: objects={bbox:[x,y,w,h]|xyxy}, bboxes, or annotations
        objs = row.get("objects") or row.get("annotations") or {}
        if isinstance(objs, dict) and "bbox" in objs:
            bboxes = objs.get("bbox", [])
            cats = objs.get("category", objs.get("categories", [default_class] * len(bboxes)))
            for bb, cat in zip(bboxes, cats):
                if len(bb) != 4:
                    continue
                x, y, bw_or_x2, bh_or_y2 = [float(v) for v in bb]
                # Heuristic: COCO [x,y,w,h] vs normalized/pixel xyxy
                if max(x, y, bw_or_x2, bh_or_y2) <= 1.05:
                    xc, yc, bw, bh = x, y, bw_or_x2, bh_or_y2
                else:
                    xc = (x + 0.5 * bw_or_x2) / w
                    yc = (y + 0.5 * bh_or_y2) / h
                    bw = bw_or_x2 / w
                    bh = bh_or_y2 / h
                if 0 < bw <= 1.0 and 0 < bh <= 1.0:
                    lines.append(f"{default_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        elif isinstance(row.get("bboxes"), list):
            for bb in row["bboxes"]:
                if len(bb) == 4:
                    x1, y1, x2, y2 = [float(v) for v in bb]
                    if max(x1, y1, x2, y2) > 1.05:
                        xc, yc, bw, bh = 0.5 * (x1 + x2) / w, 0.5 * (y1 + y2) / h, (x2 - x1) / w, (y2 - y1) / h
                    else:
                        xc, yc, bw, bh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), (x2 - x1), (y2 - y1)
                    if bw > 0 and bh > 0:
                        lines.append(f"{default_class} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    return {"repo_id": repo_id, "split": split_name, "images_exported": written, "images_dir": str(img_dir), "labels_dir": str(lbl_dir)}


def auto_convert_detected_dataset(
    info: dict[str, Any],
    out_dir: Path,
    sam_predictor: BoxPredictor | None = None,
    exclude_dir: Path | None = None,
    max_images: int | None = None,
    seed: int = 39,
    split: str = "train",
    registry_path: Path | None = Path("datasets_registry.md"),
) -> dict[str, Any]:
    """Dispatch any detected dataset descriptor into the right converter."""
    fmt = info["format"]
    name = info["name"]
    imgs = Path(info["images_dir"])
    lbls = Path(info["labels_path"])
    ds_url = info.get("url", "")

    if fmt in ("yolo_seg", "yolo_bbox"):
        return convert_yolo_dataset(
            images_dir=imgs,
            labels_dir=lbls,
            out_dir=out_dir,
            fmt=fmt,
            data_yaml=Path(info["data_yaml"]) if info.get("data_yaml") else None,
            classes_txt=Path(info["classes_txt"]) if info.get("classes_txt") else None,
            predictor=sam_predictor,
            split=split,
            source_name=name,
            exclude_dir=exclude_dir,
            max_images=max_images,
            seed=seed,
            registry_path=registry_path,
            dataset_url=ds_url,
        )
    if fmt == "pascal_voc_xml":
        if sam_predictor is None:
            raise ValueError("BoxPredictor required for pascal_voc_xml")
        return convert_pascal_voc_dataset(
            images_dir=imgs, xml_dir=lbls, out_dir=out_dir, predictor=sam_predictor,
            split=split, source_name=name, max_images=max_images, seed=seed, registry_path=registry_path,
        )
    if fmt == "mapillary_vistas":
        return convert_public_source(
            "mapillary_vistas", "mapillary", imgs, lbls, out_dir, split=split, exclude_dir=exclude_dir,
            max_images=max_images, max_image_side=1280, seed=seed, registry_path=registry_path,
            dataset_url=ds_url or "https://www.mapillary.com/dataset/vistas", license_name="CC BY-NC-SA 4.0",
        )
    if fmt == "cityscapes_gtfine":
        return convert_public_source(
            "cityscapes", "idd", imgs, lbls, out_dir, split=split, exclude_dir=exclude_dir,
            max_images=max_images, seed=seed, registry_path=registry_path,
            dataset_url=ds_url or "https://www.cityscapes-dataset.com/", license_name="Academic / Non-Commercial",
        )
    if fmt == "idd_gtfine":
        return convert_public_source(
            "idd", "idd", imgs, lbls, out_dir, split=split, exclude_dir=exclude_dir,
            max_images=max_images, seed=seed, registry_path=registry_path,
            dataset_url=ds_url or "http://idd.insaan.iiit.ac.in/", license_name="Academic / Research",
        )
    if fmt == "coco_json":
        return convert_public_source(
            "coco", "coco", imgs, lbls, out_dir, split=split, exclude_dir=exclude_dir,
            max_images=max_images, seed=seed, registry_path=registry_path,
            dataset_url=ds_url or "https://cocodataset.org", license_name="CC BY 4.0",
        )
    if fmt == "bdd100k_json":
        raw = json.loads(lbls.read_text(encoding="utf-8"))
        frames = raw if isinstance(raw, list) else raw.get("frames", [])
        has_poly2d = any("poly2d" in lbl for it in frames[:50] for lbl in it.get("labels", []))
        if has_poly2d:
            return convert_public_source(
                "bdd100k", "bdd100k", imgs, lbls, out_dir, split=split, exclude_dir=exclude_dir,
                max_images=max_images, registry_path=registry_path,
                dataset_url="https://github.com/bdd100k/bdd100k", license_name="BDD100K License",
            )
        if sam_predictor is None:
            raise ValueError("BoxPredictor required for box2d BDD100K JSON")
        map_cfg = load_class_mapping(Path("configs/class_mapping.yaml"))
        bdd_map = {str(k).lower(): int(v) for k, v in map_cfg["sources"]["bdd100k"]["direct_map"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            t_imgs, t_lbls = tmp_root / "images", tmp_root / "labels"
            t_imgs.mkdir(parents=True)
            t_lbls.mkdir(parents=True)
            cap = max_images or 1500
            kept = 0
            for item in frames:
                if kept >= cap:
                    break
                im_p = imgs / Path(str(item.get("name", ""))).name
                if not im_p.is_file():
                    continue
                w, h = image_size(im_p)
                lines = []
                for lbl in item.get("labels", []):
                    cat = str(lbl.get("category", "")).strip().lower()
                    b2d = lbl.get("box2d")
                    if cat not in bdd_map or not b2d:
                        continue
                    xc, yc = 0.5 * (b2d["x1"] + b2d["x2"]) / w, 0.5 * (b2d["y1"] + b2d["y2"]) / h
                    bw, bh = (b2d["x2"] - b2d["x1"]) / w, (b2d["y2"] - b2d["y1"]) / h
                    if bw > 0 and bh > 0:
                        lines.append(f"{bdd_map[cat]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                if lines:
                    transfer_image(im_p, t_imgs / im_p.name, mode="copy")
                    (t_lbls / f"{im_p.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                    kept += 1
            res = convert_bbox_dataset(
                t_imgs, t_lbls, out_dir, predictor=sam_predictor, split=split,
                registry_path=registry_path, source_name="bdd100k_sam2",
                dataset_url="https://github.com/bdd100k/bdd100k", license_name="BDD100K License",
            )
            for lf in sorted((out_dir / "labels" / split).glob("*.txt")):
                insts = read_yolo_seg_file(lf)
                if any(it.cls == 8 for it in insts) and any(it.cls in (4, 6) for it in insts):
                    tagged = [SegInstance(cls=it.cls, pts=it.pts, source_label="rider" if it.cls == 8 else "") for it in insts]
                    merged, _, _ = apply_geometric_mask_merger(tagged, map_cfg["geometric_merge_rules"], {"rider"}, lf.stem)
                    write_yolo_seg_label(lf, [(it.cls, it.pts) for it in merged])
            return {"format": "bdd100k_box2d_sam2", **res}
    raise ValueError(f"Unsupported detected format: {fmt}")


def detect_val_split(info: dict[str, Any]) -> dict[str, Any] | None:
    """Given a train-split descriptor from ``detect_dataset_format``, find a matching clean ``val`` descriptor if present."""
    fmt = info["format"]
    imgs = Path(info["images_dir"])
    lbls = Path(info["labels_path"])

    if fmt == "mapillary_vistas" and info.get("val_images_dir") and info.get("val_labels_path"):
        return {**info, "images_dir": str(info["val_images_dir"]), "labels_path": str(info["val_labels_path"])}
    if fmt == "coco_json" and "train" in lbls.name.lower():
        val_json = lbls.parent / lbls.name.replace("train", "val")
        val_imgs = imgs.parent / imgs.name.replace("train", "val")
        if val_json.is_file() and val_imgs.is_dir():
            return {**info, "images_dir": str(val_imgs), "labels_path": str(val_json)}
    elif fmt in ("yolo_seg", "yolo_bbox"):
        for cand_v in ("val", "valid"):
            v_img = imgs.parent.parent / cand_v / "images" if imgs.name == "images" and imgs.parent.name == "train" else imgs.parent / cand_v
            v_lbl = lbls.parent.parent / cand_v / "labels" if lbls.name == "labels" and lbls.parent.name == "train" else lbls.parent / cand_v
            if v_img.is_dir() and v_lbl.is_dir() and any(v_lbl.rglob("*.txt")):
                return {**info, "images_dir": str(v_img), "labels_path": str(v_lbl)}
    elif fmt == "bdd100k_json" and "train" in lbls.name.lower():
        val_json = lbls.parent / lbls.name.replace("train", "val")
        val_img = imgs.parent / imgs.name.replace("train", "val")
        if val_json.is_file() and val_img.is_dir():
            return {**info, "images_dir": str(val_img), "labels_path": str(val_json)}
    elif fmt in ("idd_gtfine", "cityscapes_gtfine"):
        val_lbl = lbls / "val"
        val_img = imgs / "val"
        if val_lbl.is_dir() and val_img.is_dir():
            return {**info, "images_dir": str(val_img), "labels_path": str(val_lbl)}
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-detect and convert any Kaggle dataset to 11-class YOLO-seg.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root folder of Kaggle dataset (e.g. /kaggle/input/vehicle-dataset-for-yolo)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output staged YOLO-seg directory")
    parser.add_argument("--sam-model", default="sam2_b.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=39)
    args = parser.parse_args()

    info = detect_dataset_format(args.input_dir)
    if info is None:
        raise RuntimeError(f"Could not auto-detect supported format inside {args.input_dir}")
    sam = UltralyticsSAM2Predictor(args.sam_model, device=args.device) if info["format"] in ("yolo_bbox", "pascal_voc_xml") else None
    res = auto_convert_detected_dataset(info, args.out_dir, sam_predictor=sam, max_images=args.max_images, seed=args.seed)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
