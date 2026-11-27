"""Golden Validation evaluator (using official ``detscore_selfcheck``) and Tier-2 Confident Learning QC.

Supports two modes:
  1. ``eval``: Run ``submission.zip`` (or ``model.onnx``) through ``detscore_selfcheck`` on ``data/golden_val/``,
     compute exact ``100 * mask mAP@[.5:.95]``, per-class AP, confusion matrix, and worst-performing images.
  2. ``qc-train``: Run Confident Learning disagreement mining on a training split to flag missing labels,
     wrong-class labels (e.g. Car/Bus, Truck/Other vehicle, Motorcycle/Motorcyclist), and loose masks.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from src.seg_utils import rasterize_norm_polygon, read_yolo_seg_file
from src.selfcheck_bridge import DEFAULT_NOTEBOOK, DEFAULT_WORK_SELFCHECK, materialize_selfcheck
from src.validate_dataset import IMG_EXTS, bbox_iou, image_size, load_competition_config


def _decode_rle_to_mask(rle_str: str, height: int, width: int, rle_mod: Any) -> np.ndarray:
    """Decode COCO RLE string reusing ``detscore_selfcheck.rle`` (no pycocotools required)."""
    counts = rle_mod.validate_rle_string(rle_str, height, width)
    return rle_mod.counts_to_mask(counts, height, width).astype(np.uint8)


def _mask_iou_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
    return float(inter / union) if union > 0 else 0.0


def _mask_bbox(mask: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()) / width, float(ys.min()) / height, float(xs.max() + 1) / width, float(ys.max() + 1) / height)


def _fallback_score_split(
    pred_json: dict[str, Any],
    labels_dir: Path,
    image_paths: list[Path],
    classes: list[str],
    imgsz: int = 640,
    min_area_px: float = 64.0,
    max_det: int = 100,
    rle_mod: Any = None,
) -> dict[str, Any]:
    """Pure-NumPy/OpenCV fallback scorer matching ``detscore`` logic when ``pycocotools`` is unavailable."""
    thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)
    recall_thrs = np.linspace(0.0, 1.0, 101)
    by_stem = {Path(im["file"]).stem: im for im in pred_json.get("images", [])}

    class_records: dict[int, list[tuple[float, np.ndarray, np.ndarray]]] = {c: [] for c in range(len(classes))}
    n_gt_total: dict[str, int] = {c: 0 for c in classes}

    for img_p in image_paths:
        w, h = image_size(img_p)
        scale_640 = (imgsz / max(h, w)) ** 2
        gt_insts = read_yolo_seg_file(labels_dir / f"{img_p.stem}.txt")
        gt_masks = [rasterize_norm_polygon(it.pts, w, h) for it in gt_insts]
        gt_areas = [float(m.sum()) * scale_640 for m in gt_masks]

        pred_im = by_stem.get(img_p.stem, {"instances": []})
        pred_insts = pred_im.get("instances", [])

        for c in range(len(classes)):
            g_idx = [i for i, it in enumerate(gt_insts) if it.cls == c]
            d_list = [it for it in pred_insts if int(it["c"]) == c]
            d_list.sort(key=lambda it: float(it["s"]), reverse=True)
            d_list = d_list[:max_det]

            g_ign = np.array([gt_areas[i] < min_area_px for i in g_idx], dtype=bool)
            order_g = np.argsort(g_ign, kind="stable")
            g_idx = [g_idx[i] for i in order_g]
            g_ign = g_ign[order_g]
            n_gt_total[classes[c]] += int((~g_ign).sum())

            if not d_list:
                continue

            dt_masks = [_decode_rle_to_mask(it["rle"], h, w, rle_mod) for it in d_list]
            dt_scores = np.array([float(it["s"]) for it in d_list], dtype=np.float64)
            dt_small = np.array([float(m.sum()) * scale_640 < min_area_px for m in dt_masks], dtype=bool)

            t_len, d_len, g_len = len(thresholds), len(d_list), len(g_idx)
            dtm = np.zeros((t_len, d_len), dtype=bool)
            dtig = np.zeros((t_len, d_len), dtype=bool)

            if g_len:
                ious = np.zeros((d_len, g_len), dtype=np.float64)
                for di, dm in enumerate(dt_masks):
                    for gi_pos, gi in enumerate(g_idx):
                        ious[di, gi_pos] = _mask_iou_pair(dm, gt_masks[gi])
                matched = np.zeros((t_len, g_len), dtype=bool)
                reg = ~g_ign
                for di in range(d_len):
                    row = ious[di]
                    cand = (row[None, :] >= thresholds[:, None]) & ~matched
                    if not cand.any():
                        continue
                    cand_reg = cand & reg[None, :]
                    use = np.where(cand_reg.any(axis=1, keepdims=True), cand_reg, cand)
                    hit = use.any(axis=1)
                    vals = np.where(use, row[None, :], -1.0)
                    best = g_len - 1 - np.argmax(vals[:, ::-1], axis=1)
                    rows = np.flatnonzero(hit)
                    matched[rows, best[rows]] = True
                    dtm[rows, di] = True
                    dtig[rows, di] = g_ign[best[rows]]
            if dt_small.any():
                dtig |= (~dtm) & dt_small[None, :]
            for di in range(d_len):
                class_records[c].append((float(dt_scores[di]), dtm[:, di], dtig[:, di]))

    per_class_ap: dict[str, float] = {}
    per_class_ap50: dict[str, float] = {}
    for c, cname in enumerate(classes):
        npig = n_gt_total[cname]
        if npig == 0:
            continue
        recs = class_records[c]
        if not recs:
            per_class_ap[cname] = 0.0
            per_class_ap50[cname] = 0.0
            continue
        recs.sort(key=lambda r: r[0], reverse=True)
        dtm_all = np.stack([r[1] for r in recs], axis=1)
        dtig_all = np.stack([r[2] for r in recs], axis=1)
        tps = np.cumsum(dtm_all & ~dtig_all, axis=1).astype(np.float64)
        fps = np.cumsum(~dtm_all & ~dtig_all, axis=1).astype(np.float64)
        aps = []
        for ti in range(len(thresholds)):
            rc = tps[ti] / npig
            pr = tps[ti] / (fps[ti] + tps[ti] + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            pos = np.searchsorted(rc, recall_thrs, side="left")
            q = np.zeros(len(recall_thrs))
            valid = pos < len(pr)
            q[valid] = pr[pos[valid]]
            aps.append(float(q.mean()))
        per_class_ap[cname] = float(np.mean(aps))
        per_class_ap50[cname] = float(aps[0])

    map50_95 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    map50 = float(np.mean(list(per_class_ap50.values()))) if per_class_ap50 else 0.0
    return {
        "evaluator": "local_cv2_fallback",
        "score": 100.0 * map50_95,
        "map50_95": map50_95,
        "map50": map50,
        "per_class_ap": per_class_ap,
        "per_class_ap50": per_class_ap50,
        "n_images": len(image_paths),
        "n_gt_per_class": n_gt_total,
    }


def analyze_predictions_against_gt(
    pred_json: dict[str, Any],
    labels_dir: Path,
    image_paths: list[Path],
    classes: list[str],
    rle_mod: Any,
    conf_thr: float = 0.25,
    high_conf_thr: float = 0.70,
) -> dict[str, Any]:
    """Compute confusion matrix, worst images, and Tier-2 Confident Learning suspects."""
    nc = len(classes)
    confusion = np.zeros((nc + 1, nc + 1), dtype=int)  # rows: pred (nc=missed GT), cols: gt (nc=FP bg)
    by_stem = {Path(im["file"]).stem: im for im in pred_json.get("images", [])}

    missing_labels: list[dict[str, Any]] = []
    wrong_class: list[dict[str, Any]] = []
    bad_boundaries: list[dict[str, Any]] = []
    image_errors: list[dict[str, Any]] = []

    for img_p in image_paths:
        w, h = image_size(img_p)
        gt_insts = read_yolo_seg_file(labels_dir / f"{img_p.stem}.txt")
        gt_masks = [rasterize_norm_polygon(it.pts, w, h) for it in gt_insts]

        pred_im = by_stem.get(img_p.stem, {"instances": []})
        preds = [it for it in pred_im.get("instances", []) if float(it["s"]) >= conf_thr]
        preds.sort(key=lambda it: float(it["s"]), reverse=True)
        pred_masks = [_decode_rle_to_mask(it["rle"], h, w, rle_mod) for it in preds]

        matched_gt: set[int] = set()
        img_fp = img_wrong = 0

        for pi, (pit, pm) in enumerate(zip(preds, pred_masks)):
            pcls = int(pit["c"])
            pconf = float(pit["s"])
            pbox = _mask_bbox(pm, w, h)
            best_gi = -1
            best_iou = 0.0
            for gi, gm in enumerate(gt_masks):
                if bbox_iou(pbox, gt_insts[gi].bbox) <= 0.0:
                    continue
                iou = _mask_iou_pair(pm, gm)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0 and best_iou >= 0.45:
                gcls = gt_insts[best_gi].cls
                matched_gt.add(best_gi)
                confusion[pcls, gcls] += 1
                if pcls != gcls:
                    img_wrong += 1
                    if pconf >= high_conf_thr and best_iou >= 0.55:
                        wrong_class.append({
                            "image": img_p.name,
                            "pred_cls": classes[pcls],
                            "gt_cls": classes[gcls],
                            "conf": round(pconf, 3),
                            "mask_iou": round(best_iou, 3),
                        })
                else:
                    biou = bbox_iou(pbox, gt_insts[best_gi].bbox)
                    if biou >= 0.70 and best_iou < 0.55:
                        bad_boundaries.append({
                            "image": img_p.name,
                            "cls": classes[gcls],
                            "conf": round(pconf, 3),
                            "bbox_iou": round(biou, 3),
                            "mask_iou": round(best_iou, 3),
                        })
            else:
                confusion[pcls, nc] += 1
                img_fp += 1
                if pconf >= high_conf_thr and best_iou < 0.20:
                    missing_labels.append({
                        "image": img_p.name,
                        "pred_cls": classes[pcls],
                        "conf": round(pconf, 3),
                        "max_gt_iou": round(best_iou, 3),
                    })

        img_fn = 0
        for gi, g_inst in enumerate(gt_insts):
            if gi not in matched_gt:
                confusion[nc, g_inst.cls] += 1
                img_fn += 1

        if img_fp + img_fn + img_wrong > 0:
            image_errors.append({
                "image": img_p.name,
                "fn": img_fn,
                "fp": img_fp,
                "wrong_cls": img_wrong,
                "error_score": img_fn * 2 + img_wrong * 2 + img_fp,
            })

    image_errors.sort(key=lambda r: r["error_score"], reverse=True)
    return {
        "confusion_labels": classes + ["BG_OR_MISSED"],
        "confusion_matrix": confusion.tolist(),
        "worst_images": image_errors[:25],
        "confident_qc": {
            "missing_label_suspects": missing_labels,
            "wrong_class_suspects": wrong_class,
            "bad_mask_boundary_suspects": bad_boundaries,
        },
    }


def run_eval_or_qc(
    model_path: Path,
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    mode: str = "eval",
    notebook_path: Path = DEFAULT_NOTEBOOK,
    competition_path: Path = Path("configs/competition.yaml"),
    conf_thr: float = 0.25,
    high_conf_thr: float = 0.70,
    exp_name: str | None = None,
    max_images: int | None = None,
    seed: int = 39,
) -> dict[str, Any]:
    """Evaluate ONNX model using ``detscore_selfcheck`` (or fallback) and write diagnostic JSONs."""
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    sc = materialize_selfcheck(notebook_path, DEFAULT_WORK_SELFCHECK)
    params = sc.decode.DecodeParams.from_config(sc.config)
    image_paths = sc.infer.list_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if max_images is not None and len(image_paths) > max_images:
        image_paths = sorted(random.Random(seed).sample(image_paths, max_images))

    out_dir.mkdir(parents=True, exist_ok=True)
    infer_res = sc.infer.run_images(model_path, image_paths, params, threads=2)
    pred_payload = {
        "schema_version": sc.score.PREDICTIONS_SCHEMA_VERSION,
        "format": "coco_rle",
        "classes": classes,
        "meta": {"model": str(model_path)},
        "params": {
            "conf": params.conf,
            "iou": params.iou,
            "max_det": params.max_det,
            "mask_threshold": params.mask_threshold,
        },
        "images": [
            {
                "file": im.name,
                "height": im.height,
                "width": im.width,
                "latency_ms": im.latency_ms,
                "instances": [
                    {"c": int(it.cls), "s": float(it.score), "rle": it.rle}
                    for it in im.instances
                ],
            }
            for im in infer_res.images
        ],
    }
    pred_path = out_dir / "predictions.json"
    pred_path.write_text(json.dumps(pred_payload), encoding="utf-8")

    score_dict: dict[str, Any] | None = None
    if mode == "eval":
        try:
            stems = [p.stem for p in image_paths]
            sr = sc.score.score_split(pred_path, labels_dir, stems, sc.config)
            score_dict = {"evaluator": "official_detscore", **sr.to_dict()}
        except Exception:  # noqa: BLE001 - e.g. pycocotools absent on local Windows
            score_dict = _fallback_score_split(
                pred_payload,
                labels_dir,
                image_paths,
                classes,
                imgsz=sc.config.imgsz,
                min_area_px=sc.config.min_gt_area_px,
                max_det=sc.config.max_det,
                rle_mod=sc.rle,
            )

    diag = analyze_predictions_against_gt(
        pred_payload,
        labels_dir,
        image_paths,
        classes,
        rle_mod=sc.rle,
        conf_thr=conf_thr,
        high_conf_thr=high_conf_thr,
    )
    report = {
        "mode": mode,
        "model": str(model_path),
        "images_dir": str(images_dir),
        "n_images": len(image_paths),
        "latency_ms_p50": float(np.median(infer_res.latency_ms)) if infer_res.n else 0.0,
        "score_summary": score_dict,
        **diag,
    }
    report_path = out_dir / ("golden_eval_report.json" if mode == "eval" else "confident_qc_report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if exp_name and score_dict is not None:
        state_path = Path("research-state.yaml")
        if state_path.is_file():
            state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
            exps = state.setdefault("experiments", [])
            exps.append({
                "name": exp_name,
                "evaluator": score_dict["evaluator"],
                "score_x100": round(float(score_dict["score"]), 3),
                "map50": round(float(score_dict["map50"]), 4),
                "per_class_ap": {k: round(float(v), 4) for k, v in score_dict.get("per_class_ap", {}).items()},
            })
            state_path.write_text(yaml.safe_dump(state, sort_keys=False, allow_unicode=True), encoding="utf-8")

    return report


# Pre-train zero-shot semantic rules flagging inverted class mappings before GPU training.
# COCO-80 has no autorickshaw/trailer/caravan class, so Other vehicle (3) matches to
# car/truck/bus are tracked as soft review notes instead of hard blocks, while person/bicycle
# matches on Other vehicle remain hard contradictions.
_SEMANTIC_CONTRADICTIONS: dict[int, set[str]] = {
    0: {"motorcycle", "bicycle", "person"},               # Car GT should not match motorbike/bicycle/person
    1: {"motorcycle", "bicycle", "person"},               # Bus GT should not match motorbike/bicycle
    2: {"motorcycle", "bicycle", "person"},               # Truck GT should not match motorbike/bicycle
    3: {"person", "bicycle"},                             # Other vehicle GT should not match pedestrian/bicycle
    4: {"car", "bus", "truck"},                           # Motorcycle GT should not match 4-wheel vehicles
    5: {"car", "bus", "truck"},                           # Motorcyclist GT should not match vans/cars/trucks
    6: {"car", "bus", "truck"},                           # Bicycle GT should not match 4-wheel vehicles
    7: {"car", "bus", "truck"},                           # Cyclist GT should not match 4-wheel vehicles
}

_SEMANTIC_SOFT_WARNINGS: dict[int, set[str]] = {
    3: {"car", "truck", "bus"},                           # Expected zero-shot confusion on autorickshaw/trailer/fallback
}


def check_semantic_consistency_against_coco(
    samples: list[dict[str, Any]],
    classes: list[str],
    min_conf: float = 0.45,
    min_iou: float = 0.60,
    max_contradiction_rate: float = 0.25,
    min_matches: int = 3,
) -> dict[str, Any]:
    """Check GT annotations against zero-shot COCO predictions to catch class-ID corruption early.

    Each item in ``samples`` has keys ``image``, ``gt`` (``[(cls_id, norm_bbox), ...]``),
    and ``coco_pred`` (``[(coco_label_str, conf, norm_bbox), ...]``).
    """
    matched_per_cls: Counter[int] = Counter()
    contradict_per_cls: Counter[int] = Counter()
    soft_warn_per_cls: Counter[int] = Counter()
    by_pair: Counter[str] = Counter()
    soft_by_pair: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for rec in samples:
        gt_items = rec.get("gt", [])
        preds = [p for p in rec.get("coco_pred", []) if float(p[1]) >= min_conf]
        for g_cls, g_box in gt_items:
            if g_cls not in _SEMANTIC_CONTRADICTIONS:
                continue
            best_iou = 0.0
            best_name = ""
            for p_name, _p_conf, p_box in preds:
                iou = bbox_iou(g_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_name = str(p_name).strip().lower()
            if best_iou < min_iou or not best_name:
                continue
            matched_per_cls[g_cls] += 1
            pair_key = f"GT:{classes[g_cls]} == COCO:{best_name}"
            if best_name in _SEMANTIC_CONTRADICTIONS.get(g_cls, set()):
                contradict_per_cls[g_cls] += 1
                by_pair[pair_key] += 1
                if len(examples) < 25:
                    examples.append({"image": rec.get("image", ""), "pair": pair_key, "iou": round(best_iou, 3)})
            elif best_name in _SEMANTIC_SOFT_WARNINGS.get(g_cls, set()):
                soft_warn_per_cls[g_cls] += 1
                soft_by_pair[pair_key] += 1

    total_matches = sum(matched_per_cls.values())
    total_contra = sum(contradict_per_cls.values())
    overall_rate = (total_contra / total_matches) if total_matches else 0.0

    alerts: list[str] = []
    soft_warnings: list[str] = []
    for cid, m_cnt in sorted(matched_per_cls.items()):
        c_cnt = contradict_per_cls[cid]
        if m_cnt >= 3 and (c_cnt / m_cnt) > max_contradiction_rate:
            alerts.append(
                f"CRITICAL CLASS CORRUPTION on '{classes[cid]}': {c_cnt}/{m_cnt} "
                f"({c_cnt / m_cnt:.0%}) matched contradictory COCO objects!"
            )
        w_cnt = soft_warn_per_cls[cid]
        if w_cnt >= 3:
            soft_warnings.append(
                f"REVIEW NOTE on '{classes[cid]}': {w_cnt}/{m_cnt} ({w_cnt / m_cnt:.0%}) matched "
                f"COCO car/truck/bus (expected for autorickshaw/trailer absent from COCO; verify in preview_per_class.jpg)."
            )

    if total_matches < min_matches:
        alerts.append(f"INSUFFICIENT_EVIDENCE: only {total_matches} COCO matches; need {min_matches}")
    ok = len(alerts) == 0 and overall_rate <= max_contradiction_rate
    return {
        "ok": ok,
        "status": "pass" if ok else ("insufficient_evidence" if total_matches < min_matches else "fail"),
        "checked_images": len(samples),
        "total_matches": total_matches,
        "contradictions": total_contra,
        "contradiction_rate": round(overall_rate, 4),
        "by_pair": dict(by_pair),
        "soft_by_pair": dict(soft_by_pair),
        "alerts": alerts,
        "soft_warnings": soft_warnings,
        "examples": examples,
    }


def run_pretrain_semantic_sanity_check(
    dataset_dir: Path,
    split: str = "train",
    sample_images: int = 48,
    seed: int = 39,
    weights: str = "yolo26n-seg.pt",
    max_contradiction_rate: float = 0.25,
    competition_path: Path = Path("configs/competition.yaml"),
    coco_predictor: Callable[[np.ndarray], list[tuple[str, float, tuple[float, float, float, float]]]] | None = None,
) -> dict[str, Any]:
    """Run pretrained COCO model on sampled dataset images to block inverted class maps before training."""
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    img_dir = dataset_dir / "images" / split if (dataset_dir / "images" / split).is_dir() else (dataset_dir / "images")
    lbl_dir = dataset_dir / "labels" / split if (dataset_dir / "labels" / split).is_dir() else (dataset_dir / "labels")

    imgs = sorted(p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
    rng = random.Random(seed)
    # Sample balanced across classes so rare classes (3, 5, 1, 7) are audited
    by_cls: dict[int, list[Path]] = {}
    for p in imgs:
        insts = read_yolo_seg_file(lbl_dir / f"{p.stem}.txt")
        for it in insts:
            by_cls.setdefault(it.cls, []).append(p)
    if by_cls and not (set(by_cls) & set(_SEMANTIC_CONTRADICTIONS)):
        return {"ok": None, "status": "not_applicable", "checked_images": 0,
                "alerts": [], "reason": "COCO contradiction rules cover vehicle classes 0..7 only; visual audit required"}
    picked: list[Path] = []
    seen_p: set[Path] = set()
    for cid in (3, 5, 1, 2, 7, 6, 4, 8, 9, 10, 0):
        cands = by_cls.get(cid, [])
        rng.shuffle(cands)
        for p in cands[: max(3, sample_images // 8)]:
            if p not in seen_p:
                seen_p.add(p)
                picked.append(p)
    if len(picked) < min(sample_images, len(imgs)):
        rest = [p for p in imgs if p not in seen_p]
        rng.shuffle(rest)
        picked.extend(rest[: sample_images - len(picked)])
    picked = picked[:sample_images]

    if coco_predictor is None:
        from ultralytics import YOLO  # noqa: PLC0415

        coco_model = YOLO(weights)

        def _predict_fn(bgr: np.ndarray) -> list[tuple[str, float, tuple[float, float, float, float]]]:
            ih, iw = bgr.shape[:2]
            res = coco_model.predict(bgr, conf=0.40, imgsz=640, verbose=False)[0]
            if res.boxes is None:
                return []
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cids = res.boxes.cls.cpu().numpy().astype(int)
            out_preds = []
            for b, cf, c in zip(xyxy, confs, cids):
                nm = str(coco_model.names[int(c)]).strip().lower()
                out_preds.append((nm, float(cf), (float(b[0]) / iw, float(b[1]) / ih, float(b[2]) / iw, float(b[3]) / ih)))
            return out_preds

        coco_predictor = _predict_fn

    samples: list[dict[str, Any]] = []
    for p in picked:
        raw = np.fromfile(str(p), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        insts = read_yolo_seg_file(lbl_dir / f"{p.stem}.txt")
        samples.append({
            "image": p.name,
            "gt": [(it.cls, it.bbox) for it in insts],
            "coco_pred": coco_predictor(bgr),
        })
    return check_semantic_consistency_against_coco(samples, classes, max_contradiction_rate=max_contradiction_rate)


def render_per_class_preview_grid(
    dataset_dir: Path,
    out_path: Path,
    split: str = "train",
    samples_per_class: int = 4,
    cell_w: int = 320,
    cell_h: int = 192,
    seed: int = 39,
    competition_path: Path = Path("configs/competition.yaml"),
) -> Path:
    """Render an 11-row class verification sheet highlighting only each row's target class."""
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    img_dir = dataset_dir / "images" / split if (dataset_dir / "images" / split).is_dir() else (dataset_dir / "images")
    lbl_dir = dataset_dir / "labels" / split if (dataset_dir / "labels" / split).is_dir() else (dataset_dir / "labels")

    by_cls: dict[int, list[Path]] = {c: [] for c in range(len(classes))}
    for p in sorted(img_dir.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
            continue
        present = {it.cls for it in read_yolo_seg_file(lbl_dir / f"{p.stem}.txt")}
        for cid in present:
            if cid in by_cls:
                by_cls[cid].append(p)

    rng = random.Random(seed)
    canvas = np.full((len(classes) * cell_h, samples_per_class * cell_w, 3), 28, dtype=np.uint8)
    for cid, cname in enumerate(classes):
        cands = by_cls[cid]
        chosen = rng.sample(cands, min(samples_per_class, len(cands))) if cands else []
        for col in range(samples_per_class):
            y0, x0 = cid * cell_h, col * cell_w
            tile = np.full((cell_h, cell_w, 3), 38, dtype=np.uint8)
            if col < len(chosen):
                p = chosen[col]
                bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    ih, iw = bgr.shape[:2]
                    tile = cv2.resize(bgr, (cell_w, cell_h))
                    for it in read_yolo_seg_file(lbl_dir / f"{p.stem}.txt"):
                        if it.cls != cid:
                            continue
                        pts = np.round(it.pts * np.array([cell_w, cell_h])).astype(np.int32)
                        cv2.polylines(tile, [pts], True, (0, 255, 100), 2)
            cv2.rectangle(tile, (0, 0), (cell_w, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"{cid}: {cname}" if chosen else f"{cid}: {cname} (EMPTY!)", (6, 16), 0, 0.52, (0, 255, 160) if chosen else (80, 80, 255), 1)
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = tile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(out_path.suffix or ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        out_path.write_bytes(buf.tobytes())
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ONNX on Golden Val or run Tier-2 Confident Learning QC.")
    parser.add_argument("--mode", default="eval", choices=["eval", "qc-train"])
    parser.add_argument("--model", type=Path, required=True, help="Path to model.onnx or submission.zip")
    parser.add_argument("--images-dir", type=Path, required=True, help="Images folder (e.g. data/golden_val/images)")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Labels folder (e.g. data/golden_val/labels)")
    parser.add_argument("--out-dir", type=Path, default=Path("work/eval"), help="Output directory")
    parser.add_argument("--exp-name", default=None, help="Optional experiment slug to record in research-state.yaml")
    args = parser.parse_args()

    model_file = args.model
    if model_file.suffix.lower() == ".zip":
        sc = materialize_selfcheck(DEFAULT_NOTEBOOK, DEFAULT_WORK_SELFCHECK)
        extract_dir = args.out_dir / "_extracted"
        summary = sc.package.inspect_zip(str(model_file), sc.config)
        model_file = sc.package.extract_model(str(model_file), extract_dir, sc.config, summary)

    rep = run_eval_or_qc(
        model_path=model_file,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        exp_name=args.exp_name,
    )
    if rep["score_summary"]:
        s = rep["score_summary"]
        print(f"[{s['evaluator']}] Score = {s['score']:.2f} (mAP50={s['map50']:.4f}) on {rep['n_images']} images")
        for cname, ap in sorted(s.get("per_class_ap", {}).items(), key=lambda kv: kv[1]):
            print(f"  {cname:<16}: AP50-95={ap:.4f}")
    cqc = rep["confident_qc"]
    print(
        f"Confident QC flags -> missing={len(cqc['missing_label_suspects'])}, "
        f"wrong_class={len(cqc['wrong_class_suspects'])}, bad_mask={len(cqc['bad_mask_boundary_suspects'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
