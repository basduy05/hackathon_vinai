"""Automated Benchmark Runner & Markdown/Plot Report Generator (matching the BTC Baseline Report format).

Evaluates any trained ``yolo26n-seg`` checkpoint / ``submission.zip`` on a YOLO-seg validation split
using BOTH the official ``detscore_selfcheck`` pipeline and (optionally) ``Ultralytics model.val()``,
computes ``conf_sweep``, per-class Precision/Recall/F1, Mask/Box IoU distributions, TP vs FP
confidence separation, ONNX graph stats & latency, embeds training/val plots, and outputs a full
``benchmark_report.md`` comparing against a prior baseline JSON or the organizer reference contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.eval_golden import _decode_rle_to_mask, _fallback_score_split, _mask_bbox, _mask_iou_pair, analyze_predictions_against_gt
from src.seg_utils import rasterize_norm_polygon, read_yolo_seg_file
from src.selfcheck_bridge import DEFAULT_NOTEBOOK, DEFAULT_WORK_SELFCHECK, materialize_selfcheck
from src.validate_dataset import bbox_iou, image_size, load_competition_config, run_qc


def _qstats(vals: list[float]) -> dict[str, float] | None:
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": round(float(a.min()), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "median": round(float(np.median(a)), 4),
        "mean": round(float(a.mean()), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "max": round(float(a.max()), 4),
    }


def _eval_operating_point(
    pred_payload: dict[str, Any],
    labels_dir: Path,
    image_paths: list[Path],
    classes: list[str],
    rle_mod: Any,
    imgsz: int,
    min_area_px: float,
    conf_thr: float,
    iou_match: float = 0.50,
    collect_distributions: bool = False,
) -> dict[str, Any]:
    tp = fp = fn = wrong = 0
    tp_mask_ious: list[float] = []
    tp_box_ious: list[float] = []
    tp_confs: list[float] = []
    fp_confs: list[float] = []
    per_cls = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(len(classes))}

    by_stem = {Path(im["file"]).stem: im for im in pred_payload.get("images", [])}
    for img_p in image_paths:
        w, h = image_size(img_p)
        scale_640 = (imgsz / max(h, w)) ** 2
        gt_insts = read_yolo_seg_file(labels_dir / f"{img_p.stem}.txt")
        gt_masks = [rasterize_norm_polygon(it.pts, w, h) for it in gt_insts]
        gt_valid = [i for i, m in enumerate(gt_masks) if float(m.sum()) * scale_640 >= min_area_px]

        pred_im = by_stem.get(img_p.stem, {"instances": []})
        preds = [it for it in pred_im.get("instances", []) if float(it["s"]) >= conf_thr]
        preds.sort(key=lambda it: float(it["s"]), reverse=True)
        pred_masks = [_decode_rle_to_mask(it["rle"], h, w, rle_mod) for it in preds]

        matched_gt: set[int] = set()
        for pit, pm in zip(preds, pred_masks):
            pcls = int(pit["c"])
            pconf = float(pit["s"])
            pbox = _mask_bbox(pm, w, h)
            best_gi = -1
            best_iou = 0.0
            for gi in gt_valid:
                if gi in matched_gt or gt_insts[gi].cls != pcls:
                    continue
                if bbox_iou(pbox, gt_insts[gi].bbox) <= 0.0:
                    continue
                miou = _mask_iou_pair(pm, gt_masks[gi])
                if miou > best_iou:
                    best_iou = miou
                    best_gi = gi

            if best_gi >= 0 and best_iou >= iou_match:
                matched_gt.add(best_gi)
                tp += 1
                per_cls[pcls]["tp"] += 1
                if collect_distributions:
                    tp_mask_ious.append(best_iou)
                    tp_box_ious.append(bbox_iou(pbox, gt_insts[best_gi].bbox))
                    tp_confs.append(pconf)
            else:
                if float(pm.sum()) * scale_640 < min_area_px:
                    continue
                any_iou = max((_mask_iou_pair(pm, gt_masks[gi]) for gi in gt_valid if gi not in matched_gt), default=0.0)
                if any_iou >= iou_match:
                    wrong += 1
                fp += 1
                per_cls[pcls]["fp"] += 1
                if collect_distributions:
                    fp_confs.append(pconf)

        for gi in gt_valid:
            if gi not in matched_gt:
                fn += 1
                per_cls[gt_insts[gi].cls]["fn"] += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    per_cls_prf: dict[str, dict[str, Any]] = {}
    for c, cname in enumerate(classes):
        d = per_cls[c]
        c_tp, c_fp, c_fn = d["tp"], d["fp"], d["fn"]
        if c_tp + c_fp + c_fn == 0:
            continue
        cp = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        cr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        cf1 = (2 * cp * cr / (cp + cr)) if (cp + cr) else 0.0
        per_cls_prf[cname] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "f1": round(cf1, 4),
        }

    out: dict[str, Any] = {
        "conf_thr": conf_thr,
        "iou_match": iou_match,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "wrong_class_among_fp": wrong,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "per_class_prf": per_cls_prf,
    }
    if collect_distributions:
        out["tp_mask_iou_stats"] = _qstats(tp_mask_ious)
        out["tp_box_iou_stats"] = _qstats(tp_box_ious)
        out["tp_conf_stats"] = _qstats(tp_confs)
        out["fp_conf_stats"] = _qstats(fp_confs)
    return out


def compute_benchmark_from_predictions(
    pred_payload: dict[str, Any],
    labels_dir: Path,
    image_paths: list[Path],
    classes: list[str],
    rle_mod: Any,
    imgsz: int = 640,
    min_area_px: float = 64.0,
    max_det: int = 100,
    score_res: dict[str, Any] | None = None,
    conf_show: float = 0.25,
) -> dict[str, Any]:
    """Compute full benchmark metrics (mAP, conf_sweep, IoU/conf distributions, error matrix) from predictions."""
    if score_res is None:
        score_res = _fallback_score_split(
            pred_payload, labels_dir, image_paths, classes,
            imgsz=imgsz, min_area_px=min_area_px, max_det=max_det, rle_mod=rle_mod,
        )
    diag = analyze_predictions_against_gt(
        pred_payload, labels_dir, image_paths, classes, rle_mod=rle_mod, conf_thr=conf_show, high_conf_thr=0.70,
    )
    op_025 = _eval_operating_point(
        pred_payload, labels_dir, image_paths, classes, rle_mod,
        imgsz, min_area_px, conf_show, iou_match=0.50, collect_distributions=True,
    )
    sweep = []
    for c_cand in (0.05, 0.15, 0.25, 0.35, 0.45, 0.60, 0.70, 0.80):
        r = _eval_operating_point(
            pred_payload, labels_dir, image_paths, classes, rle_mod,
            imgsz, min_area_px, c_cand, iou_match=0.50, collect_distributions=False,
        )
        sweep.append({
            "conf": c_cand, "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
        })
    best_f1 = max(sweep, key=lambda x: (x["f1"], x["conf"]))
    return {
        "detscore": score_res,
        "at_conf_025_iou_050": op_025,
        "best_f1_sweep": best_f1,
        "conf_sweep": sweep,
        "diag_conf_025": diag,
    }


def _extract_ultralytics_val_details(vres: Any, classes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for head in ("box", "seg"):
        m = getattr(vres, head)
        f1_arr = np.asarray(m.f1, dtype=float)
        f1_curve_info: dict[str, Any] = {}
        try:
            f1_x = np.asarray(m.curves_results[2][0], dtype=float)
            f1_y = np.asarray(m.curves_results[2][1], dtype=float)
            mean_f1_curve = f1_y.mean(axis=0)
            best_idx = int(np.argmax(mean_f1_curve))
            f1_curve_info = {
                "mean_max_f1": round(float(mean_f1_curve[best_idx]), 4),
                "optimal_conf_at_max_f1": round(float(f1_x[best_idx]), 4),
            }
        except Exception:  # noqa: BLE001
            pass

        ap_idx = [int(i) for i in getattr(m, "ap_class_index", [])]
        per_cls: dict[str, dict[str, Any]] = {}
        for pos, cid in enumerate(ap_idx):
            per_cls[classes[cid]] = {
                "p": round(float(m.p[pos]), 4),
                "r": round(float(m.r[pos]), 4),
                "f1": round(float(f1_arr[pos]), 4) if pos < len(f1_arr) else None,
                "ap50": round(float(m.ap50[pos]), 4),
                "ap75": round(float(m.ap[pos, 5]), 4) if m.ap.ndim == 2 and m.ap.shape[1] > 5 else None,
                "ap50_95": round(float(m.ap[pos].mean()), 4) if m.ap.ndim == 2 else round(float(m.ap[pos]), 4),
            }
        out[head] = {
            "mp": round(float(m.mp), 4),
            "mr": round(float(m.mr), 4),
            "mean_f1_at_max": round(float(f1_arr.mean()), 4) if len(f1_arr) else 0.0,
            "map50": round(float(m.map50), 4),
            "map75": round(float(getattr(m, "map75", 0.0)), 4),
            "map50_95": round(float(m.map), 4),
            **f1_curve_info,
            "per_class": per_cls,
        }
    out["speed_ms"] = {k: round(float(v), 3) for k, v in vres.speed.items()}
    return out


def _generate_charts(summary: dict[str, Any], out_dir: Path, baseline: dict[str, Any] | None = None) -> list[str]:
    """Render academic-quality comparison charts without requiring interactive GUI."""
    saved: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return saved

    det = summary["detscore_val"]["detscore"]
    classes = list(summary["classes"])
    ap_vals = [float(det.get("per_class_ap", {}).get(c, 0.0)) for c in classes]
    ap50_vals = [float(det.get("per_class_ap50", {}).get(c, 0.0)) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
    y = np.arange(len(classes))
    h = 0.36
    ax.barh(y - h / 2, ap50_vals, height=h, label="Current AP50", color="#4C72B0")
    ax.barh(y + h / 2, ap_vals, height=h, label="Current AP50-95", color="#DD8452")
    if baseline and "detscore_val" in baseline:
        b_det = baseline["detscore_val"]["detscore"]
        b_ap = [float(b_det.get("per_class_ap", {}).get(c, 0.0)) for c in classes]
        ax.scatter(b_ap, y + h / 2, color="#222222", zorder=5, s=36, label="Baseline AP50-95")
    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("Average Precision")
    ax.set_title(f"{summary['run_name']} — Per-Class Mask AP (Score = {det['score']:.2f})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p1 = out_dir / "per_class_ap_chart.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(p1.name)

    sweep = summary["detscore_val"]["conf_sweep"]
    confs = [r["conf"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=140)
    ax.plot(confs, [r["precision"] for r in sweep], marker="o", label="Precision")
    ax.plot(confs, [r["recall"] for r in sweep], marker="s", label="Recall")
    ax.plot(confs, [r["f1"] for r in sweep], marker="^", linewidth=2.2, label="F1-Score")
    best = summary["detscore_val"]["best_f1_sweep"]
    ax.axvline(best["conf"], color="gray", linestyle="--", alpha=0.6, label=f"Best F1={best['f1']:.3f} @conf={best['conf']}")
    ax.set_xlim(0.0, 0.85)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{summary['run_name']} — Precision / Recall / F1 Sweep (IoU=0.50)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    p2 = out_dir / "conf_sweep_chart.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2.name)
    return saved


def render_markdown_report(summary: dict[str, Any], baseline: dict[str, Any] | None = None, charts: list[str] | None = None) -> str:
    """Format full Markdown benchmark report matching the BTC Baseline Run structure."""
    det = summary["detscore_val"]["detscore"]
    op = summary["detscore_val"]["at_conf_025_iou_050"]
    best_f1 = summary["detscore_val"]["best_f1_sweep"]
    qc = summary["qc_report"]
    onnx_v = summary.get("onnx_validation") or {}
    onnx_d = onnx_v.get("details") or {}
    zip_s = summary.get("zip_summary") or {}
    u_val = summary.get("ultralytics_val")
    t_info = summary.get("train_history") or {}

    def fmt_delta(cur: float, key_fn) -> str:
        if not baseline or "detscore_val" not in baseline:
            return "—"
        try:
            prev = float(key_fn(baseline))
            d = cur - prev
            sign = "+" if d >= 0 else ""
            return f"{prev:.4f} (`{sign}{d:.4f}`)" if abs(cur) <= 1.5 else f"{prev:.2f} (`{sign}{d:.2f}`)"
        except Exception:  # noqa: BLE001
            return "—"

    lines = [
        f"# Báo Cáo Thực Nghiệm & Đối Chiếu: `{summary['run_name']}` (`TEAM_ID = {summary['team_id']}`)",
        "",
        f"- **Dataset kiểm định**: `{summary['dataset_dir']}` | **Model**: `{summary['model_path']}`",
        f"- **Bộ chấm**: `{det.get('evaluator', 'detscore')}` trên `{det.get('n_images', 0)}` ảnh Val",
    ]
    if t_info:
        lines.append(
            f"- **Quá trình train (`results.csv`)**: `{t_info.get('epochs_ran', '?')}` epochs | "
            f"Best epoch `mAP50-95(M)` = **{t_info.get('best_epoch', '?')}** (`{t_info.get('best_mask_map50_95', 0):.4f}`)"
        )
    lines += [
        "",
        "## 1. Chất Lượng Dữ Liệu (Tier-1 QC)",
        "",
        "| Split | Số ảnh | Tổng instances | Ảnh rỗng | Fatal | Polygon issues | Rò rỉ Train/Val | Cờ Edge-Case (`ambiguity_queue`) |",
        "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |",
    ]
    by_sp_ambig = qc.get("by_split_ambiguity") or {}
    for sp, st in qc["splits"].items():
        sp_a = by_sp_ambig.get(sp)
        amb_count = sp_a["count"] if sp_a is not None else qc["ambiguity_queue_size"]
        amb_flags = sp_a["flags"] if sp_a is not None else qc["edge_case_flag_counts"]
        lines.append(
            f"| `{sp}` | {st['images']} | {st['instances']} | {st['empty_images']} | "
            f"{qc['fatal_count']} | {qc['poly_issue_count']} | {len(qc['duplicates']['leak_exact']) + len(qc['duplicates']['leak_near'])} | "
            f"`{amb_count}` (`{amb_flags}`) |"
        )

    tp_miou = op.get("tp_mask_iou_stats") or {}
    tp_biou = op.get("tp_box_iou_stats") or {}
    tp_conf = op.get("tp_conf_stats") or {}
    fp_conf = op.get("fp_conf_stats") or {}

    u_score_s = f"`{u_val['seg']['map50_95'] * 100:.2f}`" if u_val else "—"
    u_map50_s = f"`{u_val['seg']['map50']:.4f}`" if u_val else "—"
    u_map75_s = f"`{u_val['seg']['map75']:.4f}`" if u_val else "—"
    u_box_s = f"`{u_val['box']['map50']:.4f} / {u_val['box']['map50_95']:.4f}`" if u_val else "—"
    u_mp_s = f"`{u_val['seg']['mp']:.4f}`" if u_val else "—"
    u_mr_s = f"`{u_val['seg']['mr']:.4f}`" if u_val else "—"
    u_f1_s = f"`{u_val['seg']['mean_f1_at_max']:.4f}`" if u_val else "—"
    if u_val and "optimal_conf_at_max_f1" in u_val["seg"]:
        u_f1_s += f" *(`@conf={u_val['seg']['optimal_conf_at_max_f1']}`)*"

    lines += [
        "",
        "## 2. Kết Quả Tổng Hợp & So Sánh Với Mốc Đối Chiếu",
        "",
        "| Chỉ số | Giá trị hiện tại (`detscore`) | Ultralytics `val()` | So với mốc Baseline |",
        "| :--- | :-: | :-: | :-: |",
        f"| **Score (`100 × mask mAP@[.5:.95]`)** | **`{det['score']:.2f}`** | {u_score_s} | {fmt_delta(det['score'], lambda b: b['detscore_val']['detscore']['score'])} |",
        f"| **Mask `mAP50`** | **`{det['map50']:.4f}`** | {u_map50_s} | {fmt_delta(det['map50'], lambda b: b['detscore_val']['detscore']['map50'])} |",
        f"| **Mask `mAP75` / Box `mAP50 / 50-95`** | — | {u_map75_s} / {u_box_s} | — |",
        f"| **Precision (`@conf=0.25, IoU=0.50`)** | `{op['precision']:.4f}` | {u_mp_s} | {fmt_delta(op['precision'], lambda b: b['detscore_val']['at_conf_025_iou_050']['precision'])} |",
        f"| **Recall (`@conf=0.25, IoU=0.50`)** | `{op['recall']:.4f}` | {u_mr_s} | {fmt_delta(op['recall'], lambda b: b['detscore_val']['at_conf_025_iou_050']['recall'])} |",
        f"| **F1-Score (`@conf=0.25` / `max sweep`)** | `{op['f1']:.4f}` / **`{best_f1['f1']:.4f}`** *(`@conf={best_f1['conf']}`)* | {u_f1_s} | {fmt_delta(op['f1'], lambda b: b['detscore_val']['at_conf_025_iou_050']['f1'])} |",
        f"| **Mask IoU của TP (`mean / median`)** | `{tp_miou.get('mean', 0):.4f} / {tp_miou.get('median', 0):.4f}` | — | — |",
        f"| **Box IoU của TP (`mean / median`)** | `{tp_biou.get('mean', 0):.4f} / {tp_biou.get('median', 0):.4f}` | — | — |",
        f"| **Confidence TP vs FP (`mean`)** | `TP={tp_conf.get('mean', 0):.4f}` vs `FP={fp_conf.get('mean', 0):.4f}` | — | — |",
        "",
        "## 3. Bảng Chi Tiết 11 Lớp Trên Tập Val",
        "",
        "| ID | Lớp | GT Val | `AP50-95` | `AP50` | `P@0.25` | `R@0.25` | `F1@0.25` | Ultralytics `AP50-95` |",
        "| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    n_gt_map = det.get("n_gt_per_class", {})
    ap_map = det.get("per_class_ap", {})
    ap50_map = det.get("per_class_ap50", {})
    prf_map = op.get("per_class_prf", {})
    u_cls_map = (u_val["seg"].get("per_class", {}) if u_val else {})
    for cid, cname in enumerate(summary["classes"]):
        gt_n = n_gt_map.get(cname, 0)
        ap_s = f"`{ap_map[cname]:.4f}`" if cname in ap_map else "*(no GT)*"
        ap50_s = f"`{ap50_map[cname]:.4f}`" if cname in ap50_map else "—"
        pr = prf_map.get(cname, {})
        p_s = f"{pr['precision']:.3f}" if pr else "—"
        r_s = f"{pr['recall']:.3f}" if pr else "—"
        f_s = f"{pr['f1']:.3f}" if pr else "—"
        u_c = u_cls_map.get(cname, {})
        u_s = f"`{u_c['ap50_95']:.4f}`" if "ap50_95" in u_c else "—"
        lines.append(f"| `{cid}` | **{cname}** | {gt_n} | {ap_s} | {ap50_s} | {p_s} | {r_s} | {f_s} | {u_s} |")

    lines += [
        "",
        "## 4. Đường Quét Confidence (`IoU_mask = 0.50`)",
        "",
        "| `conf` | `TP` | `FP` | `FN` | `Precision` | `Recall` | `F1` |",
        "| :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for row in summary["detscore_val"]["conf_sweep"]:
        lines.append(f"| `{row['conf']:.2f}` | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |")

    onnx_bytes = float(onnx_d.get("bytes", 0))
    lines += [
        "",
        "## 5. Kiểm Định ONNX & Hiệu Suất Suy Luận",
        "",
        f"- **ONNX Validator**: `ok={onnx_v.get('ok')}` — {onnx_v.get('message_vi', 'N/A')} (`nc={onnx_v.get('nc')}`, `arch={onnx_d.get('architecture', 'yolo26n-seg')}`)",
        f"- **Đồ thị ONNX**: `{onnx_d.get('n_nodes', '?')}` nodes, `{onnx_d.get('conv_nodes', '?')}` Conv/ConvTranspose, `{onnx_d.get('total_params', '?')}` params, opset `{onnx_d.get('opset', '?')}`",
        f"- **Dung lượng**: `model.onnx = {onnx_bytes / 2**20:.2f} MiB` | `submission.zip = {zip_s.get('zip_bytes', 0) / 2**20:.2f} MiB` (giới hạn 42.0 MiB)",
        f"- **Độ trễ CPU (`detscore`)**: model `p50={summary['latency_ms_p50']:.1f} ms` + decode `p50={summary['decode_ms_p50']:.1f} ms` (`output_bytes={summary.get('output_bytes', 0) / 1024:.1f} KB`)",
    ]

    if charts:
        lines += ["", "## 6. Biểu Đồ Trực Quan", ""]
        for ch in charts:
            lines.append(f"![{ch}]({ch})")

    return "\n".join(lines) + "\n"


def run_benchmark(
    submission_or_onnx: Path,
    dataset_dir: Path,
    out_dir: Path,
    run_name: str = "team34_run",
    weights_pt: Path | None = None,
    baseline_json: Path | None = None,
    train_run_dir: Path | None = None,
    notebook_path: Path = DEFAULT_NOTEBOOK,
    competition_path: Path = Path("configs/competition.yaml"),
    max_val_images: int | None = None,
) -> dict[str, Any]:
    """Run full benchmark suite and write ``benchmark_summary.json`` + ``benchmark_report.md``."""
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    team_id = int(cfg["competition"].get("team_id", 39))
    sc = materialize_selfcheck(notebook_path, DEFAULT_WORK_SELFCHECK)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_summary = None
    if submission_or_onnx.suffix.lower() == ".zip":
        zip_obj = sc.package.inspect_zip(str(submission_or_onnx), sc.config)
        zip_summary = zip_obj.to_dict()
        model_onnx = sc.package.extract_model(str(submission_or_onnx), out_dir / "_extracted", sc.config, zip_obj)
    else:
        model_onnx = submission_or_onnx
    onnx_val = sc.validate.validate_onnx(model_onnx, sc.config).to_dict()

    qc_rep = run_qc(dataset_dir, cfg, out_dir=out_dir / "qc")
    val_img_dir = dataset_dir / "images" / "val" if (dataset_dir / "images" / "val").is_dir() else (dataset_dir / "images")
    val_lbl_dir = dataset_dir / "labels" / "val" if (dataset_dir / "labels" / "val").is_dir() else (dataset_dir / "labels")

    image_paths = sc.infer.list_images(val_img_dir)
    if max_val_images and len(image_paths) > max_val_images:
        image_paths = image_paths[:max_val_images]

    params = sc.decode.DecodeParams.from_config(sc.config)
    infer_res = sc.infer.run_images(model_onnx, image_paths, params, threads=2, warmup=True)
    pred_payload = {
        "schema_version": sc.score.PREDICTIONS_SCHEMA_VERSION,
        "format": "coco_rle",
        "classes": classes,
        "meta": {"model": str(model_onnx)},
        "params": {"conf": params.conf, "iou": params.iou, "max_det": params.max_det, "mask_threshold": params.mask_threshold},
        "images": [
            {
                "file": im.name, "height": im.height, "width": im.width,
                "latency_ms": im.latency_ms, "decode_ms": im.decode_ms,
                "instances": [{"c": int(it.cls), "s": float(it.score), "rle": it.rle} for it in im.instances],
            }
            for im in infer_res.images
        ],
    }
    pred_path = out_dir / "predictions_val.json"
    pred_path.write_text(json.dumps(pred_payload), encoding="utf-8")

    score_res: dict[str, Any] | None = None
    try:
        sr = sc.score.score_split(pred_path, val_lbl_dir, [p.stem for p in image_paths], sc.config)
        score_res = {"evaluator": "official_detscore", **sr.to_dict()}
    except Exception:  # noqa: BLE001
        score_res = None

    det_val = compute_benchmark_from_predictions(
        pred_payload, val_lbl_dir, image_paths, classes, sc.rle,
        imgsz=sc.config.imgsz, min_area_px=sc.config.min_gt_area_px, max_det=sc.config.max_det, score_res=score_res,
    )

    u_val_dict = None
    if weights_pt is not None and weights_pt.is_file() and (dataset_dir / "data.yaml").is_file():
        from ultralytics import YOLO  # noqa: PLC0415

        uv = YOLO(str(weights_pt)).val(
            data=str(dataset_dir / "data.yaml"), imgsz=sc.config.imgsz, plots=True,
            project=str(out_dir), name="ultralytics_val", exist_ok=True, verbose=False,
        )
        u_val_dict = _extract_ultralytics_val_details(uv, classes)

    # Gather training history & standard Ultralytics curves if available
    train_history: dict[str, Any] | None = None
    copied_plots: list[str] = []
    if train_run_dir is None and weights_pt is not None and weights_pt.parent.name == "weights":
        train_run_dir = weights_pt.parent.parent
    if train_run_dir and train_run_dir.is_dir():
        res_csv = train_run_dir / "results.csv"
        if res_csv.is_file():
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(res_csv.open(encoding="utf-8"))]
            if rows:
                best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)", 0) or 0))
                train_history = {
                    "epochs_ran": len(rows),
                    "best_epoch": int(float(best_row.get("epoch", 0))),
                    "best_mask_map50_95": float(best_row.get("metrics/mAP50-95(M)", 0) or 0),
                }
    u_val_dir = out_dir / "ultralytics_val"
    if u_val_dir.is_dir():
        for pname in ("MaskF1_curve.png", "MaskPR_curve.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"):
            if (u_val_dir / pname).is_file():
                shutil.copy2(u_val_dir / pname, out_dir / pname)
                copied_plots.append(pname)

    if train_run_dir and train_run_dir.is_dir():
        for pname in ("results.png", "MaskF1_curve.png", "MaskPR_curve.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"):
            if pname not in copied_plots and (train_run_dir / pname).is_file():
                shutil.copy2(train_run_dir / pname, out_dir / pname)
                copied_plots.append(pname)

    baseline_obj = json.loads(baseline_json.read_text(encoding="utf-8")) if (baseline_json and baseline_json.is_file()) else None
    summary = {
        "run_name": run_name,
        "team_id": team_id,
        "dataset_dir": str(dataset_dir),
        "model_path": str(submission_or_onnx),
        "classes": classes,
        "qc_report": qc_rep,
        "onnx_validation": onnx_val,
        "zip_summary": zip_summary,
        "latency_ms_p50": float(np.median(infer_res.latency_ms)) if infer_res.n else 0.0,
        "decode_ms_p50": float(np.median(infer_res.decode_ms)) if infer_res.n else 0.0,
        "output_bytes": int(infer_res.output_bytes),
        "detscore_val": det_val,
        "ultralytics_val": u_val_dict,
        "train_history": train_history,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    charts = _generate_charts(summary, out_dir, baseline=baseline_obj) + copied_plots
    md = render_markdown_report(summary, baseline=baseline_obj, charts=charts)
    (out_dir / "benchmark_report.md").write_text(md, encoding="utf-8")
    return summary


def _select_stratified_val_images(
    val_img_dir: Path,
    val_lbl_dir: Path,
    max_images: int = 55,
    max_total_bytes: int = 22 * 1024 * 1024,
    seed: int = 39,
) -> list[Path]:
    """Select all Val images if compact, otherwise pick a stratified subset covering rare classes first."""
    from src.validate_dataset import IMG_EXTS  # noqa: PLC0415

    all_imgs = sorted(p for p in val_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not all_imgs:
        return []
    total_bytes = sum(p.stat().st_size for p in all_imgs)
    if total_bytes <= max_total_bytes and len(all_imgs) <= max_images * 3:
        return all_imgs

    cls_to_imgs: dict[int, list[Path]] = {}
    img_cls_count: Counter[int] = Counter()
    for im in all_imgs:
        lbl = val_lbl_dir / f"{im.stem}.txt"
        present: set[int] = set()
        if lbl.is_file():
            for raw in lbl.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = raw.split()
                if parts:
                    present.add(int(float(parts[0])))
        if not present:
            present.add(-1)
        for cid in present:
            cls_to_imgs.setdefault(cid, []).append(im)
            img_cls_count[cid] += 1

    rng = random.Random(seed)
    for cid in cls_to_imgs:
        rng.shuffle(cls_to_imgs[cid])

    picked: list[Path] = []
    seen: set[Path] = set()
    ordered_cids = sorted(cls_to_imgs.keys(), key=lambda c: (c < 0, img_cls_count[c]))
    while len(picked) < min(max_images, len(all_imgs)):
        added = False
        for cid in ordered_cids:
            for cand in cls_to_imgs[cid]:
                if cand not in seen:
                    seen.add(cand)
                    picked.append(cand)
                    added = True
                    break
            if len(picked) >= max_images:
                break
        if not added:
            break
    return sorted(picked)


def build_quick_local_pack(
    out_zip: Path,
    submission_or_onnx: Path,
    dataset_dir: Path,
    weights_pt: Path | None = None,
    train_run_dir: Path | None = None,
    extra_files: list[Path] | None = None,
    competition_path: Path = Path("configs/competition.yaml"),
    max_val_images: int = 55,
    max_val_bytes: int = 22 * 1024 * 1024,
    seed: int = 39,
) -> Path:
    """Create lightweight ``quick_local_pack.zip`` containing everything needed to test locally."""
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    val_img_dir = dataset_dir / "images" / "val" if (dataset_dir / "images" / "val").is_dir() else (dataset_dir / "images")
    val_lbl_dir = dataset_dir / "labels" / "val" if (dataset_dir / "labels" / "val").is_dir() else (dataset_dir / "labels")
    picked_imgs = _select_stratified_val_images(
        val_img_dir, val_lbl_dir, max_images=max_val_images, max_total_bytes=max_val_bytes, seed=seed
    )

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        sub_arc = "submission.zip" if submission_or_onnx.suffix.lower() == ".zip" else "model.onnx"
        zf.write(submission_or_onnx, sub_arc)
        if weights_pt is not None and weights_pt.is_file():
            zf.write(weights_pt, "best.pt")

        for im in picked_imgs:
            zf.write(im, f"val_dataset/images/val/{im.name}")
            lbl = val_lbl_dir / f"{im.stem}.txt"
            zf.writestr(f"val_dataset/labels/val/{im.stem}.txt", lbl.read_text(encoding="utf-8") if lbl.is_file() else "")

        portable_yaml = {
            "path": ".",
            "train": "images/val",
            "val": "images/val",
            "nc": len(classes),
            "names": {i: c for i, c in enumerate(classes)},
        }
        zf.writestr("val_dataset/data.yaml", yaml.safe_dump(portable_yaml, sort_keys=False, allow_unicode=True))

        resolved_run = train_run_dir
        if resolved_run is None and weights_pt is not None and weights_pt.parent.name == "weights":
            resolved_run = weights_pt.parent.parent
        if resolved_run and resolved_run.is_dir():
            for art in (
                "results.csv", "results.png", "MaskF1_curve.png", "MaskPR_curve.png",
                "val_batch0_labels.jpg", "val_batch0_pred.jpg",
            ):
                p = resolved_run / art
                if p.is_file():
                    zf.write(p, f"train_artifacts/{art}")

        for ef in (extra_files or []):
            if ef and ef.is_file():
                zf.write(ef, ef.name)
    return out_zip


def unpack_and_run_local_pack(
    pack_zip: Path,
    out_dir: Path,
    run_name: str = "local_check",
    baseline_json: Path | None = None,
    notebook_path: Path = DEFAULT_NOTEBOOK,
    competition_path: Path = Path("configs/competition.yaml"),
) -> dict[str, Any]:
    """Unpack ``quick_local_pack.zip`` downloaded from Kaggle and run full local benchmark on ``val_dataset``."""
    unpack_dir = out_dir / "_unpacked"
    shutil.rmtree(unpack_dir, ignore_errors=True)
    unpack_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pack_zip) as zf:
        zf.extractall(unpack_dir)

    sub_zip = unpack_dir / "submission.zip"
    if not sub_zip.is_file():
        sub_zip = unpack_dir / "model.onnx"
    best_pt = unpack_dir / "best.pt"
    val_ds = unpack_dir / "val_dataset"
    train_run = unpack_dir / "train_artifacts"

    if val_ds.is_dir():
        cfg = load_competition_config(competition_path)
        classes: list[str] = cfg["classes"]
        local_yaml = {
            "path": str(val_ds.resolve()),
            "train": "images/train" if (val_ds / "images" / "train").is_dir() else "images/val",
            "val": "images/val",
            "nc": len(classes),
            "names": {i: c for i, c in enumerate(classes)},
        }
        (val_ds / "data.yaml").write_text(yaml.safe_dump(local_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if baseline_json is None and (unpack_dir / "baseline_summary.json").is_file():
        baseline_json = unpack_dir / "baseline_summary.json"

    return run_benchmark(
        submission_or_onnx=sub_zip,
        dataset_dir=val_ds,
        out_dir=out_dir,
        run_name=run_name,
        weights_pt=best_pt if best_pt.is_file() else None,
        baseline_json=baseline_json,
        train_run_dir=train_run if train_run.is_dir() else None,
        notebook_path=notebook_path,
        competition_path=competition_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full BTC-style benchmark report and comparison plots.")
    parser.add_argument("--unpack-pack", type=Path, default=None, help="Path to quick_local_pack.zip from Kaggle")
    parser.add_argument("--submission", type=Path, default=None, help="Path to submission.zip or model.onnx")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="YOLO-seg dataset dir (with images/val + labels/val)")
    parser.add_argument("--out-dir", type=Path, default=Path("work/benchmark/latest"))
    parser.add_argument("--run-name", default="team34_eval")
    parser.add_argument("--weights", type=Path, default=None, help="Optional .pt checkpoint for Ultralytics curves")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Prior benchmark_summary.json for comparison")
    parser.add_argument("--max-val-images", type=int, default=None)
    args = parser.parse_args()

    if args.unpack_pack is not None:
        res = unpack_and_run_local_pack(args.unpack_pack, args.out_dir, run_name=args.run_name)
    elif args.submission is not None and args.dataset_dir is not None:
        res = run_benchmark(
            submission_or_onnx=args.submission,
            dataset_dir=args.dataset_dir,
            out_dir=args.out_dir,
            run_name=args.run_name,
            weights_pt=args.weights,
            baseline_json=args.baseline_json,
            max_val_images=args.max_val_images,
        )
    else:
        parser.error("Provide either --unpack-pack <zip> OR both --submission and --dataset-dir")
        return 2

    ds = res["detscore_val"]["detscore"]
    op = res["detscore_val"]["at_conf_025_iou_050"]
    print(f"Benchmark [{res['run_name']}]: Score={ds['score']:.2f} | mAP50={ds['map50']:.4f} | P={op['precision']:.4f} R={op['recall']:.4f} F1={op['f1']:.4f}")
    print(f"Report saved to: {args.out_dir / 'benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
