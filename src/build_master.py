"""Assemble clean staged datasets + locked Golden Val into ``data/master/`` and optionally package ``du_lieu.zip``.

Enforces zero train/val leakage against ``data/golden_val/`` using SHA-256 + ``dhash64`` and runs
Tier-1 ``validate_dataset.run_qc`` before finalizing the master training bundle.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from src.seg_utils import (
    collect_exclusion_hashes,
    is_near_duplicate,
    read_yolo_seg_file,
    transfer_image,
    write_data_yaml,
    write_yolo_seg_label,
)
from src.validate_dataset import IMG_EXTS, load_competition_config, run_qc


def build_master_dataset(
    train_sources: list[Path],
    golden_val_dir: Path | None,
    master_dir: Path,
    competition_path: Path = Path("configs/competition.yaml"),
    transfer_mode: str = "copy",
    zip_out: Path | None = None,
) -> dict[str, Any]:
    cfg = load_competition_config(competition_path)
    classes: list[str] = cfg["classes"]
    dup_hamming_max = int(cfg["qc_thresholds"]["dup_hamming_max"])

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (master_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Copy locked Golden Val first and record its fingerprints
    val_count = 0
    if golden_val_dir is not None and golden_val_dir.exists():
        val_img_dir = golden_val_dir / "images" / "val" if (golden_val_dir / "images" / "val").is_dir() else (golden_val_dir / "images")
        val_lbl_dir = golden_val_dir / "labels" / "val" if (golden_val_dir / "labels" / "val").is_dir() else (golden_val_dir / "labels")
        for img_p in sorted(p for p in val_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
            insts = read_yolo_seg_file(val_lbl_dir / f"{img_p.stem}.txt")
            transfer_image(img_p, master_dir / "images" / "val" / img_p.name, mode=transfer_mode)
            write_yolo_seg_label(master_dir / "labels" / "val" / f"{img_p.stem}.txt", [(it.cls, it.pts) for it in insts])
            val_count += 1

    seen_shas, seen_dhashes = collect_exclusion_hashes(master_dir / "images" / "val")

    # 2. Merge all staged train sources while blocking any duplicate or val leakage
    train_count = skipped_dup = 0
    for src_root in train_sources:
        if not src_root.exists():
            continue
        t_img_dir = src_root / "images" / "train" if (src_root / "images" / "train").is_dir() else (src_root / "images")
        t_lbl_dir = src_root / "labels" / "train" if (src_root / "labels" / "train").is_dir() else (src_root / "labels")
        if not t_img_dir.is_dir():
            continue
        prefix = src_root.name
        for img_p in sorted(p for p in t_img_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS):
            is_dup, sha, dh = is_near_duplicate(img_p, seen_shas, seen_dhashes, max_hamming=dup_hamming_max)
            if is_dup:
                skipped_dup += 1
                continue
            seen_shas.add(sha)
            seen_dhashes.append(dh)
            stem = f"{prefix}_{img_p.stem}"
            insts = read_yolo_seg_file(t_lbl_dir / f"{img_p.stem}.txt")
            transfer_image(img_p, master_dir / "images" / "train" / f"{stem}{img_p.suffix.lower()}", mode=transfer_mode)
            write_yolo_seg_label(master_dir / "labels" / "train" / f"{stem}.txt", [(it.cls, it.pts) for it in insts])
            train_count += 1

    write_data_yaml(master_dir, classes, has_val=val_count > 0)
    qc_report = run_qc(master_dir, cfg, out_dir=Path("work/qc/master"))

    zip_written = None
    if zip_out is not None and qc_report["ok"]:
        zip_out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(master_dir.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(master_dir).as_posix())
        zip_written = str(zip_out)

    return {
        "master_dir": str(master_dir),
        "train_images": train_count,
        "val_images": val_count,
        "skipped_duplicates_or_leak": skipped_dup,
        "qc_ok": qc_report["ok"],
        "fatal_count": qc_report["fatal_count"],
        "ambiguity_queue_size": qc_report["ambiguity_queue_size"],
        "zip_path": zip_written,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge clean staged batches + Golden Val into data/master.")
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="List of staged dataset dirs")
    parser.add_argument("--golden-val", type=Path, default=Path("data/golden_val"), help="Locked Golden Val directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/master"), help="Output master dataset directory")
    parser.add_argument("--zip-out", type=Path, default=None, help="Optional du_lieu.zip output path for Colab/Kaggle")
    args = parser.parse_args()

    res = build_master_dataset(
        train_sources=args.sources,
        golden_val_dir=args.golden_val if args.golden_val.exists() else None,
        master_dir=args.out_dir,
        zip_out=args.zip_out,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["qc_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
