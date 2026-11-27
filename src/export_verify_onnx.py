"""Export ``yolo26n-seg`` to ONNX following the exact competition contract and validate with ``detscore_selfcheck``."""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Any

from src.selfcheck_bridge import (
    DEFAULT_NOTEBOOK,
    DEFAULT_WORK_SELFCHECK,
    check_toolchain_versions,
    materialize_selfcheck,
)


EXPORT_KEYS = ("format", "imgsz", "opset", "simplify", "dynamic", "batch")


def build_export_kwargs(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract exact Ultralytics export args (never passes ``nms`` or ``half``)."""
    if contract.get("nms") is not None or contract.get("half") not in (False, None):
        raise ValueError(f"Unsafe EXPORT_CONTRACT: nms={contract.get('nms')!r}, half={contract.get('half')!r}")
    return {k: contract[k] for k in EXPORT_KEYS}


def write_reproducible_zip(model_path: Path, zip_path: Path, entry_name: str = "model.onnx") -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    info = zipfile.ZipInfo(entry_name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, model_path.read_bytes())
    return zip_path


def export_and_verify(
    weights: Path,
    out_dir: Path,
    notebook_path: Path = DEFAULT_NOTEBOOK,
    allow_version_mismatch: bool = False,
    smoke_images_dir: Path | None = None,
) -> dict[str, Any]:
    """Export ``.pt`` weights to FP32 ONNX, verify against platform validator, and build ``submission.zip``."""
    weights = Path(weights)
    if weights.suffix != ".pt" or not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Valid .pt weights not found: {weights}")

    sc = materialize_selfcheck(notebook_path, DEFAULT_WORK_SELFCHECK)
    constants = sc.constants
    versions = check_toolchain_versions(constants["PINNED_PACKAGES"])
    mismatches = [f"{k}: got {v['installed']}, expected {v['expected']}" for k, v in versions.items() if not v["match"]]
    if mismatches and not allow_version_mismatch:
        raise RuntimeError(
            "Toolchain version mismatch (pass --allow-version-mismatch for local dry-runs only):\n  - "
            + "\n  - ".join(mismatches)
        )
    if mismatches:
        print("[WARN] Toolchain mismatch (local dry-run mode):")
        for m in mismatches:
            print(f"  - {m}")

    from ultralytics import YOLO

    yolo_model = YOLO(str(weights))
    model_names = [yolo_model.names[i] for i in sorted(yolo_model.names)]
    expected_classes = list(constants["CLASSES"])
    if model_names != expected_classes:
        raise ValueError(f"Model class names {model_names} != competition classes {expected_classes}")

    out_dir.mkdir(parents=True, exist_ok=True)
    export_kwargs = build_export_kwargs(constants["EXPORT_CONTRACT"])
    print(f"Exporting with contract kwargs: {export_kwargs}")
    exported_onnx = Path(str(yolo_model.export(**export_kwargs)))
    if not exported_onnx.is_file():
        raise RuntimeError(f"Ultralytics export did not produce {exported_onnx}")

    model_onnx = out_dir / "model.onnx"
    shutil.copyfile(exported_onnx, model_onnx)

    res = sc.validate.validate_onnx(model_onnx, sc.config)
    if not res.ok:
        raise RuntimeError(f"Platform validator rejected ONNX [{res.code}]: {res.message_vi}")

    zip_path = write_reproducible_zip(model_onnx, out_dir / "submission.zip")
    zip_summary = sc.package.inspect_zip(str(zip_path), sc.config)
    extract_dir = out_dir / "_zipcheck"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extracted = sc.package.extract_model(str(zip_path), extract_dir, sc.config, zip_summary)
    if sc.validate.sha256_file(extracted) != res.file_sha256:
        raise RuntimeError("Extracted model.onnx SHA-256 mismatch")

    print(f"[PASS] {model_onnx} ({model_onnx.stat().st_size / 2**20:.2f} MiB) | zip={zip_path.stat().st_size / 2**20:.2f} MiB")
    print(f"  architecture={res.details.get('architecture')} | nc={res.nc} | fingerprint={res.fingerprint[:16]}...")

    infer_summary = None
    if smoke_images_dir is not None:
        imgs = sc.infer.list_images(smoke_images_dir)
        if not imgs:
            raise FileNotFoundError(f"No images found in smoke test dir: {smoke_images_dir}")
        params = sc.decode.DecodeParams.from_config(sc.config)
        infer_res = sc.infer.run_images(model_onnx, imgs[:5], params, threads=1)
        infer_summary = {
            "n_images": infer_res.n,
            "n_instances": infer_res.n_instances,
            "latency_ms_p50": float(infer_res.latency_ms.mean()) if infer_res.n else 0.0,
        }
        print(f"[PASS] Smoke inference on {infer_res.n} images: {infer_res.n_instances} instances decoded.")

    return {
        "onnx": str(model_onnx),
        "zip": str(zip_path),
        "validation": res.to_dict(),
        "zip_summary": zip_summary.to_dict(),
        "smoke_inference": infer_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO26n-seg to ONNX and verify with official validator.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO26n-seg .pt checkpoint")
    parser.add_argument("--out-dir", type=Path, default=Path("work/submission"), help="Output directory")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK, help="Path to train_and_export.ipynb")
    parser.add_argument("--allow-version-mismatch", action="store_true", help="Allow local dry-run with different package versions")
    parser.add_argument("--smoke-images", type=Path, default=None, help="Optional folder of images to test ONNX inference")
    args = parser.parse_args()

    export_and_verify(
        weights=args.weights,
        out_dir=args.out_dir,
        notebook_path=args.notebook,
        allow_version_mismatch=args.allow_version_mismatch,
        smoke_images_dir=args.smoke_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
