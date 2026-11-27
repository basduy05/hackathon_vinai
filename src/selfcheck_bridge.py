"""Bridge to the official embedded ``detscore_selfcheck`` bundle inside ``train_and_export.ipynb``.

Extracts ``SELFCHECK_BUNDLE``, ``EXPORT_CONTRACT``, ``PINNED_PACKAGES``, and class definitions
from the starter notebook so local validation, ONNX verification, and Val scoring stay
100% identical to the competition grading server.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


DEFAULT_NOTEBOOK = Path("train_and_export.ipynb")
DEFAULT_WORK_SELFCHECK = Path("work/_selfcheck")


def load_notebook_constants(notebook_path: str | Path = DEFAULT_NOTEBOOK) -> dict[str, Any]:
    """Read constant definitions (SELFCHECK_BUNDLE, EXPORT_CONTRACT, CLASSES, etc.) from the notebook."""
    nb_path = Path(notebook_path)
    if not nb_path.is_file():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    target_src: str | None = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "SELFCHECK_BUNDLE" in src and "EXPORT_CONTRACT" in src:
            target_src = src
            break
    if target_src is None:
        raise ValueError(f"Could not find SELFCHECK_BUNDLE constants cell in {nb_path}")
    ns: dict[str, Any] = {}
    exec(compile(target_src, str(nb_path), "exec"), ns)  # noqa: S102 - trusted competition notebook cell
    return ns


def materialize_selfcheck(
    notebook_path: str | Path = DEFAULT_NOTEBOOK,
    dest_root: str | Path = DEFAULT_WORK_SELFCHECK,
) -> SimpleNamespace:
    """Write ``detscore_selfcheck`` package files and import its modules cleanly (Windows-safe)."""
    constants = load_notebook_constants(notebook_path)
    bundle = constants["SELFCHECK_BUNDLE"]
    dest_root = Path(dest_root)
    pkg = dest_root / "detscore_selfcheck"
    if pkg.exists():
        shutil.rmtree(pkg)
    for rel, text in bundle["files"].items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (pkg / "competition_config.yaml").write_text(bundle["config_yaml"], encoding="utf-8")

    for name in [m for m in sys.modules if m == "detscore_selfcheck" or m.startswith("detscore_selfcheck.")]:
        del sys.modules[name]
    root_str = str(dest_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    importlib.invalidate_caches()

    ns = SimpleNamespace(constants=constants)
    for mod in ("validate", "package", "errors", "score", "decode", "infer", "rle"):
        setattr(ns, mod, importlib.import_module(f"detscore_selfcheck.{mod}"))
    # cli_infer imports Unix-only `resource`; load it when available without failing on Windows.
    try:
        ns.cli_infer = importlib.import_module("detscore_selfcheck.cli_infer")
    except ModuleNotFoundError as exc:
        if exc.name == "resource":
            ns.cli_infer = None
        else:
            raise
    cfg_mod = importlib.import_module("detscore_selfcheck.config")
    ns.config = cfg_mod.load_config(pkg / "competition_config.yaml")
    return ns


def check_toolchain_versions(pinned: dict[str, str]) -> dict[str, dict[str, str | bool | None]]:
    """Compare locally installed package versions against competition pinned versions."""
    report: dict[str, dict[str, str | bool | None]] = {}
    for pkg, expected in pinned.items():
        try:
            installed = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        report[pkg] = {
            "expected": expected,
            "installed": installed,
            "match": installed == expected,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify competition notebook SELFCHECK_BUNDLE and local toolchain.")
    parser.add_argument("--verify-notebook", type=Path, default=DEFAULT_NOTEBOOK, help="Path to train_and_export.ipynb")
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_WORK_SELFCHECK, help="Directory to extract detscore_selfcheck")
    args = parser.parse_args()

    sc = materialize_selfcheck(args.verify_notebook, args.dest_root)
    c = sc.constants
    versions = check_toolchain_versions(c["PINNED_PACKAGES"])
    print(f"Notebook: {c['NOTEBOOK_NAME']} v{c['NOTEBOOK_VERSION']} | detscore v{c['DETSCORE_VERSION']}")
    print(f"Classes ({len(c['CLASSES'])}): {', '.join(c['CLASSES'])}")
    print(f"Contract: imgsz={sc.config.imgsz}, min_gt_area_px={sc.config.min_gt_area_px}, max_zip_mb={sc.config.max_zip_mb}")
    print(f"Export kwargs: {c['EXPORT_CONTRACT']['export_call']}")
    print("Toolchain status:")
    for pkg, info in versions.items():
        status = "OK" if info["match"] else "MISMATCH"
        print(f"  - {pkg}: installed={info['installed']} expected={info['expected']} [{status}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
