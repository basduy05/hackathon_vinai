"""Shared YOLO-seg polygon, mask, dataset I/O, and registry helpers.

Centralizes polygon normalization, tight contour extraction (preserving high-IoU boundaries for
``100 * mask mAP@[.5:.95]``), Shapely validity repair, and provenance logging so Phase 2+ modules
share one canonical implementation.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image, ImageOps
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from src.validate_dataset import IMG_EXTS, dhash64, make_shapely_polygon, polygon_area, sha256_file


@dataclass
class SegInstance:
    """Single normalized YOLO-seg instance with cached Shapely geometry and bbox."""

    cls: int
    pts: np.ndarray  # shape (N, 2), normalized [0, 1]
    geom: Polygon | None = None
    source_label: str = ""

    def __post_init__(self) -> None:
        self.pts = np.clip(np.asarray(self.pts, dtype=np.float64).reshape(-1, 2), 0.0, 1.0)
        if self.geom is None and len(self.pts) >= 3:
            self.geom = make_shapely_polygon(self.pts)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.pts[:, 0].min()),
            float(self.pts[:, 1].min()),
            float(self.pts[:, 0].max()),
            float(self.pts[:, 1].max()),
        )

    def area_640(self, width: int, height: int, imgsz: int = 640) -> float:
        scale_640 = (imgsz / max(width, height)) ** 2
        return polygon_area(self.pts) * width * height * scale_640


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read BGR image with EXIF orientation applied (Windows-safe unicode paths via np.fromfile)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        rgb = np.asarray(im, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: Path, img_bgr: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    path.write_bytes(buf.tobytes())


def extract_polygons_from_shapely(geom: Any) -> list[np.ndarray]:
    """Convert any Shapely Polygon/MultiPolygon into normalized ``(N, 2)`` arrays (>=3 vertices)."""
    if geom is None or geom.is_empty:
        return []
    polys: list[Polygon] = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
    out: list[np.ndarray] = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0:
            continue
        coords = np.asarray(poly.exterior.coords, dtype=np.float64)
        if len(coords) >= 4 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) >= 3:
            coords = np.clip(coords, 0.0, 1.0)
            if len(np.unique(np.round(coords, 6), axis=0)) >= 3:
                out.append(coords)
    return out


def largest_polygon_from_geom(geom: Any) -> np.ndarray | None:
    """Return the largest valid exterior polygon from a Shapely geometry."""
    cands = extract_polygons_from_shapely(geom)
    if not cands:
        return None
    return max(cands, key=polygon_area)


def mask_to_polygon(
    mask: np.ndarray,
    width: int,
    height: int,
    epsilon_ratio: float = 0.0015,
    min_area_px: float = 12.0,
    max_vertices: int = 128,
) -> np.ndarray | None:
    """Convert binary mask ``(H, W)`` into one tight normalized polygon ``(N, 2)``.

    Uses a tight ``approxPolyDP`` tolerance (default 0.15% of perimeter) to preserve boundary IoU
    for ``mAP@[.5:.95]``. When SAM/COCO produces multiple disconnected blobs for one object, keeps
    the dominant contour if secondary blobs are tiny, or bridges close parts cleanly.
    """
    bin_mask = (np.asarray(mask) > 0).astype(np.uint8)
    if bin_mask.shape[:2] != (height, width):
        bin_mask = cv2.resize(bin_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    valid_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area_px]
    if not valid_contours:
        return None
    valid_contours.sort(key=lambda item: item[1], reverse=True)
    main_cnt, main_area = valid_contours[0]

    # If significant secondary fragments exist (e.g., split by thin pole), keep main if >=75% area,
    # otherwise bridge fragments via Shapely convex bridge/union or take largest clean contour.
    chosen = main_cnt
    if len(valid_contours) > 1 and valid_contours[1][1] >= 0.25 * main_area:
        # Try gentle closing kernel to reconnect closely separated halves of an occluded object
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        c2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if c2:
            chosen = max(c2, key=cv2.contourArea)

    peri = cv2.arcLength(chosen, True)
    eps = max(0.5, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(chosen, eps, True).reshape(-1, 2)
    while len(approx) > max_vertices and eps < 0.02 * peri:
        eps *= 1.35
        approx = cv2.approxPolyDP(chosen, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        approx = chosen.reshape(-1, 2)
    if len(approx) < 3:
        return None

    pts = approx.astype(np.float64)
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
    if len(np.unique(np.round(pts, 6), axis=0)) < 3:
        return None
    return pts


def rasterize_norm_polygon(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rasterize normalized ``(N, 2)`` polygon into uint8 binary mask ``(height, width)``."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    px = np.round(np.asarray(pts, dtype=np.float64) * np.array([width, height], dtype=np.float64)).astype(np.int32)
    px[:, 0] = np.clip(px[:, 0], 0, max(0, width - 1))
    px[:, 1] = np.clip(px[:, 1], 0, max(0, height - 1))
    if len(px) >= 3:
        cv2.fillPoly(canvas, [px], 1)
    return canvas


def format_yolo_seg_line(cls_id: int, pts: np.ndarray, precision: int = 6) -> str:
    flat = np.clip(pts.reshape(-1), 0.0, 1.0)
    coords_str = " ".join(f"{v:.{precision}f}" for v in flat)
    return f"{int(cls_id)} {coords_str}"


def write_yolo_seg_label(path: Path, instances: Sequence[tuple[int, np.ndarray]], precision: int = 6) -> None:
    """Write YOLO-seg label file; creates an empty 0-byte ``.txt`` when ``instances`` is empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_yolo_seg_line(cls_id, pts, precision=precision) for cls_id, pts in instances if len(pts) >= 3]
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def read_yolo_seg_file(path: Path) -> list[SegInstance]:
    if not path.is_file():
        return []
    out: list[SegInstance] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cols = raw.split()
        if len(cols) < 7 or len(cols) % 2 == 0:
            continue
        cls_id = int(float(cols[0]))
        pts = np.array([float(x) for x in cols[1:]], dtype=np.float64).reshape(-1, 2)
        out.append(SegInstance(cls=cls_id, pts=pts))
    return out


def merge_polygons_union(polys: Sequence[np.ndarray]) -> np.ndarray | None:
    """Merge multiple normalized polygons (e.g. motorcycle + 1..2 riders) into one tight polygon."""
    geoms = [g for g in (make_shapely_polygon(p) for p in polys) if g is not None]
    if not geoms:
        return None
    if len(geoms) == 1:
        return largest_polygon_from_geom(geoms[0])
    merged = unary_union(geoms)
    poly = largest_polygon_from_geom(merged)
    if poly is not None and len(extract_polygons_from_shapely(merged)) == 1:
        return poly
    # If rider and vehicle have a tiny sub-pixel gap in annotation, bridge via tiny dilation then erode
    bridged = unary_union([g.buffer(0.004) for g in geoms]).buffer(-0.0035)
    return largest_polygon_from_geom(bridged) or poly


def write_data_yaml(dataset_dir: Path, classes: Sequence[str], has_val: bool = True) -> Path:
    data: dict[str, Any] = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val" if has_val else "images/train",
        "nc": len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def transfer_image(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    elif mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def collect_exclusion_hashes(exclude_dir: Path | None) -> tuple[set[str], list[int]]:
    """Collect SHA-256 and dhash64 fingerprints of images in ``exclude_dir`` (e.g. golden_val) to block leakage."""
    if exclude_dir is None or not exclude_dir.exists():
        return set(), []
    shas: set[str] = set()
    dhashes: list[int] = []
    for p in sorted(exclude_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            shas.add(sha256_file(p))
            dhashes.append(dhash64(p))
    return shas, dhashes


def is_near_duplicate(img_path: Path, seen_shas: set[str], seen_dhashes: list[int], max_hamming: int = 2) -> tuple[bool, str, int]:
    """Return ``(is_dup, sha, dh)`` checking exact SHA-256 and ``dhash64`` Hamming distance."""
    sha = sha256_file(img_path)
    dh = dhash64(img_path)
    if sha in seen_shas:
        return True, sha, dh
    if seen_dhashes:
        arr = np.array(seen_dhashes, dtype=np.uint64)
        xor_bits = np.unpackbits((arr ^ np.uint64(dh)).view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        if int(xor_bits.min()) <= max_hamming:
            return True, sha, dh
    return False, sha, dh


def register_dataset_usage(
    registry_path: Path,
    name: str,
    url: str,
    license_name: str,
    raw_format: str,
    train_count: int,
    val_count: int,
    mapping_note: str,
    status: str = "Active",
) -> None:
    """Append or update one dataset entry in ``datasets_registry.md`` for presentation transparency."""
    if not registry_path.is_file():
        registry_path.write_text(
            "# Public Datasets Registry (Mandatory for Final Presentation)\n\n"
            "| # | Tên bộ dữ liệu | URL | Giấy phép (License) | Dạng nhãn gốc | Số ảnh đã dùng (Train / Val) | Cách ánh xạ về 11 lớp | Trạng thái |\n"
            "| :-: | :--- | :--- | :--- | :--- | :-: | :--- | :--- |\n",
            encoding="utf-8",
        )
    lines = registry_path.read_text(encoding="utf-8").splitlines()
    row_prefix = f"| "
    data_rows = [ln for ln in lines if ln.startswith(row_prefix) and not ln.startswith("| #") and not ln.startswith("| :-:")]
    updated = False
    new_rows: list[str] = []
    idx_counter = 1
    for ln in data_rows:
        cols = [c.strip() for c in ln.strip("|").split("|")]
        if len(cols) >= 8 and cols[1] == name:
            new_rows.append(f"| {idx_counter} | {name} | {url} | {license_name} | {raw_format} | {train_count} / {val_count} | {mapping_note} | {status} |")
            updated = True
        else:
            cols[0] = str(idx_counter)
            new_rows.append("| " + " | ".join(cols[:8]) + " |")
        idx_counter += 1
    if not updated:
        new_rows.append(f"| {idx_counter} | {name} | {url} | {license_name} | {raw_format} | {train_count} / {val_count} | {mapping_note} | {status} |")

    header = [ln for ln in lines if not (ln.startswith(row_prefix) and not ln.startswith("| #") and not ln.startswith("| :-:"))]
    while header and not header[-1].strip():
        header.pop()
    registry_path.write_text("\n".join(header + new_rows) + "\n", encoding="utf-8")
