"""Failure containment helpers for the self-contained V3.4 notebook."""
from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def release_gpu() -> None:
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retry_download(operation, attempts=3, sleep=time.sleep):
    """Retry transient connection/server failures; do not retry permission or data errors."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            text = str(exc).lower()
            transient = status in (408, 429, 500, 502, 503, 504) or any(
                token in text for token in ("timed out", "timeout", "connection reset", "connection aborted", "temporary failure"))
            if status in (401, 403, 404) or not transient or attempt + 1 == attempts:
                raise
            sleep(2 ** (attempt + 1))


class LazySAM:
    def __init__(self, model_name, device):
        self.model_name, self.device, self.predictor = model_name, device, None

    def __call__(self, image, boxes):
        if not boxes:
            return []
        if self.predictor is None:
            from src.score_focus_v5 import CorrectedSAM2Predictor
            self.predictor = CorrectedSAM2Predictor(self.model_name, self.device)
        return self.predictor(image, boxes)

    def release(self):
        self.predictor = None
        release_gpu()


def unpack_readonly_input(source: Path, destination: Path, min_free_bytes=2*1024**3) -> Path:
    """Read upstream ZIPs without modifying them; extract only under a dedicated owned directory."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Extraction destination must be separate from the input tree")
    archives = sorted(source.rglob("*.zip"))
    if not archives:
        return source
    destination.mkdir(parents=True, exist_ok=True)
    for number, archive in enumerate(archives):
        target_root = destination / f"archive_{number:03d}_{archive.stem}"
        signature = {"path": str(archive), "bytes": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns}
        marker = target_root / ".extraction_complete.json"
        if marker.is_file() and json.loads(marker.read_text()) == signature:
            continue
        with zipfile.ZipFile(archive) as bundle:
            required = sum(item.file_size for item in bundle.infolist())
            if shutil.disk_usage(destination).free < required + min_free_bytes:
                raise OSError(f"Not enough free disk to extract {archive.name}; mount an already-extracted dataset")
            for member in bundle.infolist():
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ZIP symlinks are not supported")
                relative = Path(member.filename.replace("\\", "/"))
                target = (target_root / relative).resolve()
                if target_root.resolve() not in target.parents and target != target_root.resolve():
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target_root.mkdir(parents=True, exist_ok=True)
            bundle.extractall(target_root)
        atomic_json(marker, signature)
    return destination


def cached_stage(destination: Path, signature: dict, build):
    """Reuse only an intact completed conversion with the same source/config signature."""
    marker = destination / ".stage_complete.json"
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("signature") == signature and previous.get("tree_sha256") == dataset_fingerprint(destination):
            print("[CACHE]", destination.name, flush=True)
            return previous["result"]
    # A partial conversion is never accepted as a successful stage.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    result = build(destination)
    if not any((destination / "images").rglob("*.*")):
        raise ValueError(f"No images converted for {destination.name}")
    atomic_json(marker, {"signature": signature, "result": result, "tree_sha256": dataset_fingerprint(destination)})
    return result


def dataset_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for sub in ("images", "labels"):
        for file in sorted(p for p in (root / sub).rglob("*") if p.is_file() and p.suffix != ".cache"):
            digest.update(file.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def train_resilient(factory, weights, kwargs: dict, deadline_callback_factory=None, min_batch=2):
    """Retry CUDA OOM with smaller batches; propagate unrelated errors and record each attempt.

    Each attempt starts from the specified weights with a distinct run name. Checkpoints
    from an interrupted attempt are kept rather than silently treated as completed training.
    """
    attempts = []
    batch = int(kwargs["batch"])
    while batch >= min_batch:
        model = factory(str(weights))
        current = dict(kwargs, batch=batch, name=f"{kwargs['name']}_b{batch}")
        if deadline_callback_factory is not None:
            model.add_callback("on_train_batch_end", deadline_callback_factory())
        failure = None
        try:
            model.train(**current)
            best = Path(model.trainer.best)
            if not best.is_file() or not best.stat().st_size:
                raise RuntimeError("Trainer finished without a valid best.pt")
            attempts.append({"batch": batch, "status": "complete", "weights": str(best)})
            return best, attempts
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or not any(token in str(exc).lower() for token in ("cuda", "cudnn", "gpu")):
                raise
            failure = str(exc).splitlines()[0]
            # Do not retain the exception traceback (and its GPU tensors) across retries.
        finally:
            del model
            release_gpu()
        attempts.append({"batch": batch, "status": "cuda_oom", "message": failure})
        batch //= 2
        print(f"[OOM] Retrying with batch={batch}; previous checkpoints retained", flush=True)
    raise RuntimeError(f"CUDA memory still insufficient at minimum batch={min_batch}; attempts={attempts}")
