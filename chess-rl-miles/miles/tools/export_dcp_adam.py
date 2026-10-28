# Modified for the chess RL integration; see the bundled Miles README.md and LICENSE.
"""Export a committed Miles DCP AdamW state as a named portable payload."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from tools.convert_fsdp_to_hf import validate_committed_source


PAYLOAD_SCHEMA = "interleaved-portable-named-adam-v1"
MARKER_SCHEMA = "interleaved-portable-named-adam-export-v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_identity(source: Mapping[str, Any]) -> dict[str, Any]:
    marker = source["marker"]
    return {
        "iteration": int(marker["iteration"]),
        "commit_sha256": str(marker["commit_sha256"]),
        "marker_sha256": str(source["marker_sha256"]),
        "checkpoint_root": str(source["checkpoint_root"]),
    }


def _extract_optimizer_payload(converted: Any) -> tuple[dict[str, Any], list[Any]]:
    if not isinstance(converted, Mapping):
        raise RuntimeError("converted DCP optimizer is not a mapping")
    node: Any = converted.get("optim_state", converted)
    if isinstance(node, Mapping) and set(node) == {"optim"}:
        node = node["optim"]
    if not isinstance(node, Mapping):
        raise RuntimeError("converted DCP optimizer payload is incomplete")
    state = node.get("state")
    groups = node.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        raise RuntimeError("converted DCP optimizer lacks state or parameter groups")
    return {str(name): dict(row) for name, row in state.items()}, list(groups)


def export_committed_dcp_adam(
    input_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    source = validate_committed_source(input_dir)
    marker = source["marker"]
    if marker.get("optimizer_included") is not True:
        raise RuntimeError("committed checkpoint does not include optimizer state")
    source_identity = _source_identity(source)
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"portable Adam export already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    converted_path = staging / "converted.pt"
    try:
        dcp_to_torch_save(source["checkpoint_root"] / "optimizer", converted_path)
        converted = torch.load(converted_path, map_location="cpu", weights_only=True)
        state, groups = _extract_optimizer_payload(converted)
        if not state:
            raise RuntimeError("committed optimizer contains no parameter state")
        steps: set[int] = set()
        for name, row in state.items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                value = row.get(field)
                if not isinstance(value, torch.Tensor) or value.dtype is not torch.float32:
                    raise RuntimeError(f"committed optimizer {name}.{field} must be FP32")
            if tuple(row["exp_avg"].shape) != tuple(row["exp_avg_sq"].shape):
                raise RuntimeError(f"committed optimizer moment shapes differ for {name}")
            scalar = row["step"]
            if scalar.numel() != 1 or not math.isfinite(float(scalar.item())):
                raise RuntimeError(f"committed optimizer step is invalid for {name}")
            steps.add(int(scalar.item()))
        if len(steps) != 1:
            raise RuntimeError(f"committed optimizer Adam steps differ: {sorted(steps)}")
        adam_step = next(iter(steps))
        if adam_step < int(marker["iteration"]):
            raise RuntimeError("committed optimizer Adam step precedes RL iteration")
        portable = {
            "schema": PAYLOAD_SCHEMA,
            "source_checkpoint": source_identity,
            "source_commit_sha256": source_identity["commit_sha256"],
            "adam_step": adam_step,
            "state": state,
            "param_groups": groups,
        }
        payload_path = staging / "optimizer_state.pt"
        torch.save(portable, payload_path)
        converted_path.unlink()
        source_after = validate_committed_source(input_dir)
        if _source_identity(source_after) != source_identity:
            raise RuntimeError("committed optimizer source changed during export")
        core = {
            "schema": MARKER_SCHEMA,
            "source_checkpoint": source_identity,
            "source_commit_sha256": source_identity["commit_sha256"],
            "adam_step": adam_step,
            "parameter_count": len(state),
            "parameter_group_count": len(groups),
            "parameter_names_sha256": _canonical_sha256(sorted(state)),
            "payload_bytes": payload_path.stat().st_size,
            "payload_sha256": _sha256_file(payload_path),
        }
        complete = {**core, "marker_sha256": _canonical_sha256(core)}
        _atomic_json(staging / "COMPLETE.json", complete)
        os.replace(staging, output)
        return {
            **complete,
            "output": str(output),
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = ["export_committed_dcp_adam"]
