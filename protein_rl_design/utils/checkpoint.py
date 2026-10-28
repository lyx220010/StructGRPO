from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

CHECKPOINT_TYPE_PRETRAIN = "pretrain"
CHECKPOINT_TYPE_RL = "rl"


def detect_checkpoint_type(ckpt: Dict[str, Any]) -> str:
    if not isinstance(ckpt, dict):
        raise ValueError("A checkpoint must be a dictionary containing policy and hgnn states.")
    missing = [key for key in ("policy", "hgnn") if key not in ckpt]
    if missing:
        raise ValueError(f"Unrecognized checkpoint: missing required model state keys {missing}.")
    explicit = ckpt.get("checkpoint_type")
    if explicit in (CHECKPOINT_TYPE_PRETRAIN, CHECKPOINT_TYPE_RL):
        return explicit
    if ckpt.get("optimizer") is not None or ("ref_policy" in ckpt and "ref_hgnn" in ckpt):
        return CHECKPOINT_TYPE_RL
    return CHECKPOINT_TYPE_PRETRAIN


def load_model_weights(
    ckpt_path: str,
    hgnn,
    policy,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(ckpt_path, map_location=map_location)
    ckpt_type = detect_checkpoint_type(ckpt)
    policy.load_state_dict(ckpt["policy"], strict=strict)
    hgnn.load_state_dict(ckpt["hgnn"], strict=strict)
    policy.eval()
    hgnn.eval()
    ckpt["_loaded_type"] = ckpt_type
    ckpt["_loaded_path"] = str(ckpt_path)
    return ckpt


def save_pretrain_checkpoint(
    path: str,
    policy,
    hgnn,
    step: int,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "checkpoint_type": CHECKPOINT_TYPE_PRETRAIN,
        "step": int(step),
        "policy": policy.state_dict(),
        "hgnn": hgnn.state_dict(),
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, out)
    return str(out)
