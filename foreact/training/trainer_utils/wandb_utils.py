"""Shared Weights & Biases helpers for ForeAct trainers."""

import os
from pathlib import Path
from typing import Any

import wandb
from accelerate.utils import DistributedType
from omegaconf import OmegaConf

from foreact.training.trainer_utils.config_tracker import AccessTrackedConfig


_DISABLED_VALUES = {"1", "true", "yes", "on"}
_SUPPORTED_MODES = {"online", "offline"}


def get_wandb_mode() -> str:
    mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    return {"dryrun": "offline", "run": "online"}.get(mode, mode)


def wandb_is_disabled() -> bool:
    disabled = os.environ.get("WANDB_DISABLED", "").strip().lower()
    return get_wandb_mode() == "disabled" or disabled in _DISABLED_VALUES


def config_for_wandb(config: Any, accelerator: Any) -> dict[str, Any]:
    raw_config = config.unwrap() if isinstance(config, AccessTrackedConfig) else config
    if OmegaConf.is_config(raw_config):
        serialized = OmegaConf.to_container(raw_config, resolve=True)
    elif isinstance(raw_config, dict):
        serialized = dict(raw_config)
    else:
        serialized = {"config": str(raw_config)}
    if not isinstance(serialized, dict):
        serialized = {"config": serialized}
    serialized["wandb_runtime"] = {
        "num_processes": accelerator.num_processes,
        "gradient_accumulation_steps": accelerator.gradient_accumulation_steps,
    }
    return serialized


def init_wandb(config: Any, accelerator: Any, logger: Any, *, group: str) -> bool:
    enabled = False
    mode = get_wandb_mode()
    if not wandb_is_disabled() and accelerator.is_main_process:
        try:
            if mode not in _SUPPORTED_MODES:
                raise ValueError(f"Unsupported WANDB_MODE={mode!r}; expected online, offline, or disabled")
            entity = getattr(config, "wandb_entity", None)
            if entity is not None:
                entity = str(entity).strip() or None
            wandb.init(
                name=config.run_id,
                dir=str(Path(config.output_dir)),
                project=config.wandb_project,
                entity=entity,
                group=group,
                config=config_for_wandb(config, accelerator),
                mode=mode,
            )
            enabled = True
            logger.info(f"W&B initialized in {mode} mode")
        except Exception as exc:
            logger.warning(f"W&B init failed; continuing without W&B: {exc}")
    accelerator.wait_for_everyone()
    return enabled


def finish_wandb(enabled: bool, logger: Any) -> None:
    if not enabled:
        return
    try:
        wandb.finish()
    except Exception as exc:
        logger.warning(f"W&B finish failed: {exc}")


def configure_deepspeed_gradient_clipping(accelerator: Any, max_norm: Any) -> bool:
    """Propagate trainer clipping into DeepSpeed before ``accelerator.prepare``.

    Accelerate's clipping API intentionally returns ``None`` under DeepSpeed:
    clipping runs inside ``DeepSpeedEngine.step`` instead. The threshold must
    therefore be present in the DeepSpeed config before engine construction.
    """
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        return False
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return False
    clip_value = 0.0 if max_norm is None else float(max_norm)
    if clip_value < 0:
        raise ValueError(f"gradient_clipping must be non-negative, got {clip_value}")
    plugin.deepspeed_config["gradient_clipping"] = clip_value
    return True


def clip_grad_norm_and_get(
    accelerator: Any, model: Any, parameters: Any, max_norm: float
) -> Any:
    """Clip gradients and return their pre-clipping global norm.

    With DeepSpeed, ``accelerator.backward`` has already run engine.step(),
    which clips gradients and caches the global norm. Reading the engine cache
    fixes the ``None`` returned by Accelerate's ``clip_grad_norm_`` DS branch.
    """
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        getter = getattr(model, "get_global_grad_norm", None)
        total_norm = getter() if callable(getter) else None
        if total_norm is None:
            optimizer = getattr(model, "optimizer", None)
            total_norm = getattr(optimizer, "_global_grad_norm", None)
        return total_norm
    return accelerator.clip_grad_norm_(parameters, max_norm)


def gradient_clip_metrics(total_norm: Any, max_norm: float) -> dict[str, float]:
    if total_norm is None:
        return {}
    if hasattr(total_norm, "detach"):
        total_norm = total_norm.detach()
    if hasattr(total_norm, "item"):
        total_norm = total_norm.item()
    norm = float(total_norm)
    limit = float(max_norm)
    return {
        "train/grad_norm": norm,
        "train/grad_was_clipped": float(norm > limit),
        "train/grad_clip_scale": min(1.0, limit / (norm + 1e-12)),
    }
