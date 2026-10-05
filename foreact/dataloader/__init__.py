"""Dataloader construction for ForeAct training."""

from pathlib import Path

import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def build_dataloader(cfg, dataset_py="lerobot_datasets"):
    """Build the LeRobot dataloader used by the released ForeAct recipes."""
    if dataset_py != "lerobot_datasets":
        raise ValueError(
            f"Unsupported dataset backend {dataset_py!r}; "
            "ForeAct release recipes use 'lerobot_datasets'."
        )

    from foreact.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset = get_vla_dataset(
        data_cfg=vla_dataset_cfg,
        balance_dataset_weights=vla_dataset_cfg.get(
            "balance_dataset_weights", False
        ),
        balance_trajectory_weights=vla_dataset_cfg.get(
            "balance_trajectory_weights", False
        ),
    )

    num_workers = int(vla_dataset_cfg.get("num_workers", 4))
    dataloader_kwargs = {
        "batch_size": vla_dataset_cfg.per_device_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(
            vla_dataset_cfg.get("persistent_workers", True)
        )
        dataloader_kwargs["prefetch_factor"] = int(
            vla_dataset_cfg.get("prefetch_factor", 2)
        )

    dataloader = DataLoader(vla_dataset, **dataloader_kwargs)
    if not dist.is_initialized() or dist.get_rank() == 0:
        vla_dataset.save_dataset_statistics(
            Path(cfg.output_dir) / "dataset_statistics.json"
        )
    return dataloader
