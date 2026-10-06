# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the terms in the repository LICENSE file.
# Implemented by [Jinhui YE / HKUST University] in [2025].

import argparse
import logging
import random
import socket

import numpy as np
import torch

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args) -> None:
    """Build the policy wrapper and start the websocket server.

    The wrapper now owns un-normalization + chunk_size discovery so that all
    eval clients (LIBERO / SimplerEnv / etc.) just need to forward `examples`
    and consume already-unnormalized actions from the response.
    """
    config_overrides = getattr(args, "config_override", [])
    seed = getattr(args, "seed", None)
    action_seed = getattr(args, "action_seed", None)
    seed = int(seed) if isinstance(seed, int) else None
    action_seed = int(action_seed) if isinstance(action_seed, int) else None
    if seed is not None:
        _seed_everything(seed)
    if config_overrides:
        override_keys = [item.split("=", 1)[0] for item in config_overrides]
        logging.info("Applying config override keys: %s", override_keys)
    wrapper_kwargs = {
        "ckpt_path": args.ckpt_path,
        "device": "cuda",
        "use_bf16": args.use_bf16,
        "config_overrides": config_overrides,
    }
    effective_action_seed = action_seed if action_seed is not None else seed
    if effective_action_seed is not None:
        wrapper_kwargs["action_seed"] = effective_action_seed
    wrapper = PolicyServerWrapper(**wrapper_kwargs)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # =========================================================================
    # !!! TRAIN / TEST CONSISTENCY — READ BEFORE SERVING !!!
    # -------------------------------------------------------------------------
    # This server replays the *training-time* observation contract. The eval
    # client MUST feed observations exactly as the model saw them at TRAIN time,
    # otherwise the success rate silently drops (no error is raised):
    #   - state   : is proprioceptive state used? (use_state) and its dim/order
    #   - img size: resize / crop resolution (e.g. 224x224)
    #   - img num : how many camera views are fed
    #   - img order: the ordering of those camera views
    #   - action normalization: unnorm_key / dataset stats must match training
    # `wrapper.metadata` (logged below and sent at handshake) exposes
    # action_chunk_size / state_keys / action_keys — cross-check these against
    # the training config used to produce `args.ckpt_path`.
    # =========================================================================
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s — verify eval observations "
        "(state / image size / image count / image order / action normalization) match "
        "the training config. metadata=%s",
        args.ckpt_path,
        wrapper.metadata,
    )

    # start websocket server; wrapper.metadata is sent at handshake.
    server = WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    logging.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Address to bind. Use 0.0.0.0 only on a trusted network.",
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed global RNGs and action/VAE inference streams for reproducible evaluation.",
    )
    parser.add_argument(
        "--action-seed",
        "--action_seed",
        dest="action_seed",
        type=int,
        default=None,
        help="Override the action/VAE inference seed; defaults to --seed.",
    )
    parser.add_argument("--idle_timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OmegaConf dotlist override applied before model construction.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
