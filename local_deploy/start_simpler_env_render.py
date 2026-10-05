import os
import random

import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

from examples.simBenchmarks.SimplerEnv.eval_files.custom_argparse import get_args
from examples.simBenchmarks.SimplerEnv.eval_files.model2simpler_interface import ModelClient


if __name__ == "__main__":
    seed = int(os.environ.get("SEED", "42"))
    random.seed(seed)
    np.random.seed(seed % (2**32))

    args = get_args()
    render_device = os.environ.get("SAPIEN_RENDER_DEVICE", "cuda:2")
    env_kwargs = dict(args.additional_env_build_kwargs or {})
    env_kwargs["renderer_kwargs"] = {
        "offscreen_only": True,
        "device": render_device,
    }
    args.additional_env_build_kwargs = env_kwargs

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
