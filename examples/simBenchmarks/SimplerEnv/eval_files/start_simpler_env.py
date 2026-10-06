import os

import numpy as np
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

# from IPython import embed; embed()
from examples.simBenchmarks.SimplerEnv.eval_files.custom_argparse import get_args
from examples.simBenchmarks.SimplerEnv.eval_files.model2simpler_interface import ModelClient


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    model = ModelClient(
        policy_ckpt_path=args.ckpt_path,  # to get unnormalization stats
        policy_setup=args.policy_setup,
        host=args.host,
        port=args.port,
        action_scale=args.action_scale,
        action_ensemble=not args.disable_action_ensemble,
        action_ensemble_horizon=args.action_ensemble_horizon,
        adaptive_ensemble_alpha=args.adaptive_ensemble_alpha,
        num_ddim_steps=args.num_ddim_steps,
        cfg_scale=1.5,  # cfg from 1.5 to 7 also performs well
    )

    # policy model creation; update this if you are using a new policy model
    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
