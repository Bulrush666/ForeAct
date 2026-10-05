# LIBERO evaluation

ForeAct evaluates LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-10 with
the main camera only. The policy server and simulator use separate Python
environments.

## Policy server

Start one server per suite to reproduce the recorded RNG boundary:

```bash
FOREACT_PYTHON=/path/to/foreact-env/bin/python \
CKPT=/absolute/path/to/steps_100000_pytorch_model.pt \
bash examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh
```

The launcher defaults to seed 42, bfloat16, and 10 ActionDiT denoising steps.

## Simulator

In a second shell, select one suite and run 50 trials per task:

```bash
LIBERO_HOME=/absolute/path/to/LIBERO \
LIBERO_PYTHON=/path/to/libero-env/bin/python \
CKPT=/absolute/path/to/steps_100000_pytorch_model.pt \
TASK_SUITE_NAME=libero_spatial \
bash examples/simBenchmarks/LIBERO/eval_files/eval_libero.sh
```

Valid suites are `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`. Restart the policy server before the next suite. The evaluator uses
the main `agentview_image`, a 180-degree rotation, 224x224 client resize, no
proprioceptive state, no action ensemble, and an eight-action execution chunk.

The complete reported protocol and success counts are recorded in
`configs/evaluation/libero.yaml`. That YAML is an
auditable experiment record, not a training configuration.
