# SimplerEnv-Bridge evaluation

ForeAct evaluates Spoon on Towel, Carrot on Plate, Stack Blocks, and Eggplant in
Basket. The policy server and SimplerEnv simulator use separate Python
environments.

## Policy server

```bash
FOREACT_PYTHON=/path/to/foreact-env/bin/python \
CKPT=/absolute/path/to/steps_50000_pytorch_model.pt \
bash examples/simBenchmarks/SimplerEnv/eval_files/run_policy_server.sh
```

## Simulator

```bash
SIM_PYTHON=/path/to/simpler-env/bin/python \
SIMPLER_ENV_ROOT=/absolute/path/to/SimplerEnv \
CKPT=/absolute/path/to/steps_50000_pytorch_model.pt \
bash examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.sh
```

For the exact reported RNG boundary, evaluate one task at a time and restart the
policy server before each invocation. Set `TASK` to `spoon`, `carrot`, `cube`, or
`eggplant`:

```bash
TASK=spoon SIM_PYTHON=/path/to/simpler-env/bin/python \
SIMPLER_ENV_ROOT=/absolute/path/to/SimplerEnv \
CKPT=/absolute/path/to/steps_50000_pytorch_model.pt \
bash examples/simBenchmarks/SimplerEnv/eval_files/start_simpler_env.sh
```

The launchers default to the reported setting: seed 7, four ActionDiT denoising
steps, action scale 1.091, ensemble horizon 13, ensemble alpha 0.1, control
frequency 5, simulation frequency 500, and 24 episodes per task. The complete
record is `configs/evaluation/simplerenv_bridge.yaml`.
