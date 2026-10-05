# Local deployment

This directory contains optional helpers for preparing a local Bridge overlay,
downloading the pretrained Cosmos-Predict2 components and launching a policy
server. Large datasets and model weights remain untracked.

## Verify paths

Export the locations described in `.env.example`, then run:

```bash
python local_deploy/verify_deployment.py
```

## Dataset smoke test

```bash
PYTHON_BIN=/path/to/foreact-env/bin/python \
bash local_deploy/run_dataset_smoke.sh
```

## Policy server

```bash
FOREACT_PYTHON=/path/to/foreact-env/bin/python \
COSMOS2_ACTION_CKPT=/path/to/checkpoint.pt \
bash local_deploy/run_policy_server.sh
```

The simulator should run in its own environment. Use
`local_deploy/start_simpler_env_render.py` only after installing SimplerEnv and
its rendering dependencies.
