# GitHub release and privacy checklist

ForeAct keeps experiment settings in version control while keeping machine
locations, credentials, datasets, model weights, generated depth caches and
evaluation outputs local. Run this checklist before every public push.

## Portable path contract

Tracked configurations use repository-relative defaults:

| Resource | Default | Runtime override |
| --- | --- | --- |
| Python executable | `python3` from the active environment | `PYTHON_BIN` |
| Cosmos-Predict2 | `pretrained/cosmos2` | `MODEL_DIR` or `COSMOS2_MODEL_DIR` |
| LeRobot data | `data/...` | `DATA_ROOT_DIR` |
| MoGe depth cache | `data/depth/...` | `DEPTH_CACHE_ROOT` |
| Training outputs | `outputs/` | `OUTPUT_ROOT` |
| SimplerEnv | `third_party/SimplerEnv` | `SIMPLER_ENV_ROOT` |
| SimplerEnv Python | `python3` | `SIM_PYTHON` |

Use `.env.example` as a reference, but export variables in the shell or use an
untracked local launcher. The project does not automatically source `.env`.
Never replace portable defaults with a personal path in a tracked file.

MoGe configurations live below `tools/moge3/configs/`, so their `../../../`
paths resolve back to the repository root. Evaluation YAML files are historical
parameter records; they are not training configs and are not automatically
loaded by the current evaluation CLI.

## Files that must stay out of Git

- `.env`, API tokens, SSH credentials and private keys;
- `data/`, `local_deploy/data/` and any symlink into a private dataset;
- `pretrained/`, `checkpoints/`, `outputs/`, `wandb/` and depth caches;
- `.codex_backups/`, local recovery archives and machine-specific YAML files;
- licensed datasets or model weights that cannot be redistributed.

These paths are covered by `.gitignore`. Do not bypass the rules with
`git add -f`. Check a symlink before adding it: a relative-looking link can still
resolve outside the repository and disclose a server layout.

## Public evaluation records

`configs/evaluation/` retains reproducibility parameters and aggregate counts,
but removes author usernames and filesystem locations. Current records are:

- LIBERO: 100k checkpoint, denoise 10, seed/action seed 42, execution horizon 8,
  no action ensemble, 1931/2000 (96.55%).
- SimplerEnv-Bridge: 50k checkpoint, denoise 4, seed/action seed 7, ensemble
  horizon 13, alpha 0.1, action scale 1.091, 61/96 (63.54%).

Report these as single evaluation records, not multi-seed confidence estimates.
The released repository does not include the checkpoints or raw videos.

## Safety notes for inherited scripts

Legacy SimplerEnv launchers now require an explicit checkpoint argument and use
`FOREACT_PYTHON`, `SIM_PYTHON` and `SIMPLER_ENV_ROOT` instead of cluster paths.
The checkpoint cleanup helper defaults to dry-run and refuses an empty, root or
home-directory target. Review every command before running it on shared storage.

## Pre-push commands

```bash
python3 scripts/check_public_release.py
git status --short
git diff --check
git ls-files -s | awk '$1 == "120000" {print $4}'
```

Also review the full Git history with a secret scanner such as `gitleaks` before
making an existing repository public. Cleaning the current working tree does not
remove a secret or private path from earlier commits. If history contains one,
rotate the credential first and rewrite history with an appropriate tool.

Finally, verify `LICENSE`, `CITATION.cff`, third-party notices and model/dataset
licenses. Required upstream copyright and license notices must remain intact.
