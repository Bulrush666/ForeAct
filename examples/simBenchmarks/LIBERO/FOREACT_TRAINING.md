# ForeAct LIBERO integration

This document defines the LIBERO data and supervision contract used by ForeAct.

## Training contract

- Config: `configs/train/libero.yaml`; mixture `libero_all_v30_main`.
- Data: `data/libero/lerobot_v30` by default (override with `DATA_ROOT_DIR`), containing four `libero_{object,goal,spatial,10}_no_noops_lerobot` suites.
- Sample the four suites equally; sample episodes equally within a suite, then
  uniformly sample a timestep in that episode (not frame-proportional sampling).
- Input: main camera `observation.images.image[t]` only, no wrist or proprioceptive state.
- Actions: stored 7-D commands at `[t, ..., t+7]`, repeat-last at episode end.
  `action_mode: abs` preserves the stored delta-EE commands; it does NOT convert
  them to absolute poses. Normalize the six motion dimensions with min/max;
  gripper is not additionally normalized.
- RGB: ForeAct's shared preprocessing, resize to 224x224 without cropping,
  then the existing Cosmos preprocessing to 320x576. ForeAct uses Pillow
  bilinear for the first resize; the reference's bare Pillow resize uses
  bicubic. We preserve ForeAct's train/deploy shared helper rather than change
  real-world preprocessing globally. No extra flip is applied to stored RGB.
- Depth: main-camera MoGe target `min(t+1, episode_length-1)`, NOT t+8.
  The model uses eight geometry tokens, VAE seed fusion and weak depth loss.

## Depth cache layout (generated separately)

```text
<DEPTH_CACHE_ROOT>/cache/
  libero_goal_no_noops_lerobot/chunk-000/episode_000000.npz
  libero_object_no_noops_lerobot/chunk-000/episode_000000.npz
  libero_spatial_no_noops_lerobot/chunk-000/episode_000000.npz
  libero_10_no_noops_lerobot/chunk-000/episode_000000.npz
```

NPZ: `depth` and `valid_mask` [episode_length,320,576], `available`
[episode_length], scalar `episode_index`, scalar `horizon=1`. These are metric
depth arrays, NOT color-mapped PNGs. Index the arrays by original episode frame
number; do not pack only target frames densely. All t+1 targets (including the
last frame reused for padding) must be available. Never set a single
`depth_cache_dataset_name` for this mixture: episode IDs repeat between suites.
Preflight checks all cache identities, array headers and target availability;
it does not validate numerical depth quality or prove which camera generated a
cache. The offline producer must use the main camera and matching RGB geometry.

The default depth root was not present on this server when integrated. Point
`DEPTH_CACHE_ROOT` at the complete caches on the actual training server.
Missing caches fail before launching GPU workers, not midway through training.

## Commands (from ForeAct root)

```bash
# Read-only RGB metadata/path checks; does not claim depth is ready.
python3 scripts/check_libero_data.py --metadata-only

# Print launch command, allocate no GPU, write no run directory.
bash scripts/train_libero.sh --dry-run

# Actual training after generating/moving all depth caches.
DEPTH_CACHE_ROOT=/your/libero_depth_next1_moge3 bash scripts/train_libero.sh

# Example: four GPUs, batch 8 each, effective batch 32.
GPU_IDS=0,1,2,3 PER_DEVICE_BATCH_SIZE=8 DEPTH_CACHE_ROOT=/your/cache_root bash scripts/train_libero.sh

# Regression tests, CPU only.
python3 -m unittest discover -s tests -p test_libero_integration.py -v
```

Default: 8 GPUs, batch 8/GPU, accumulation 1, 100k steps, 5k warmup/save,
seed 42, ZeRO-2 BF16, offline W&B. This is a configuration, not a measured
memory-capacity recommendation. Override `CONFIG`, `PYTHON_BIN`, `MODEL_DIR`,
`DATA_ROOT_DIR`, `DEPTH_CACHE_ROOT`, `OUTPUT_ROOT`, `GPU_IDS`,
`PER_DEVICE_BATCH_SIZE`, `NUM_WORKERS`, `MAIN_PROCESS_PORT`, `MAX_STEPS`, `SEED`.
Runtime values are copied into the effective model/data/trainer config before
checking it; the saved `launch_config.yaml` is exactly what training consumes.
Fresh runs only; existing run directories are not overwritten. No automatic
model/data downloads. Logs/checkpoints go into the timestamped ForeAct/log run.

## Video alignment fix

LeRobot v3 partitions action parquet and each camera's videos independently.
The loader now uses `videos/<camera>/chunk_index,file_index` for videos and
`data/chunk_index,file_index` only for parquet. Video timestamps retain each
episode's camera-specific `from_timestamp`; actions are filtered by the actual
episode ID. Existing v2 behavior and legacy missing-camera-index fallback are
unchanged. The strict LIBERO preflight requires the camera index metadata, so
this new entry cannot silently use the fallback for incomplete LIBERO data.
