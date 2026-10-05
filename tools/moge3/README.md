# Offline MoGe-3 depth preprocessing

This directory builds metric-depth caches used only during ForeAct training.
MoGe-3, FlexGEMM, model weights, datasets, and generated caches are external
assets and are not included in this repository.

## Cache contract

For an observation at frame `t`, the public Bridge and LIBERO configurations
store the target at `min(t + 1, episode_length - 1)`. The cache keeps original
episode frame indices and writes one compressed NPZ file per episode:

```text
<depth-root>/cache/<dataset-or-suite>/chunk-NNN/episode_NNNNNN.npz
```

ForeAct reads the main camera only. RGB observations are resized to `224x224`
before the model-specific `320x576` preprocessing stage.

## Configuration

Edit the path fields before launching:

- `paths.python_bin`
- `paths.moge_source_dir`
- `paths.flex_gemm_source_dir`
- `paths.checkpoint`
- `paths.dataset_root`
- `paths.output_root`

The default public configurations are:

| Dataset | Configuration | Launcher |
| --- | --- | --- |
| Bridge | `configs/train.yaml` | `launch_precompute_4gpu.py` |
| LIBERO v3 | `configs/libero_next1.yaml` | `launch_precompute_libero_8gpu.py` |
| Fractal / RT-1 | `configs/fractal_next1.yaml` | `launch_precompute_fractal_8gpu.py` |

The configured output roots match the training defaults under `data/depth/`.
GPU IDs, worker count, and batch size are examples and must be adjusted to the
available hardware. Keep `distributed.num_workers` equal to the number of
entries in `distributed.gpu_ids`.

## Bridge

```bash
MOGE_PY=/path/to/moge3-env/bin/python
"$MOGE_PY" tools/moge3/launch_precompute_4gpu.py \
  --config tools/moge3/configs/train.yaml --dry-run
"$MOGE_PY" tools/moge3/launch_precompute_4gpu.py \
  --config tools/moge3/configs/train.yaml
"$MOGE_PY" tools/moge3/validate_depth_cache.py \
  --config tools/moge3/configs/train.yaml
```

`configs/bridge_reuse.yaml` is an optional migration recipe for deriving a
`t+1` cache from an existing `t+16` cache. Set `paths.reuse_cache_root` to the
old cache and use `launch_precompute_reuse_8gpu.py`. It always writes to a new
output root and never overwrites the source cache.

## LIBERO

```bash
MOGE_PY=/path/to/moge3-env/bin/python
"$MOGE_PY" tools/moge3/check_libero_setup.py \
  --config tools/moge3/configs/libero_next1.yaml
"$MOGE_PY" tools/moge3/launch_precompute_libero_8gpu.py \
  --config tools/moge3/configs/libero_next1.yaml --dry-run
"$MOGE_PY" tools/moge3/launch_precompute_libero_8gpu.py \
  --config tools/moge3/configs/libero_next1.yaml
"$MOGE_PY" tools/moge3/validate_libero_depth_cache.py \
  --config tools/moge3/configs/libero_next1.yaml
```

After cache generation, validate the training-side data contract:

```bash
python scripts/check_libero_data.py
bash scripts/train_libero.sh --dry-run
```

## Lightweight tests

```bash
python -m unittest discover -s tools/moge3/tests -v
```
