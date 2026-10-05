# Reported evaluation records

The YAML files in this directory are auditable records of reported runs. They
are not training configurations and are not loaded directly by the benchmark
clients. The launchers under `examples/simBenchmarks/` expose the corresponding
runtime values through environment variables and command-line arguments.

| Benchmark | Record | Checkpoint step | Denoising steps | Seed | Result |
| --- | --- | ---: | ---: | ---: | ---: |
| LIBERO | `libero.yaml` | 100k | 10 | 42 | 1931/2000 (96.55%) |
| SimplerEnv-Bridge | `simplerenv_bridge.yaml` | 50k | 4 | 7 | 61/96 (63.54%) |

The SimplerEnv record contains 21/24 Spoon, 13/24 Carrot, 3/24 Cube, and
24/24 Eggplant successes. This is an individual fixed-seed evaluation, not a
multi-seed confidence estimate.

Checkpoint files and raw evaluation videos are not distributed with the source
repository. Record the SHA-256 digest of any externally obtained checkpoint
before comparing results.
