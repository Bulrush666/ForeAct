# LIBERO reported protocol

The reported 96.55% result uses the 100k checkpoint, seed 42 for the simulator
and policy RNGs, bfloat16 inference, 10 ActionDiT denoising steps, and an
eight-action execution chunk without action ensembling.

| Suite | Successes | Rate |
| --- | ---: | ---: |
| LIBERO-Spatial | 492/500 | 98.4% |
| LIBERO-Object | 494/500 | 98.8% |
| LIBERO-Goal | 486/500 | 97.2% |
| LIBERO-10 | 459/500 | 91.8% |
| **Overall** | **1931/2000** | **96.55%** |

Each suite contains ten tasks with 50 fixed initial states per task. Evaluation
uses only `agentview_image`, rotates it by 180 degrees, resizes it to 224x224,
and sends neither wrist images nor proprioceptive state. The policy uses
`unnorm_key=franka`; the gripper output is mapped with
`1 - 2 * (open > 0.5)`.

Episodes begin with ten no-op stabilization steps. The action budgets are 220,
280, 300, and 520 steps for Spatial, Object, Goal, and LIBERO-10. The recorded
run used OSMesa rendering and a persistent policy server within each suite, with
fresh simulator processes for individual episodes. Runtime errors may be retried
with the same task and initial state; normal policy failures remain failures.

The compact public evaluator matches the observation, action, seed, and model
settings. It creates environments inside a conventional suite process rather
than reproducing the original external per-episode process scheduler, so raw
third-party reruns should report their exact launcher and environment versions.

The full machine-readable record is
`configs/evaluation/libero.yaml`. Checkpoints and raw
status/video files are not part of the source release.
