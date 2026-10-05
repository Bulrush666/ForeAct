# Policy server

ForeAct serves actions over a lightweight WebSocket API. The server loads one
checkpoint, applies the checkpoint's action unnormalization, and returns a full
action chunk to the benchmark client.

## Start

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/checkpoint.pt \
  --host 127.0.0.1 \
  --port 10093 \
  --use_bf16
```

Useful reproducibility options are `--seed`, `--action-seed`, and repeatable
`--config_override KEY=VALUE` arguments. For example:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path /absolute/path/to/checkpoint.pt \
  --seed 7 --action-seed 7 \
  --config_override framework.action_model.num_inference_timesteps=4
```

The benchmark launchers under `examples/simBenchmarks/` set the reported
evaluation values for LIBERO and SimplerEnv.

## Security

The server binds to `127.0.0.1` by default and does not implement authentication
or transport encryption. Bind to `0.0.0.0` only inside a trusted network or
behind an authenticated proxy/firewall. Request bodies are limited to 64 MiB,
and internal tracebacks are logged server-side rather than returned to clients.

## Protocol

At connection time, the server sends metadata including `action_chunk_size`,
`available_unnorm_keys`, and the action/state key contract. Inference requests
contain `examples` and, for multi-dataset checkpoints, an `unnorm_key`.
Successful responses expose unnormalized actions at
`response["data"]["actions"]` with shape `[batch, horizon, action_dim]`.
