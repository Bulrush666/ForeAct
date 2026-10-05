# Docker

ForeAct provides separate policy-serving and training images.

## Build

Run from the repository root:

```bash
docker build -f deployment/docker/Dockerfile.server -t foreact-server .
docker build -f deployment/docker/Dockerfile.train -t foreact-train .
```

Both builds accept `--build-arg PIP_INDEX_URL=<mirror>`; use `TORCH_INDEX_URL` to override the CUDA wheel index.

## Policy server

```bash
docker run --gpus all --rm -p 127.0.0.1:5678:5678 \
  -v /absolute/path/to/checkpoints:/checkpoints:ro \
  -v /absolute/path/to/cosmos2:/workspace/ForeAct/pretrained/cosmos2:ro \
  foreact-server \
  --ckpt_path /checkpoints/steps_50000_pytorch_model.pt \
  --host 0.0.0.0 --port 5678 --use_bf16 --idle_timeout -1
```

The host-side port is restricted to loopback in this example. The WebSocket
server has no authentication or TLS, so do not expose it directly to an
untrusted network.

## Training

```bash
docker run --gpus all --rm -it --shm-size 16g \
  -v /absolute/path/to/cosmos2:/workspace/ForeAct/pretrained/cosmos2:ro \
  -v /absolute/path/to/data:/workspace/ForeAct/data:ro \
  -v /absolute/path/to/outputs:/workspace/ForeAct/outputs \
  foreact-train bash
```

Inside the container, run `scripts/train_bridge.sh` or
`scripts/train_libero.sh`. Datasets, weights, and outputs are mounted at runtime
and are excluded from the image build context.

## Compose

```bash
export CHECKPOINT_DIR=/absolute/path/to/checkpoints
export COSMOS_DIR=/absolute/path/to/cosmos2
docker compose -f deployment/docker/docker-compose.yml up
```
