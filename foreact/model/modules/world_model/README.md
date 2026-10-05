# Cosmos-Predict2 world-model adapter

`CosmoPredict2.py` wraps the pretrained Cosmos-Predict2 Video2World components
used by ForeAct. The adapter loads the frozen text encoder and video VAE together
with the trainable video DiT, prepares the current RGB observation and language
instruction, and returns dense visual states plus the learned geometry-token
states.

The public ForeAct checkout supports this backend through:

```python
from foreact.model.modules.world_model import get_world_model

world_model = get_world_model(config)
inputs = world_model.build_inputs(images, instructions)
outputs = world_model(**inputs, output_hidden_states=True, return_dict=True)
```

The factory reads `config.framework.world_model.base_wm`. The path must point to
a local Cosmos-Predict2 directory containing the tokenizer, text encoder,
transformer, VAE, and scheduler subdirectories.

During a forward pass, image patches and eight geometry tokens are processed by
the same unmasked transformer. The action head consumes the final dense visual
states. During training, the geometry-token states feed the lightweight depth
readout; this readout is not required for action inference.
