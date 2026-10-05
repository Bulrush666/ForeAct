def get_world_model(config):
    """Factory for world model backends.

    Routes to the correct world-model wrapper based on
    ``config.framework.world_model.base_wm`` (or falls back to
    ``config.framework.qwenvl.base_vlm`` for backward compatibility).

    Every world-model wrapper exposes:
      - ``forward(**kwargs)`` → model outputs with hidden_states
      - ``build_inputs(images, instructions)`` → dict of tensors
      - ``generate(**kwargs)`` → generation (optional)
    """

    # Prefer explicit world_model config; fall back to qwenvl for compat
    wm_cfg = config.framework.get("world_model", None)
    if wm_cfg is not None:
        wm_name = wm_cfg.get("base_wm", "")
    else:
        wm_name = config.framework.qwenvl.base_vlm

    if "cosmos-predict2" in wm_name.lower() or wm_name.lower().rstrip("/").endswith("cosmos2"):
        from .CosmoPredict2 import _CosmoPredict2_Interface

        return _CosmoPredict2_Interface(config)
    else:
        raise NotImplementedError(
            f"World model {wm_name!r} is not supported by this ForeAct checkout; "
            "only the Cosmos-Predict2 backend is retained."
        )
