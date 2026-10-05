from omegaconf import OmegaConf

from foreact.model.modules.world_model.CosmoPredict2 import (
    DEFAULT_PROMPT_TEMPLATE,
    build_prompts,
)


def test_cosmos_prompt_matches_t_plus_one_depth_contract():
    config = OmegaConf.create({"prompt": {"template": DEFAULT_PROMPT_TEMPLATE}})
    assert build_prompts(["  pick   up\n the cup "], config) == [
        "Robot task: pick up the cup. Predict the depth of the next observation "
        "frame and the next 16 robot actions."
    ]


def test_cosmos_prompt_rejects_empty_instruction():
    try:
        build_prompts(["  "], OmegaConf.create({}))
    except ValueError as error:
        assert "cannot be empty" in str(error)
    else:
        raise AssertionError("Expected an empty instruction to be rejected")
