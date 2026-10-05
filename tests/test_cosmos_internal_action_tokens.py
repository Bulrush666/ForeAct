import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusers import CosmosTransformer3DModel

from foreact.model.framework.WM4A.CosmoPredict2GR00T import (
    CosmoPredict2GR00TDefaultConfig,
    CosmoPredict2_GR00T,
)
from foreact.model.framework.share_tools import merge_framework_config
from foreact.model.modules.world_model.CosmoPredict2 import (
    GlobalTokenSeedDepthHead,
    _CosmoPredict2_Interface,
)


def _build_tiny_wrapper() -> _CosmoPredict2_Interface:
    wrapper = _CosmoPredict2_Interface.__new__(_CosmoPredict2_Interface)
    nn.Module.__init__(wrapper)
    wrapper.transformer = CosmosTransformer3DModel(
        in_channels=5,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=12,
        num_layers=2,
        mlp_ratio=2.0,
        text_embed_dim=16,
        adaln_lora_dim=8,
        max_size=(4, 8, 8),
        patch_size=(1, 2, 2),
        concat_padding_mask=True,
        extra_pos_embed_type="learnable",
    )
    wrapper.use_internal_depth_tokens = True
    wrapper.num_internal_depth_tokens = 8
    wrapper.learnable_depth_tokens = nn.Parameter(torch.randn(1, 8, 24) * 0.02)
    wrapper.depth_head = GlobalTokenSeedDepthHead(
        hidden_size=24,
        patch_dim=wrapper.transformer.proj_out.out_features,
        num_tokens=8,
        head_dim=16,
        coord_dim=8,
        seed_grid=(2, 4),
        num_res_blocks=1,
        use_vae_seed_fusion=True,
        vae_channels=4,
        vae_downsample_factor=2,
    )
    wrapper.depth_backbone_gradient_scale = 0.01
    return wrapper


def _inputs(batch_size=2):
    return {
        "hidden_states": torch.randn(batch_size, 4, 1, 4, 8),
        "timestep": torch.zeros(batch_size, dtype=torch.long),
        "encoder_hidden_states": torch.randn(batch_size, 5, 16),
        "attention_mask": torch.ones(batch_size, 5),
        "condition_mask": torch.ones(batch_size, 1, 1, 4, 8),
        "padding_mask": torch.zeros(1, 1, 4, 8),
    }


def test_global_seed_head_generates_native_dense_patch_grid():
    head = GlobalTokenSeedDepthHead(
        hidden_size=24,
        patch_dim=16,
        num_tokens=8,
        head_dim=16,
        coord_dim=8,
        seed_grid=(5, 9),
        num_res_blocks=1,
    )
    tokens = torch.randn(2, 8, 24, requires_grad=True)
    patches = head(tokens, output_grid=(20, 36))

    assert patches.shape == (2, 720, 16)
    patches.square().mean().backward()
    assert tokens.grad is not None
    assert tokens.grad.abs().sum() > 0


def test_every_output_quadrant_depends_on_all_global_tokens():
    head = GlobalTokenSeedDepthHead(
        hidden_size=24,
        patch_dim=16,
        num_tokens=8,
        head_dim=16,
        coord_dim=8,
        seed_grid=(5, 9),
        num_res_blocks=1,
    )
    tokens = torch.randn(1, 8, 24, requires_grad=True)
    patch_map = head(tokens, output_grid=(20, 36)).view(1, 20, 36, 16)
    quadrants = (
        patch_map[:, :10, :18],
        patch_map[:, :10, 18:],
        patch_map[:, 10:, :18],
        patch_map[:, 10:, 18:],
    )

    for quadrant in quadrants:
        token_grad = torch.autograd.grad(
            quadrant.square().mean(), tokens, retain_graph=True
        )[0]
        per_token_grad = token_grad.abs().sum(dim=-1)
        assert torch.all(per_token_grad > 0)


def test_vae_seed_fusion_is_zero_initialized_and_learns_a_residual():
    head = GlobalTokenSeedDepthHead(
        hidden_size=24,
        patch_dim=16,
        num_tokens=8,
        head_dim=16,
        coord_dim=8,
        seed_grid=(2, 4),
        num_res_blocks=1,
        use_vae_seed_fusion=True,
        vae_channels=4,
        vae_downsample_factor=2,
    )
    tokens = torch.randn(2, 8, 24)
    first_latents = torch.randn(2, 4, 1, 4, 8)
    second_latents = torch.randn(2, 4, 1, 4, 8)

    first_output = head(tokens, output_grid=(2, 4), vae_latents=first_latents)
    second_output = head(tokens, output_grid=(2, 4), vae_latents=second_latents)

    # Zero initialization preserves the query-only behavior exactly even when
    # the input VAE latent changes.
    assert torch.equal(first_output, second_output)
    assert torch.count_nonzero(head.vae_seed_out.weight) == 0

    first_output.square().mean().backward()
    assert head.vae_seed_out.weight.grad is not None
    assert head.vae_seed_out.weight.grad.abs().sum() > 0

    # Once the residual projection learns non-zero weights, spatial VAE input
    # changes must affect the predicted depth patches.
    with torch.no_grad():
        head.vae_seed_out.weight.fill_(0.01)
    first_output = head(tokens, output_grid=(2, 4), vae_latents=first_latents)
    second_output = head(tokens, output_grid=(2, 4), vae_latents=second_latents)
    assert not torch.equal(first_output, second_output)


def test_global_tokens_are_distinct_and_have_neutral_rope():
    wrapper = _build_tiny_wrapper()
    pairwise_distance = torch.pdist(wrapper.learnable_depth_tokens[0])
    assert torch.all(pairwise_distance > 0)

    cos, sin = wrapper._depth_token_rotary_emb(_inputs()["hidden_states"])
    assert cos.shape[0] == 8
    assert torch.equal(cos, torch.ones_like(cos))
    assert torch.equal(sin, torch.zeros_like(sin))


def test_depth_tokens_traverse_cosmos_and_receive_scaled_gradients():
    wrapper = _build_tiny_wrapper()
    sample, visual_features, depth_features, depth_latents = (
        wrapper._forward_with_internal_depth_tokens(**_inputs())
    )

    assert sample.shape == (2, 4, 1, 4, 8)
    assert visual_features.shape == (2, 8, 24)
    assert depth_features.shape == (2, 8, 24)
    assert depth_latents.shape == (2, 4, 1, 4, 8)

    depth_latents.square().mean().backward()
    assert wrapper.learnable_depth_tokens.grad is not None
    assert wrapper.learnable_depth_tokens.grad.abs().sum() > 0
    assert wrapper.transformer.transformer_blocks[0].attn1.to_q.weight.grad is not None
    assert wrapper.transformer.transformer_blocks[-1].attn1.to_q.weight.grad is not None


def test_full_attention_connects_depth_tokens_to_dense_action_features():
    wrapper = _build_tiny_wrapper()
    _, visual_features, _, _ = wrapper._forward_with_internal_depth_tokens(**_inputs())
    visual_features.square().mean().backward()
    token_grad = wrapper.learnable_depth_tokens.grad
    assert token_grad is not None
    assert torch.count_nonzero(token_grad) > 0


def test_train_config_enables_eight_depth_observers():
    config = merge_framework_config(
        CosmoPredict2GR00TDefaultConfig,
        OmegaConf.load("train.yaml"),
    )
    token_config = config.framework.world_model.internal_depth_tokens
    assert token_config.enabled is True
    assert token_config.num_tokens == 8
    assert token_config.position_mode == "global"
    assert list(token_config.seed_grid) == [10, 18]
    assert token_config.coord_dim == 128
    assert token_config.num_res_blocks == 2
    assert token_config.vae_seed_fusion.enabled is True
    assert token_config.vae_seed_fusion.input_channels == 16
    assert token_config.vae_seed_fusion.downsample_factor == 4
    assert config.depth.loss_weight == 0.01
    assert config.depth.backbone_gradient_scale == 0.01


def test_action_interface_always_selects_dense_visual_features():
    framework = CosmoPredict2_GR00T.__new__(CosmoPredict2_GR00T)
    nn.Module.__init__(framework)

    class Outputs:
        pass

    outputs = Outputs()
    outputs.action_token_features = torch.randn(2, 8, 8)
    outputs.depth_token_features = torch.randn(2, 8, 8)
    dense_features = torch.randn(2, 720, 8)
    outputs.hidden_states = (dense_features,)

    selected = framework._action_conditioning_features(outputs)
    assert selected is dense_features
    assert selected.shape == (2, 720, 8)
