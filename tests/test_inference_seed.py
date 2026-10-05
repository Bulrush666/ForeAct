import types
import unittest
from unittest import mock

import torch

from foreact.model.framework.WM4A.CosmoPredict2GR00T import CosmoPredict2_GR00T
from foreact.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from foreact.model.modules.world_model.CosmoPredict2 import _CosmoPredict2_Interface


class _StopSampling(Exception):
    pass


class InferenceSeedTest(unittest.TestCase):
    def test_action_head_forwards_generator_to_initial_noise(self):
        generator = torch.Generator().manual_seed(42)
        fake_head = types.SimpleNamespace(action_horizon=16, action_dim=7)
        embeddings = torch.zeros(1, 2, 4)

        with mock.patch(
            "foreact.model.modules.action_model.GR00T_ActionHeader.torch.randn",
            side_effect=_StopSampling,
        ) as randn:
            with self.assertRaises(_StopSampling):
                FlowmatchingActionHead.predict_action(
                    fake_head, embeddings, generator=generator
                )

        self.assertIs(randn.call_args.kwargs["generator"], generator)

    def test_cosmos_vae_forwards_generator_to_latent_sample(self):
        generator = torch.Generator().manual_seed(43)
        distribution = mock.Mock()
        distribution.sample.return_value = torch.zeros(1, 16, 1, 1, 1)
        vae = mock.Mock()
        vae.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        vae.dtype = torch.float32
        vae.config.latents_mean = None
        vae.encode.return_value.latent_dist = distribution
        interface = types.SimpleNamespace(
            vae=vae,
            video_processor=mock.Mock(),
        )
        interface.video_processor.preprocess_video.return_value = torch.zeros(
            1, 3, 1, 8, 8
        )

        _CosmoPredict2_Interface._encode_images(
            interface, images=[[object()]], generator=generator
        )

        distribution.sample.assert_called_once_with(generator=generator)

    def test_reseeding_replays_both_rng_streams(self):
        framework = types.SimpleNamespace(action_model=torch.nn.Linear(1, 1))

        CosmoPredict2_GR00T.set_inference_seed(framework, 42)
        first_action = torch.randn(8, generator=framework._action_generator)
        first_vae = torch.randn(8, generator=framework._vae_generator)
        CosmoPredict2_GR00T.set_inference_seed(framework, 42)
        second_action = torch.randn(8, generator=framework._action_generator)
        second_vae = torch.randn(8, generator=framework._vae_generator)

        torch.testing.assert_close(first_action, second_action, rtol=0, atol=0)
        torch.testing.assert_close(first_vae, second_vae, rtol=0, atol=0)
        self.assertFalse(torch.equal(first_action, first_vae))


if __name__ == "__main__":
    unittest.main()
