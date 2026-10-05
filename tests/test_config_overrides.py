import os
import stat
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import torch
from omegaconf import OmegaConf

from deployment.model_server import policy_wrapper, server_policy
from foreact.model.framework import base_framework


def _checkpoint_config(use_canonical_forward=True):
    return {
        "framework": {
            "name": "DummyFramework",
            "action_model": {
                "action_horizon": 8,
                "diffusion_model_cfg": {
                    "use_canonical_forward": use_canonical_forward,
                    "num_inference_timesteps": 4,
                },
            },
        },
        "trainer": {"pretrained_checkpoint": "original.pt"},
        "datasets": {"vla_data": {"data_mix": "dummy_mix"}},
    }


class _FakeFramework:
    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dict = state_dict
        self.loaded_strict = strict


class _FakeServerModel:
    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def set_inference_seed(self, seed):
        self.inference_seed = seed


class ConfigOverrideLoadTest(unittest.TestCase):
    def _load_with_config(self, config, overrides=None):
        captured = {}

        def fake_build_framework(cfg):
            captured["cfg"] = cfg
            return _FakeFramework()

        with (
            mock.patch.object(base_framework, "read_mode_config", return_value=(config, {"stats": {}})),
            mock.patch.object(base_framework, "build_framework", side_effect=fake_build_framework),
            mock.patch.object(torch, "load", return_value={}),
        ):
            model = base_framework.baseframework.from_pretrained(
                "/tmp/run/checkpoints/steps_1_pytorch_model.pt",
                config_overrides=overrides,
            )
        return model, captured["cfg"]

    def test_from_pretrained_without_overrides_preserves_checkpoint_config(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True))

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_false_override_resolves_to_bool_false_before_build_framework(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            ["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
        )

        resolved = OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward")
        self.assertIs(resolved, False)

    def test_override_has_precedence_over_checkpoint_yaml(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            ["framework.action_model.diffusion_model_cfg.num_inference_timesteps=6"],
        )

        self.assertEqual(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.num_inference_timesteps"),
            6,
        )

    def test_repeatable_overrides_are_accepted(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            (
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
                "framework.action_model.diffusion_model_cfg.num_inference_timesteps=6",
            ),
        )

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            False,
        )
        self.assertEqual(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.num_inference_timesteps"),
            6,
        )

    def test_duplicate_override_later_value_wins(self):
        _, cfg = self._load_with_config(
            _checkpoint_config(use_canonical_forward=True),
            [
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=true",
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            ],
        )

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            False,
        )

    def test_invalid_override_syntax_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Expected KEY=VALUE"):
            self._load_with_config(_checkpoint_config(), ["framework.action_model.bad_override"])

    def test_bare_string_config_overrides_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "sequence of KEY=VALUE strings"):
            self._load_with_config(
                _checkpoint_config(),
                "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            )

    def test_empty_string_config_overrides_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "sequence of KEY=VALUE strings"):
            self._load_with_config(_checkpoint_config(), "")

    def test_empty_list_config_overrides_is_valid_noop(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True), [])

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_empty_tuple_config_overrides_is_valid_noop(self):
        _, cfg = self._load_with_config(_checkpoint_config(use_canonical_forward=True), ())

        self.assertIs(
            OmegaConf.select(cfg, "framework.action_model.diffusion_model_cfg.use_canonical_forward"),
            True,
        )

    def test_unrelated_interpolation_is_not_resolved_when_override_applies(self):
        config = _checkpoint_config()
        config["run_root_dir"] = "/tmp/root"
        config["run_id"] = "run_a"
        config["output_dir"] = "${run_root_dir}/${run_id}"

        merged = base_framework.merge_config_overrides(
            config,
            ["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
        )

        self.assertEqual(merged["output_dir"], "${run_root_dir}/${run_id}")

    def test_checkpoint_config_file_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True)
            ckpt_path = ckpt_dir / "steps_1_pytorch_model.pt"
            ckpt_path.write_bytes(b"placeholder")
            config_path = run_dir / "config.yaml"
            config_text = textwrap.dedent("""
                framework:
                  name: DummyFramework
                  action_model:
                    action_horizon: 8
                    diffusion_model_cfg:
                      use_canonical_forward: true
                trainer:
                  pretrained_checkpoint: original.pt
                datasets:
                  vla_data:
                    data_mix: dummy_mix
                """).lstrip()
            config_path.write_text(config_text)
            (run_dir / "dataset_statistics.json").write_text("{}")

            with (
                mock.patch.object(base_framework, "build_framework", return_value=_FakeFramework()),
                mock.patch.object(torch, "load", return_value={}),
            ):
                base_framework.baseframework.from_pretrained(
                    str(ckpt_path),
                    config_overrides=["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"],
                )

            self.assertEqual(config_path.read_text(), config_text)


class PolicyServerOverrideTest(unittest.TestCase):
    def test_policy_server_wrapper_passes_config_overrides_to_from_pretrained(self):
        overrides = ["framework.action_model.diffusion_model_cfg.use_canonical_forward=false"]
        model_cfg = _checkpoint_config()
        norm_stats = {"key_a": {}, "key_b": {}}

        with (
            mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=_FakeServerModel()) as load,
            mock.patch.object(policy_wrapper, "read_mode_config", return_value=(model_cfg, norm_stats)),
        ):
            policy_wrapper.PolicyServerWrapper(
                ckpt_path="/tmp/model.pt",
                device="cpu",
                config_overrides=overrides,
            )

        load.assert_called_once_with("/tmp/model.pt", config_overrides=overrides)

    def test_policy_server_wrapper_applies_and_reports_action_seed(self):
        model = _FakeServerModel()
        model_cfg = _checkpoint_config()
        norm_stats = {"key_a": {}, "key_b": {}}

        with (
            mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=model),
            mock.patch.object(policy_wrapper, "read_mode_config", return_value=(model_cfg, norm_stats)),
        ):
            wrapper = policy_wrapper.PolicyServerWrapper(
                ckpt_path="/tmp/model.pt",
                device="cpu",
                action_seed=42,
            )

        self.assertEqual(model.inference_seed, 42)
        self.assertEqual(wrapper.metadata["action_seed"], 42)

    def test_policy_server_wrapper_metadata_uses_resolved_config(self):
        overrides = [
            "framework.action_model.action_horizon=12",
            "datasets.vla_data.data_mix=override_mix",
            "datasets.vla_data.obs_image_size=[128,128]",
        ]
        model_cfg = _checkpoint_config()
        norm_stats = {"key_a": {}, "key_b": {}}

        with (
            mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=_FakeServerModel()),
            mock.patch.object(policy_wrapper, "read_mode_config", return_value=(model_cfg, norm_stats)),
        ):
            wrapper = policy_wrapper.PolicyServerWrapper(
                ckpt_path="/tmp/model.pt",
                device="cpu",
                config_overrides=overrides,
            )

        self.assertEqual(wrapper.metadata["action_chunk_size"], 12)
        self.assertEqual(wrapper.metadata["training_data_mix"], "override_mix")
        self.assertEqual(wrapper.metadata["training_obs_image_size"], [128, 128])
        self.assertEqual(wrapper._model_cfg["framework"]["action_model"]["action_horizon"], 12)

    def test_policy_server_wrapper_rejects_bare_string_config_overrides(self):
        model_cfg = _checkpoint_config()
        norm_stats = {"key_a": {}, "key_b": {}}

        with (
            mock.patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=_FakeServerModel()),
            mock.patch.object(policy_wrapper, "read_mode_config", return_value=(model_cfg, norm_stats)),
            self.assertRaisesRegex(TypeError, "sequence of KEY=VALUE strings"),
        ):
            policy_wrapper.PolicyServerWrapper(
                ckpt_path="/tmp/model.pt",
                device="cpu",
                config_overrides="framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            )

    def test_server_cli_parses_repeated_config_override_values(self):
        parser = server_policy.build_argparser()
        args = parser.parse_args(
            [
                "--ckpt_path",
                "/tmp/model.pt",
                "--config_override",
                "a.b=false",
                "--config_override",
                "x.y=7",
            ]
        )

        self.assertEqual(args.config_override, ["a.b=false", "x.y=7"])

        seeded = parser.parse_args(
            [
                "--ckpt_path",
                "/tmp/model.pt",
                "--seed",
                "42",
                "--action-seed",
                "43",
            ]
        )
        self.assertEqual(seeded.seed, 42)
        self.assertEqual(seeded.action_seed, 43)

    def test_server_main_logs_override_keys_not_values(self):
        overrides = [
            "framework.action_model.diffusion_model_cfg.use_canonical_forward=false",
            "secret.token=do-not-log",
        ]
        args = mock.Mock(
            ckpt_path="/tmp/model.pt",
            use_bf16=False,
            port=12345,
            idle_timeout=-1,
            config_override=overrides,
        )
        fake_wrapper = mock.Mock()
        fake_wrapper.metadata = {"env": "test"}
        fake_server = mock.Mock()

        with (
            mock.patch.object(server_policy, "PolicyServerWrapper", return_value=fake_wrapper) as wrapper_cls,
            mock.patch.object(server_policy, "WebsocketPolicyServer", return_value=fake_server),
            mock.patch.object(server_policy.logging, "info") as log_info,
            mock.patch.object(server_policy.logging, "warning"),
        ):
            server_policy.main(args)

        wrapper_cls.assert_called_once_with(
            ckpt_path="/tmp/model.pt",
            device="cuda",
            use_bf16=False,
            config_overrides=overrides,
        )
        fake_server.serve_forever.assert_called_once_with()

        info_messages = [call.args for call in log_info.call_args_list]
        override_log = next(args for args in info_messages if args[0] == "Applying config override keys: %s")
        self.assertEqual(
            override_log[1],
            [
                "framework.action_model.diffusion_model_cfg.use_canonical_forward",
                "secret.token",
            ],
        )
        rendered_logs = "\n".join(str(args) for args in info_messages)
        self.assertNotIn("false", rendered_logs)
        self.assertNotIn("do-not-log", rendered_logs)


class LiberoLauncherContractTest(unittest.TestCase):
    def _run_launcher(self, **overrides):
        script = Path(
            "examples/simBenchmarks/LIBERO/eval_files/run_policy_server.sh"
        ).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Path(tmp) / "record_args.py"
            output = Path(tmp) / "args.txt"
            recorder.write_text(
                textwrap.dedent(
                    f"""\
                    #!/usr/bin/env python3
                    import pathlib
                    import sys
                    pathlib.Path({str(output)!r}).write_text("\\n".join(sys.argv[1:]))
                    """
                )
            )
            recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env.update(
                {
                    "FOREACT_DIR": str(Path.cwd()),
                    "FOREACT_PYTHON": str(recorder),
                    "CKPT": "/tmp/model.pt",
                    "GPU_ID": "0",
                    "PORT": "6694",
                    "USE_BF16": "1",
                }
            )
            env.update({key: str(value) for key, value in overrides.items()})
            result = subprocess.run(
                ["bash", str(script)],
                cwd=Path.cwd(),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            recorded_args = (
                output.read_text().splitlines() if output.exists() else None
            )
            return result, recorded_args

    def test_default_command_matches_public_libero_contract(self):
        result, args = self._run_launcher()

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("--ckpt_path", args)
        self.assertEqual(args[args.index("--ckpt_path") + 1], "/tmp/model.pt")
        self.assertEqual(args[args.index("--seed") + 1], "42")
        self.assertEqual(args[args.index("--action-seed") + 1], "42")
        self.assertIn(
            "framework.action_model.num_inference_timesteps=10",
            args,
        )
        self.assertIn("depth.enabled=false", args)
        self.assertIn("--use_bf16", args)

    def test_environment_overrides_seed_denoising_and_precision(self):
        result, args = self._run_launcher(
            SEED=7,
            DENOISE_STEPS=4,
            USE_BF16="0",
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(args[args.index("--seed") + 1], "7")
        self.assertEqual(args[args.index("--action-seed") + 1], "7")
        self.assertIn(
            "framework.action_model.num_inference_timesteps=4",
            args,
        )
        self.assertNotIn("--use_bf16", args)


if __name__ == "__main__":
    unittest.main()
