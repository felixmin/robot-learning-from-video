"""Tests for Hydra configuration composition."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    return str(Path(__file__).parent.parent / "config")


class TestExperimentConfigs:
    def test_stage2_local_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config", overrides=["experiment=stage2_local"]
            )

            assert cfg.experiment.name == "stage2_local"
            assert cfg.model.training_mode == "latent_flow"
            assert cfg.data.backend == "lerobot_v3"
            assert cfg.data.output_format == "stage2"

    def test_stage1_local_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_local"])

            assert cfg.experiment.name == "stage1_local"
            assert cfg.data.backend == "lerobot_v3"
            assert cfg.data.output_format == "stage1"
            assert len(cfg.data.dataset.lerobot.sources) == 24
            assert cfg.data.loader.batch_size == 64
            assert cfg.data.dataset.lerobot.sources[0].video_backend == "pyav"

    def test_stage2_local_octo24_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage2_local"])

            assert cfg.experiment.name == "stage2_local"
            assert cfg.data.backend == "lerobot_v3"
            assert cfg.data.output_format == "stage2"
            assert len(cfg.data.dataset.lerobot.sources) == 24
            assert cfg.stage2_policy_core.policy.image_size == [256, 256]

    def test_stage1_octo24_libero_data_override(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage1_local",
                    "data=octo24_libero",
                ],
            )

            assert cfg.data.output_format == "stage1"
            assert len(cfg.data.dataset.lerobot.sources) == 25
            assert cfg.data.dataset.lerobot.sources[5].repo_id == "HuggingFaceVLA/libero"

    def test_stage2_octo24_libero_data_override(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage2_local",
                    "data=octo24_libero",
                ],
            )

            assert cfg.data.backend == "lerobot_v3"
            assert len(cfg.data.dataset.lerobot.sources) == 25
            assert cfg.model.training_mode == "latent_flow"
            assert cfg.data.request.state_request.deltas_steps == [0]


class TestConfigComposition:
    def test_cli_overrides(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage1_local",
                    "data.loader.batch_size=16",
                    "training.optimizer.lr=5e-5",
                    "seed=123",
                ],
            )

            assert cfg.data.loader.batch_size == 16
            assert cfg.training.optimizer.lr == 5e-5
            assert cfg.seed == 123
            assert cfg.experiment.name == "stage1_local"

    def test_config_is_valid_omegaconf(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_local"])

            assert OmegaConf.is_config(cfg)
            assert OmegaConf.is_dict(cfg)


class TestExperimentConsistency:
    @pytest.mark.parametrize(
        "experiment",
        [
            "stage1_local",
            "stage1_cluster",
            "stage2_local",
            "stage2_cluster",
        ],
    )
    def test_stage12_experiments_load(self, config_dir, experiment):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

            assert hasattr(cfg, "experiment")
            assert hasattr(cfg, "model")
            assert hasattr(cfg, "data")
            assert hasattr(cfg, "training")
            assert hasattr(cfg, "cluster")
            assert cfg.experiment.name is not None
            assert cfg.experiment.description is not None

    @pytest.mark.parametrize(
        "experiment",
        [
            "stage3_local",
            "stage3_cluster",
        ],
    )
    def test_lerobot_experiments_load(self, config_dir, experiment):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])

            assert hasattr(cfg, "experiment")
            assert hasattr(cfg, "lerobot")
            assert hasattr(cfg, "cluster")
            assert cfg.experiment.name is not None
            assert cfg.experiment.description is not None
            assert cfg.lerobot.command is not None
            assert cfg.lerobot.policy.type is not None
            assert cfg.lerobot.policy.init_mode in {"artifact", "scratch"}

    def test_stage3_rollout_experiment_loads(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage3_rollout"])

            assert hasattr(cfg, "experiment")
            assert hasattr(cfg, "lerobot_eval")
            assert hasattr(cfg, "cluster")
            assert cfg.experiment.name == "stage3_rollout"
            assert cfg.lerobot_eval.command is not None

    def test_stage3_sweep_experiment_loads(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage3_latent_vs_multitask_sweep",
                    "variant_id=mt10_lat100",
                ],
            )

            assert hasattr(cfg, "experiment")
            assert hasattr(cfg, "lerobot")
            assert cfg.variant_id == "mt10_lat100"
            assert cfg.lerobot.policy.stage3_training_mode == "multitask"
            assert cfg.lerobot.policy.action_subset_ratio == 0.1
            assert cfg.lerobot.policy.latent_scope == "all"

    def test_stage3_runtime_override_switches_gl_backend(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage3_local",
                    "stage3_profile=multitask_scratch",
                    "runtime=stage3_cluster",
                ],
            )

            assert cfg.lerobot.shell_env.MUJOCO_GL == "osmesa"
            assert str(cfg.lerobot.shell_env.LIBERO_CONFIG_PATH).endswith("/cache/libero_config")
