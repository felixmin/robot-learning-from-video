"""Tests for Hydra configuration composition."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@pytest.fixture
def config_dir():
    return str(Path(__file__).parent.parent / "config")


class TestExperimentConfigs:
    def test_laq_oxe_local_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_laq_oxe_local"])

            assert cfg.experiment.name == "laq_oxe_local"
            assert cfg.data.backend == "oxe_local_indexed"
            assert cfg.data.adapter.openx_local.mode == "indexed_full"
            assert bool(cfg.data.adapter.openx_local.auto_discover) is True
            assert cfg.cluster.name == "local_dev"

    def test_laq_oxe_cluster_flow_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_laq_oxe_cluster"])

            assert cfg.experiment.name == "laq_oxe_cluster"
            assert cfg.model.flow.model == "raft_large"
            assert cfg.model.flow.decoder_depth == 2
            assert bool(cfg.validation.strategies.flow_visualization.enabled) is True

    def test_vla_smol_flow_shared_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config", overrides=["experiment=stage2_smol_flow"]
            )

            assert cfg.experiment.name == "latent_smolvla"
            assert cfg.model.training_mode == "latent_flow"
            assert cfg.data.backend == "oxe_local_indexed"
            assert cfg.data.adapter.openx_local.mode == "indexed_full"

    def test_vla_cosmos2_tokens_overfit_config(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config", overrides=["experiment=stage2_cosmos2_tokens_overfit"]
            )

            assert cfg.experiment.name == "vla_cosmos2_tokens_overfit"
            assert cfg.data.backend == "oxe_local_indexed"
            assert cfg.training.overfit_batches == 1


class TestConfigComposition:
    def test_cli_overrides(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=stage1_laq_oxe_local",
                    "data.loader.batch_size=16",
                    "training.optimizer.lr=5e-5",
                    "seed=123",
                ],
            )

            assert cfg.data.loader.batch_size == 16
            assert cfg.training.optimizer.lr == 5e-5
            assert cfg.seed == 123
            assert cfg.experiment.name == "laq_oxe_local"

    def test_config_is_valid_omegaconf(self, config_dir):
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="config", overrides=["experiment=stage1_laq_oxe_local"])

            assert OmegaConf.is_config(cfg)
            assert OmegaConf.is_dict(cfg)


class TestExperimentConsistency:
    @pytest.mark.parametrize(
        "experiment",
        [
            "stage1_laq_oxe_cluster",
            "stage1_laq_oxe_local",
            "stage1_laq_oxe_local_sweep",
            "stage1_laq_token_analysis",
            "stage2_cosmos2_tokens",
            "stage2_cosmos2_tokens_overfit",
            "stage2_smol_flow",
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
            "stage3_hlrp_libero_action_scratch",
            "stage3_hlrp_libero_multitask_scratch",
            "stage3_hlrp_libero_multitask_scratch_local",
            "stage3_hlrp_libero_multitask_scratch_cluster",
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
