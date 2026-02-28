This code is build on [LAPA](https://github.com/LatentActionPretraining/LAPA).

# Robot-learning from Video


A three-stage robot learning system that learns policies from videos without action labels.

**Three Training Stages:**
1. **Stage 1 (LAQ)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: Vision-Language model predicting latent actions from images + text
3. **Stage 3 (Finetuning)**: Adapting the foundation model to output continuous robot commands

## Performance and resource efficiency
For performance reasons over the course of the project we switched from Open X Embodiment in rlds format to a preprocessed version that allows direct index based access. With the current setup we achieve full GPU utilization.

<p align="center">
  <img src="docs/assets/lam_gpu_utilization.png" alt="LAM training GPU utilization" width="700">
</p>

## Repository Structure

```
├── packages/              # Installable Python packages
│   ├── common/           # Shared utilities, logging, data interfaces
│   ├── laq/              # Stage 1: Latent action quantization (VQ-VAE)
│   ├── foundation/       # Stage 2: Vision-Language-Action model
├── config/               # Hydra configurations (modular, composable)
│   ├── experiment/       # Complete experiment setups
│   ├── model/, data/, training/, cluster/  # Config components
├── scripts/              # Training entry points (numbered by stage)
├── tests/                # Unit and integration tests
├── lerobot_policy_hlrp / # Definition of the installable lerobot policy to be used with the lerobot library
├── docs/                 # Documentation (LRZ workflow guide)
└── containers/           # Enroot/Docker definitions for LRZ
```

## Getting Started

### Installation

```bash
# Create conda environment (Python 3.12)
conda env create -f environment.yml
conda activate hlrp

# Install PyTorch 2.9.1 with CUDA support
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Install project packages
pip install -e .

# Verify setup
python scripts/0_setup_environment.py
```

### Running Training

```bash
# Stage 1 (LAQ) local
python scripts/2_train_laq.py experiment=stage1_laq_oxe_local

# Stage 2 (foundation SmolVLA flow) local
python scripts/4_train_foundation.py experiment=stage2_smol_flow

# Stage 3 (LeRobot action-only) local
python scripts/6_train_lerobot.py experiment=stage3_hlrp_libero_action_scratch

# Cluster submit (all stages)
python scripts/submit_job.py experiment=stage3_hlrp_libero_action_scratch cluster=lrz_x100
```

## Configuration

Uses [Hydra](https://hydra.cc) for composable configuration. Experiments compose modular config components:

```bash
# Override from CLI
python scripts/2_train_laq.py experiment=stage1_laq_oxe_local data.loader.batch_size=32 training.optimizer.lr=5e-5
```

See `CLAUDE.md` for architecture and config structure details.

## Dataset Sources and Downloads

- Workstation/local training usually uses local dataset paths:
  - Stage 1/2 OXE local shards: `/mnt/data/oxe` (via `data=oxe_local_indexed`)
  - Stage 3 Libero snapshot: `/mnt/data/workspace/hflibero/datasets--HuggingFaceVLA--libero/snapshots/<snapshot>`
- Cluster training:
  - Stage 1/2 uses dataset paths available on cluster storage; OXE is not auto-downloaded by the training scripts.
  - Stage 3 can download Libero at run start if `lerobot.dataset.root=null` and `lerobot.dataset.repo_id=HuggingFaceVLA/libero`.
- Runtime-downloaded datasets/assets:
  - `HuggingFaceVLA/libero` dataset cache (`HF_DATASETS_CACHE`)
  - LIBERO environment assets (`~/.cache/libero/assets`) if missing
  - Hugging Face model/checkpoint cache (`HF_HUB_CACHE`)

## Development

```bash
# Run tests
pytest tests/

# Format code
black packages/ scripts/ tests/

# Lint
ruff check packages/ scripts/ tests/
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project architecture, development commands, and setup for Claude Code
- **[AGENTS.md](AGENTS.md)** - Operational rules and runbook for local and cluster execution

## Dependencies

Python 3.12 with PyTorch 2.9.1. Key packages: pytorch-lightning, transformers, webdataset, hydra-core, wandb, accelerate.

See `environment.yml` for complete dependency list.

## Learning Robotic Action from Human Video - Hardware Setup

Human demonstration data is significantly easier and more cost-effective to collect than traditional robot data, and it is widely available across the internet. By engineering a setup to record humans performing everyday actions, we can use this rich dataset to train a Franka Emika robot. Ultimately, this greater data diversity allows the robot to learn new tasks more efficiently and achieve better generalization in real-world environments.

### Demonstrations

**Human Demonstration:**




https://github.com/user-attachments/assets/c85271ec-d35f-4310-ac96-08a0e8d20476




**Franka Emika Robot Execution:**




https://github.com/user-attachments/assets/0cc694b1-ff8c-406d-b5a7-a6815fe8c1af




## Citation

```bibtex
@misc{hlrp2024,
  title={High-Level Robot Planner: Learning Policies from Videos},
  author={Your Team},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details
