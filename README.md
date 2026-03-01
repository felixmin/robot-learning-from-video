This code is build on [LAPA](https://github.com/LatentActionPretraining/LAPA).

# Robot-learning from Video


A three-stage robot learning system that learns policies from videos without action labels.

**Three Training Stages:**
1. **Stage 1 (LAQ)**: VQ-VAE compressing frame-to-frame transitions into discrete latent codes
2. **Stage 2 (Foundation)**: Vision-Language model predicting latent actions from images + text
3. **Stage 3 (Finetuning)**: Adapting the foundation model to output continuous robot commands

## Performance and resource efficiency
Performance benchmark of the training pipeline on an Desktop Nvidia 5090 GPU. With the current setup we achieve full GPU utilization.

For performance reasons over the course of the project we switched from Open X Embodiment in rlds format to a preprocessed version that allows direct index based access.

<p align="center">
  <img src="docs/assets/lam_gpu_utilization.png" alt="LAM training GPU utilization" width="700">
</p>

## Deployment

For robot teleoperation and policy deployment, we use our [CRISP Controller](https://utiasdsl.github.io/crisp_controllers/) setup:
<img width="830" height="458" alt="image" src="https://github.com/user-attachments/assets/8a4fe7b9-691f-4c27-b92b-4bf3062fcb92" />


Here is a dataset view of a human demonstration:

https://github.com/user-attachments/assets/c85271ec-d35f-4310-ac96-08a0e8d20476

Here is a view of a policy deployed to Lego stacking:

https://github.com/user-attachments/assets/0cc694b1-ff8c-406d-b5a7-a6815fe8c1af

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



## Citation

```bibtex
@misc{hlrp2024,
  title={V-PRO: Video Pretraining for Robot Action Policies},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details
