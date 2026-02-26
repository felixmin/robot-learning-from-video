# VLA Training Survey (2026-02-24)

This folder contains one file per model/paper with implementation notes and training-config extraction from primary sources.

Name mapping used for your request:
- `Laper` interpreted as `LAPA`.
- `Small VLA` interpreted as `SmolVLA`.
- `NVIDIA Grud` interpreted as `NVIDIA GR00T`.
- `cosmos policy mimic video` mapped to publicly documented `GR00T-Mimic / Cosmos policy-mimic context`.

## Files
- `openvla.md`
- `lapa.md`
- `smolvla.md`
- `pi0.md`
- `pi0_5.md`
- `nvidia_gr00t_n1_5.md`
- `nvidia_gr00t_mimic_cosmos.md`
- `rt1.md`
- `rt2.md`
- `octo.md`

## Cross-model snapshot
| Model | Full training | LoRA path | Steps | LR | Batch size | Hardware |
|---|---|---|---|---|---|---|
| OpenVLA | Yes | Yes (repo) | ~45k (paper main recipe) | 2e-5 (paper main recipe) | 256 (paper main recipe) | 8xA100 suggested in repo full-FT command |
| LAPA | Yes | Not primary | Not fully centralized | Stage1 2e-4, Stage2 5e-5 | Stage1 512, Stage2 2048 | 64xH100 |
| SmolVLA | Yes | Not primary | 300k pretrain | 1e-3 base, cosine | 256 pretrain; 64 task FT (paper simulation) | 4 GPUs (paper pretraining setup) |
| pi0 | Yes | Not primary in paper | Not fully disclosed | Not fully disclosed (OpenPI example 5e-5) | Not fully disclosed (OpenPI example 32) | Not fully disclosed |
| pi0.5 | Yes (post-training pipeline) | Not primary | Not fully disclosed | Not fully disclosed | Not fully disclosed | Not fully disclosed |
| NVIDIA GR00T N1.5 | Yes | Yes | Epoch-based (50 in public finetune config) | 5e-6 (public finetune config) | global 256, micro 4 | 1 node x 8 GPUs in public finetune config |
| GR00T-Mimic / Cosmos | N/A (data blueprint) | N/A | N/A | N/A | N/A | N/A |
| RT-1 | Yes | No | Not fully disclosed | Not fully disclosed | 1024 | Not fully disclosed |
| RT-2 | Yes (co-finetune) | Not primary | Not fully disclosed | Not fully disclosed | Not fully disclosed | Not fully disclosed |
| Octo | Yes | Not primary | ~700k pretrain | peak 3e-4 in docs example | 2048 in docs example | Not fully disclosed |

## Notes
- This survey is limited to publicly disclosed values in papers/repos/docs.
- Several major industrial VLA papers do not publish a full reproducibility-grade training card (all of LR schedule, steps, global batch, hardware, and data mixture proportions in one place).
