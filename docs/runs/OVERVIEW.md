# Latent Action Model Versions

## Randomly initialized checkpoint without training for comparisons


## Libero overfitting run
Pixel encoder
Pixel + flow decoder
2026-03-14_19-27-26_stage1_local

# LIBERO Experiments

# Full libero dataset

## Baseline action only
100% action labeled: train only action head



# Libero with 5% action labeled and 95% without real action labels

## Baseline only action
5% action labeled: train only action head

## Baseline multitask
Latent action model: Pretrained on octo24 + Libero (5% of the dataset)
5% action labeled: train multitask action head and latent head at the same time

## 95% for latents balanced 50/50
Latent action model: Pretrained on octo24 + Libero (5% of the dataset)
5% action labeled: train multitask action head and latent head at the same time
95% no action labeles: train latent head only

## 95% for latents balanced 50/50 Libero overfitting
Latent action model: Pretrained on Libero only (2026-03-14_19-27-26_stage1_local)
5% action labeled: train multitask action head and latent head at the same time
95% no action labeles: train latent head only
2026-03-15_01-43-48_stage3_local_libero_95latent_5mt_balanced_bs64_latent0p1

## Baseline multitask
5% action labeled: train multitask action head and latent head at the same time


