# Run Note: 2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h

## Meta

- Date: 2026-03-15
- Status: running
- Mode: cluster
- Host: tueilsy-st-022
- Code Commit: 5976ad102c58793424c5b6314e71c77f8b060ae4
- Worktree State: clean
- Logical Cluster Target: lrz cluster
- Stage: stage1
- Script: scripts/2_train_stage1_lam.py
- Base Experiment: stage1_cluster
- Config Path: config/experiment/runs/2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h.yaml
- Experiment Name: 2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h
- Intended Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h_{lrz|mcml}
- Final Run Dir: /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h_mcml
- LRZ Job ID: 5519373
- MCML Job ID: 5519374

## Purpose

- Run Stage 1 for 8 hours on the existing `octo24_libero` data mix, using 4 GPUs and per-GPU batch size 64.

## Config Delta Vs Default

- `data=octo24_libero`
- `cluster.compute.gpus_per_node=4`
- `cluster.compute.time_limit=08:00:00`
- `data.loader.batch_size=64`
- `experiment.name` set to the documented run stem
- `logging.tags` extended for the Octo24+Libero 4-GPU batch-64 run

## Upstream Artifacts / Checkpoints

- Type: stage1 checkpoint
  Source Run: none
  Path: none
  Notes: Fresh Stage 1 training run; no upstream checkpoint was used.

## Launch Command

```bash
# LRZ queue
python scripts/submit_job.py \
  experiment=stage1_cluster \
  experiment.name=2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h \
  experiment.description="Cluster Stage 1 training on the Octo24+Libero weight-5 mix using 4 GPUs and per-GPU batch size 64." \
  cluster=lrz_x100 \
  cluster.compute.gpus_per_node=4 \
  cluster.compute.time_limit=08:00:00 \
  data=octo24_libero \
  data.loader.batch_size=64 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h_lrz

# MCML queue
python scripts/submit_job.py \
  experiment=stage1_cluster \
  experiment.name=2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h \
  experiment.description="Cluster Stage 1 training on the Octo24+Libero weight-5 mix using 4 GPUs and per-GPU batch size 64." \
  cluster=mcml_x100 \
  cluster.compute.gpus_per_node=4 \
  cluster.compute.time_limit=08:00:00 \
  data=octo24_libero \
  data.loader.batch_size=64 \
  logging.runs_dir=/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/runs/2026-03-15_20-08-57_stage1_cluster_octo24_libero_weight5_4gpu_bs64_8h_mcml
```

## Results / Findings

- Submitted to LRZ and MCML.
- Both duplicates started immediately.
- LRZ `5519373` was canceled after start.
- Active run is MCML `5519374` on `mcml-hgx-h100-003`.
