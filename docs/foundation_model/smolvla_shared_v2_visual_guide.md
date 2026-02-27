# SmolVLA Shared v2 Visual Guide

This page gives an intuitive map of the new shared pipeline:
- one batch contract
- one transform logic
- one flow objective family for latent and real-action domains

## 1) End-to-End Architecture

```mermaid
flowchart LR
    subgraph S2["Stage 2 (Foundation)"]
      A2["OXE batch\nframes + language + state + action"] --> B2["FoundationBatch adapter"]
      B2 --> T["Shared Input Transform\n(text/image/state/action semantics)"]
      T --> C["SmolVLASharedCore"]
      C --> L["Latent flow loss\n[B,S_lat,D_step]"]
      C --> X["Artifact v2\n(strict manifest + core_state_dict)"]
    end

    subgraph S3["Stage 3 (LeRobot)"]
      A3["LeRobot batch\nmulti-camera + task + state + action chunk"] --> B3["HLRP policy adapter"]
      B3 --> T
      T --> C
      C --> R["Action chunk flow loss\n[B,T,A_max] + action_is_pad mask"]
      C --> Q["Action chunk sampler\nqueue -> select_action"]
    end

    X --> S3
```

## 2) Stage-3 Training Step (Chunk Flow)

```mermaid
sequenceDiagram
    participant D as Dataloader (LeRobot)
    participant P as HLRP Shared Policy
    participant T as Shared Input Transform
    participant C as Shared Core

    D->>P: batch (images/state/task/action, masks)
    P->>P: build FoundationBatch (image_streams, task_text, action_is_pad)
    P->>T: normalize/prepare state+actions, tokenize text, preprocess images
    T->>C: canonical tensors
    C->>C: sample t, build x_t and u_t
    C->>C: predict v_t over full chunk [B,T,A_max]
    C-->>P: masked flow loss
    P-->>D: scalar loss + metrics
```

## 3) Canonical Batch Shape Cheat Sheet

| Name | Shape | Meaning |
|---|---|---|
| `image_streams[key]` | `[B,O,C,H,W]` or `[B,O,H,W,C]` | Multi-camera observation stream |
| `image_padding_masks[key]` | `[B,O]` or `[B]` | Camera frame validity |
| `task_text` | `len=B` list[str] | Raw instruction text |
| `language_tokens` | `[B,L]` | Optional pretokenized fast path |
| `language_attention_mask` | `[B,L]` | Optional pretokenized mask |
| `state` | `[B,S]` or `[B,O,S]` | Robot state (last obs selected) |
| `target_latent_vectors` | `[B,D_lat]` or `[B,S_lat,D_step]` | Stage-2 latent target |
| `target_actions` | `[B,T,A]` (or `[B,A]` only if `chunk_size=1`) | Action target with strict chunk contract |
| `action_is_pad` | `[B,T]` bool | True means padded timestep |
| `normalization_stats` | dict | Shared state/action mean/std used in Stage2+Stage3 |

## 4) Domain Semantics

```mermaid
flowchart TD
    A["Latent domain"] --> A1["one latent action = [S_lat, D_step]"]
    A1 --> A2["flattened latent = [S_lat*D_step] = [D_lat]"]
    B["Action domain"] --> B1["one action chunk = [T, A_max]"]
    B1 --> B2["queue consumes first n_action_steps at inference"]
```

## 5) Why update_s changes with chunk flow

With one-step MSE, Stage-3 computed one vector and repeated it over the chunk.
With chunk flow, Stage-3 predicts a full velocity field over `[T, A]` and runs denoising integration over `flow_steps`.

So lower `train/update_s` is expected when moving from one-step MSE to true chunk flow matching.

## 6) Camera Behavior

Current behavior in v2:
- if `camera_keys` is set: use exactly those keys in that order
- else: use all provided streams from adapter in deterministic order
- missing configured cameras can be represented using `empty_cameras` synthetic masked streams
- every used camera key must provide `image_padding_masks[key]`
- `action_is_pad` can be provided as `actions_id_pad` alias in Stage 3 (conflict fails if both are present and differ)
