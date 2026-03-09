#!/usr/bin/env python
"""Inspect what SmolVLM processor actually does to understand preprocessing requirements."""

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("SmolVLM Processor Inspection")
    print("=" * 60)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", trust_remote_code=True
    )

    # Create test image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    # Process with the processor
    text = "<image>Test instruction"
    inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)

    print("\nProcessor output keys:", list(inputs.keys()))
    print("\nShapes:")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}, dtype={v.dtype}")

    # Check pixel_values range and characteristics
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        print("\npixel_values stats:")
        print(f"  min: {pv.min().item():.4f}")
        print(f"  max: {pv.max().item():.4f}")
        print(f"  mean: {pv.mean().item():.4f}")

    # Check image processor config
    print("\n" + "=" * 60)
    print("Image Processor Configuration")
    print("=" * 60)
    ip = processor.image_processor
    print(f"  type: {type(ip).__name__}")
    print(f"  size: {getattr(ip, 'size', None)}")
    print(f"  rescale_factor: {getattr(ip, 'rescale_factor', None)}")
    print(f"  image_mean: {getattr(ip, 'image_mean', None)}")
    print(f"  image_std: {getattr(ip, 'image_std', None)}")
    print(f"  do_resize: {getattr(ip, 'do_resize', None)}")
    print(f"  do_rescale: {getattr(ip, 'do_rescale', None)}")
    print(f"  do_normalize: {getattr(ip, 'do_normalize', None)}")
    print(f"  do_pad: {getattr(ip, 'do_pad', None)}")
    print(f"  max_image_size: {getattr(ip, 'max_image_size', None)}")

    # Compare with GPU preprocessing
    print("\n" + "=" * 60)
    print("GPU Preprocessing Comparison")
    print("=" * 60)

    # Convert PIL to tensor (simulating what we'd have in batch)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

    # GPU preprocessing (LeRobot style)
    target_size = (384, 384)

    # Resize with aspect ratio
    cur_h, cur_w = img_tensor.shape[2:]
    ratio = max(cur_w / target_size[1], cur_h / target_size[0])
    new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
    resized = torch.nn.functional.interpolate(
        img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
    )

    # Pad to target size
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    padded = torch.nn.functional.pad(resized, (pad_w, 0, pad_h, 0), value=0.0)

    # Normalize to [-1, 1] (SigLIP style)
    normalized = padded * 2.0 - 1.0

    print("\nGPU preprocessed stats:")
    print(f"  shape: {normalized.shape}")
    print(f"  min: {normalized.min().item():.4f}")
    print(f"  max: {normalized.max().item():.4f}")
    print(f"  mean: {normalized.mean().item():.4f}")

    # Check if HF processor uses mean/std normalization vs [-1,1]
    if "pixel_values" in inputs:
        hf_pv = inputs["pixel_values"]
        # If HF uses mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], output should be in [-1,1]
        # If HF uses ImageNet mean/std, output will be different
        print(
            f"\nHF processor pixel_values range: [{hf_pv.min():.4f}, {hf_pv.max():.4f}]"
        )
        print(
            f"GPU preprocessing range: [{normalized.min():.4f}, {normalized.max():.4f}]"
        )

        if (
            abs(hf_pv.min().item() - (-1.0)) < 0.5
            and abs(hf_pv.max().item() - 1.0) < 0.5
        ):
            print("\n✓ HF processor likely uses [-1, 1] normalization (SigLIP style)")
        else:
            print("\n⚠ HF processor may use different normalization!")


if __name__ == "__main__":
    main()
