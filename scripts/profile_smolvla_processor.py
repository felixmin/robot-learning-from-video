#!/usr/bin/env python
"""Profile SmolVLA processor to identify bottlenecks."""

import time
import torch
from PIL import Image
import numpy as np


def profile_section(name, fn, warmup=2, runs=10):
    """Profile a function and return timing stats."""
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)

    return {
        "name": name,
        "mean": np.mean(times) * 1000,  # ms
        "std": np.std(times) * 1000,
        "min": np.min(times) * 1000,
        "max": np.max(times) * 1000,
    }


def main():
    print("=" * 60)
    print("SmolVLA Processor Profiling")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and processor
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    print(f"\nLoading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Print model structure for debugging
    print(f"\nModel type: {type(model).__name__}")
    print("Model children:")
    for name, child in model.named_children():
        print(f"  {name}: {type(child).__name__}")
    if hasattr(model, "model"):
        print("model.model children:")
        for name, child in model.model.named_children():
            print(f"  {name}: {type(child).__name__}")

    # Create test data
    batch_size = 8
    img_size = 256

    # Create PIL images (simulating what we do currently)
    pil_images = [
        [
            Image.fromarray(
                np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            )
        ]
        for _ in range(batch_size)
    ]

    # Create tensor images (simulating LeRobot approach)
    tensor_images = torch.rand(batch_size, 3, img_size, img_size, device=device)

    # Create text prompts (must include <image> placeholder for SmolVLM2)
    texts = [
        "<image>Pick up the red block and place it on the blue block."
    ] * batch_size

    print(f"\nBatch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")

    # Profile different components
    results = []

    # 1. Profile full processor call (current approach)
    def full_processor():
        inputs = processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        return inputs

    results.append(profile_section("Full HF Processor (PIL images)", full_processor))

    # 2. Profile just text tokenization
    def text_tokenization():
        return processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )

    results.append(profile_section("Text tokenization only", text_tokenization))

    # 3. Profile PIL image processing
    def pil_to_tensor():
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        tensors = [transform(img[0]) for img in pil_images]
        return torch.stack(tensors).to(device)

    results.append(profile_section("PIL to tensor (torchvision)", pil_to_tensor))

    # 4. Profile GPU-based image preprocessing (LeRobot style)
    def gpu_image_preprocess():
        # Resize with F.interpolate (GPU)
        img = torch.nn.functional.interpolate(
            tensor_images, size=(384, 384), mode="bilinear", align_corners=False
        )
        # Normalize to [-1, 1]
        img = img * 2.0 - 1.0
        return img

    results.append(
        profile_section("GPU image preprocess (F.interpolate)", gpu_image_preprocess)
    )

    # 5. Profile vision encoder directly
    # Find the vision model
    vision_model = None
    if hasattr(model, "model") and hasattr(model.model, "vision_model"):
        vision_model = model.model.vision_model
    elif hasattr(model, "vision_model"):
        vision_model = model.vision_model

    if vision_model is not None:
        print(f"\nVision model found: {type(vision_model).__name__}")

        def vision_encoder_direct():
            img = torch.nn.functional.interpolate(
                tensor_images.to(torch.bfloat16),
                size=(384, 384),
                mode="bilinear",
                align_corners=False,
            )
            img = img * 2.0 - 1.0
            with torch.no_grad():
                vision_outputs = vision_model(pixel_values=img)
            return vision_outputs.last_hidden_state

        results.append(
            profile_section("Vision encoder (direct)", vision_encoder_direct)
        )
    else:
        print("\nWARNING: Could not find vision_model in model structure")

    # 6. Profile full forward pass with processor (current approach)
    cached_inputs = full_processor()

    def forward_with_processor():
        with torch.no_grad():
            out = model(**cached_inputs, output_hidden_states=True)
        return out.hidden_states[-1]

    results.append(
        profile_section(
            "Full forward (with cached processor output)", forward_with_processor
        )
    )

    # 7. Profile forward WITHOUT output_hidden_states
    def forward_no_hidden():
        with torch.no_grad():
            out = model(**cached_inputs, output_hidden_states=False)
        return out.logits

    results.append(
        profile_section("Full forward (NO hidden states)", forward_no_hidden)
    )

    # Print results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"{'Component':<45} {'Mean (ms)':>10} {'Std':>8}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<45} {r['mean']:>10.2f} {r['std']:>8.2f}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Find results by name
    result_map = {r["name"]: r["mean"] for r in results}

    processor_time = result_map.get("Full HF Processor (PIL images)", 0)
    gpu_preprocess_time = result_map.get("GPU image preprocess (F.interpolate)", 0)
    if processor_time > 0 and gpu_preprocess_time > 0:
        speedup = processor_time / gpu_preprocess_time
        print(f"\nHF Processor: {processor_time:.2f} ms")
        print(f"GPU preprocess: {gpu_preprocess_time:.2f} ms")
        print(f"Potential preprocessing speedup: {speedup:.1f}x")

    forward_with_hidden = result_map.get(
        "Full forward (with cached processor output)", 0
    )
    forward_no_hidden = result_map.get("Full forward (NO hidden states)", 0)
    if forward_with_hidden > 0 and forward_no_hidden > 0:
        hidden_overhead = forward_with_hidden - forward_no_hidden
        print(f"\nForward (with hidden_states): {forward_with_hidden:.2f} ms")
        print(f"Forward (no hidden_states): {forward_no_hidden:.2f} ms")
        print(
            f"Hidden states overhead: {hidden_overhead:.2f} ms ({hidden_overhead/forward_with_hidden*100:.1f}%)"
        )

    # Breakdown of total time per forward pass
    print("\n" + "=" * 60)
    print("TOTAL TIME BREAKDOWN (per forward pass)")
    print("=" * 60)
    if processor_time > 0 and forward_with_hidden > 0:
        total_time = processor_time + forward_with_hidden
        print(
            f"Preprocessing: {processor_time:.2f} ms ({processor_time/total_time*100:.1f}%)"
        )
        print(
            f"Forward pass:  {forward_with_hidden:.2f} ms ({forward_with_hidden/total_time*100:.1f}%)"
        )
        print(f"TOTAL:         {total_time:.2f} ms")
        print(
            f"\nAt batch_size={batch_size}: {total_time:.2f}ms -> {1000/total_time:.2f} batches/sec -> {batch_size * 1000 / total_time:.1f} samples/sec"
        )


if __name__ == "__main__":
    main()
