#!/usr/bin/env python
"""Test SmolVLM processor with image splitting disabled."""

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Testing SmolVLM Processor with Image Splitting Disabled")
    print("=" * 60)

    from transformers import AutoProcessor

    # Load processor
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        trust_remote_code=True,
    )

    # Check processor config
    print("\nOriginal processor config:")
    ip = processor.image_processor
    print(f"  do_image_splitting: {getattr(ip, 'do_image_splitting', 'N/A')}")
    print(f"  max_image_size: {getattr(ip, 'max_image_size', 'N/A')}")
    print(f"  size: {getattr(ip, 'size', 'N/A')}")

    # Create test image
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    text = "<image>Pick up the red block."

    # Test 1: Default (with splitting)
    print("\n" + "=" * 60)
    print("Test 1: Default processor (with image splitting)")
    print("=" * 60)
    inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)
    print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
    print(f"  pixel_attention_mask shape: {inputs.get('pixel_attention_mask', 'N/A')}")
    if "pixel_attention_mask" in inputs:
        print(f"  pixel_attention_mask shape: {inputs['pixel_attention_mask'].shape}")
    print(f"  input_ids shape: {inputs['input_ids'].shape}")

    # Test 2: Try to disable image splitting via processor config
    print("\n" + "=" * 60)
    print("Test 2: Attempting to disable image splitting")
    print("=" * 60)

    # Try setting do_image_splitting to False
    try:
        processor.image_processor.do_image_splitting = False
        print("  Set do_image_splitting = False")

        inputs2 = processor(
            text=[text], images=[[img]], return_tensors="pt", padding=True
        )
        print(f"  pixel_values shape: {inputs2['pixel_values'].shape}")
        if "pixel_attention_mask" in inputs2:
            print(
                f"  pixel_attention_mask shape: {inputs2['pixel_attention_mask'].shape}"
            )
        print(f"  input_ids shape: {inputs2['input_ids'].shape}")

        if inputs2["pixel_values"].shape != inputs["pixel_values"].shape:
            print("  ✓ Image splitting was disabled!")
            print(f"    Before: {inputs['pixel_values'].shape}")
            print(f"    After: {inputs2['pixel_values'].shape}")
        else:
            print("  ✗ Setting do_image_splitting had no effect")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 3: Try using a smaller max_image_size
    print("\n" + "=" * 60)
    print("Test 3: Using smaller max_image_size")
    print("=" * 60)

    # Reset and try with smaller max_image_size
    processor2 = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        trust_remote_code=True,
    )

    try:
        # Try setting a smaller max size that might avoid splitting
        processor2.image_processor.max_image_size = {"longest_edge": 256}
        processor2.image_processor.do_image_splitting = False
        print("  Set max_image_size = {'longest_edge': 256}")
        print("  Set do_image_splitting = False")

        inputs3 = processor2(
            text=[text], images=[[img]], return_tensors="pt", padding=True
        )
        print(f"  pixel_values shape: {inputs3['pixel_values'].shape}")
        if "pixel_attention_mask" in inputs3:
            print(
                f"  pixel_attention_mask shape: {inputs3['pixel_attention_mask'].shape}"
            )
        print(f"  input_ids shape: {inputs3['input_ids'].shape}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test 4: Check if processor has arguments to disable splitting
    print("\n" + "=" * 60)
    print("Test 4: Checking processor call signature")
    print("=" * 60)

    import inspect

    sig = inspect.signature(processor.__call__)
    print("  Processor __call__ parameters:")
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            print(f"    {name}: default={param.default}")
        else:
            print(f"    {name}")

    # Test 5: Try passing do_image_splitting as argument
    print("\n" + "=" * 60)
    print("Test 5: Passing do_image_splitting=False to processor call")
    print("=" * 60)

    processor3 = AutoProcessor.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        trust_remote_code=True,
    )

    try:
        inputs5 = processor3(
            text=[text],
            images=[[img]],
            return_tensors="pt",
            padding=True,
            do_image_splitting=False,
        )
        print(f"  pixel_values shape: {inputs5['pixel_values'].shape}")
        if "pixel_attention_mask" in inputs5:
            print(
                f"  pixel_attention_mask shape: {inputs5['pixel_attention_mask'].shape}"
            )
        print(f"  input_ids shape: {inputs5['input_ids'].shape}")

        if inputs5["pixel_values"].shape[1] == 1:
            print("  ✓ SUCCESS! do_image_splitting=False works as call argument!")
        else:
            print(f"  Still getting {inputs5['pixel_values'].shape[1]} patches")
    except TypeError as e:
        print(f"  TypeError: {e}")
    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    main()
