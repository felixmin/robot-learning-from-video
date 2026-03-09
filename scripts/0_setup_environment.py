#!/usr/bin/env python3
"""
Script 0: Environment Setup and Validation

Verifies that all dependencies are installed correctly and the environment is ready.
Run this first before starting any training.
"""

import sys
from pathlib import Path

# Add packages to path
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages"))

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

console = Console()


def check_imports():
    """Check if all required packages can be imported."""
    console.print("\n[bold blue]Checking imports...[/bold blue]")

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("lightning_fabric", "Lightning Fabric"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("webdataset", "WebDataset"),
        ("transformers", "Transformers"),
        ("timm", "TIMM"),
        ("wandb", "Weights & Biases"),
        ("einops", "Einops"),
    ]

    table = Table(title="Package Status")
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version", style="yellow")

    all_ok = True
    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            table.add_row(display_name, "✓ Installed", version)
        except ImportError:
            table.add_row(display_name, "✗ Missing", "-")
            all_ok = False

    console.print(table)
    return all_ok


def check_cuda():
    """Check GPU availability (CUDA or MPS)."""
    console.print("\n[bold blue]Checking GPU...[/bold blue]")

    import torch

    gpu_available = False

    # Check CUDA (NVIDIA GPUs on Linux/Windows)
    if torch.cuda.is_available():
        console.print(f"[green]✓ CUDA is available[/green]")
        console.print(f"  CUDA version: {torch.version.cuda}")
        console.print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            console.print(
                f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB)"
            )
        gpu_available = True

    # Check MPS (Apple Silicon on macOS)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print(f"[green]✓ MPS (Apple Silicon) is available[/green]")
        console.print(f"  Device: Apple Silicon GPU")
        console.print(f"  Note: Training will use Metal Performance Shaders")
        gpu_available = True

    else:
        console.print("[yellow]⚠ No GPU available (CPU only)[/yellow]")
        console.print("  Training will be slow on CPU")

    return gpu_available


def check_hydra_config():
    """Check if Hydra configs are valid."""
    console.print("\n[bold blue]Checking Hydra configuration...[/bold blue]")

    from hydra import compose, initialize_config_dir

    config_dir = workspace_root / "config"

    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config", overrides=["experiment=lam_debug"])
            console.print("[green]✓ Hydra configuration is valid[/green]")
            console.print(f"  Experiment: {cfg.experiment.name}")
            console.print(f"  Data batch size: {cfg.data.loader.batch_size}")
            console.print(f"  Training epochs: {cfg.training.epochs}")
            return True
    except Exception as e:
        console.print(f"[red]✗ Hydra configuration error: {e}[/red]")
        return False


def check_directories():
    """Check if required directories exist."""
    console.print("\n[bold blue]Checking directory structure...[/bold blue]")

    required_dirs = [
        "packages/common",
        "packages/lam",
        "packages/stage2",
        "packages/low_level",
        "config/experiment",
        "config/model",
        "config/data",
        "config/training",
        "config/cluster",
        "scripts",
        "slurm",
        "containers",
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = workspace_root / dir_path
        if full_path.exists():
            console.print(f"[green]✓ {dir_path}[/green]")
        else:
            console.print(f"[red]✗ {dir_path}[/red]")
            all_ok = False

    return all_ok


def main():
    """Run all checks."""
    console.print("[bold magenta]LAPA Environment Setup and Validation[/bold magenta]")
    console.print(f"Workspace: {workspace_root}")

    checks = [
        ("Directory Structure", check_directories),
        ("Python Packages", check_imports),
        ("GPU Support", check_cuda),
        ("Hydra Configuration", check_hydra_config),
    ]

    results = {}
    for name, check_func in checks:
        results[name] = check_func()

    # Summary
    console.print("\n[bold blue]Summary:[/bold blue]")
    all_passed = all(results.values())

    for name, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"  {name}: {status}")

    if all_passed:
        console.print(
            "\n[bold green]All checks passed! Environment is ready.[/bold green]"
        )
        return 0
    else:
        console.print(
            "\n[bold red]Some checks failed. Please fix the issues above.[/bold red]"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
