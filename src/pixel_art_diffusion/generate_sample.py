import typer
from typing import Annotated
from pathlib import Path
from pixel_art_diffusion.model import PixelArtDiffusion
from pixel_art_diffusion.visualize import visualize_samples
import torch
from diffusers import DDPMScheduler

generate_app = typer.Typer()


def get_models_dir():
    """Helper function to get the models directory path"""
    return Path(__file__).parent.parent.parent / "models"


def cpu_map_location(storage, loc):
    return storage


@generate_app.command(name="list")
def list_models():
    """List all available trained models"""
    models_dir = get_models_dir()
    checkpoints = list(models_dir.glob("*.pt"))

    if not checkpoints:
        print("No trained models found in models directory")
        return

    print("\nAvailable models:")
    print("-" * 50)

    # Group checkpoints by model name
    model_groups = {}
    for checkpoint in checkpoints:
        # Split at first hyphen to get base model name
        model_name = checkpoint.stem.split("-")[0]
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(checkpoint)

    # Print organized list
    for model_name, checkpoints in model_groups.items():
        print(f"\n{model_name}:")
        for checkpoint in sorted(checkpoints):
            print(f"  - {checkpoint.name}")


@generate_app.command(name="generate")
def generate_samples(
    model_name: Annotated[str, typer.Option(help="Name of the model")] = "pixel_art_diffusion",
    num_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 32,
    output_path: Annotated[str, typer.Option(help="Path to save the generated samples")] = None,
    custom_checkpoint_path: Annotated[str, typer.Option(help="Optional: Full path to a checkpoint file")] = None,
    force_cpu: Annotated[bool, typer.Option(help="Force CPU usage even if CUDA is available")] = False,
):
    """Generate samples from a trained PixelArtDiffusion model"""
    # Set up paths
    models_dir = get_models_dir()

    # Determine checkpoint path
    if custom_checkpoint_path:
        checkpoint_path = Path(custom_checkpoint_path)
        if not checkpoint_path.exists():
            print(f"\nError: Checkpoint file not found at {checkpoint_path}")
            raise typer.Exit(1)
    else:
        # Look for checkpoint files matching the model name
        possible_checkpoints = list(models_dir.glob(f"{model_name}*.pt"))
        if not possible_checkpoints:
            print(f"\nError: No checkpoint found for model '{model_name}' in {models_dir}")
            print("\nAvailable models:")
            list_models()
            raise typer.Exit(1)

        # Use the latest checkpoint if multiple exist
        checkpoint_path = sorted(possible_checkpoints)[-1]

    try:
        # Determine device
        if force_cpu:
            device = torch.device("cpu")
            print("Using CPU as requested")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                print("CUDA not available. Using CPU. This might be slow!")
            else:
                print("Using CUDA for generation")

        # Initialize model with correct device
        model = PixelArtDiffusion(device=device.type)

        # Load checkpoint with explicit CPU mapping if needed
        print(f"Loading checkpoint from {checkpoint_path}")
        # Use a map function to ensure proper device mapping
        if device.type == "cpu":
            # Force CPU mapping for non-CUDA systems
            map_location = cpu_map_location
        else:
            # Use the device directly for CUDA systems
            map_location = device

        checkpoint = torch.load(str(checkpoint_path), map_location=map_location)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.noise_scheduler = DDPMScheduler.from_config(checkpoint["scheduler_config"])

        # Generate samples
        print(f"Generating {num_samples} samples...")
        samples = model.generate_samples(num_samples=num_samples)

        # Visualize and save
        print("Saving visualization...")
        visualize_samples(samples, save_path=output_path)
        if output_path:
            print(f"Done! Samples saved to {output_path}")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nError: GPU out of memory. Try:")
            print("1. Reducing the number of samples")
            print("2. Using --force-cpu option")
            print("3. Freeing up GPU memory from other applications")
        elif "CUDA" in str(e):
            print("\nError loading CUDA checkpoint on CPU system.")
            print("Try using --force-cpu option to ensure proper CPU loading")
        else:
            print(f"\nError during generation: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    generate_app()
