import typer
from typing import Annotated
from pathlib import Path
from .model import PixelArtDiffusion
from .visualize import visualize_samples
import torch

generate_app = typer.Typer()

def get_models_dir():
    """Helper function to get the models directory path"""
    return Path(__file__).parent.parent.parent / "models"

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
        model_name = checkpoint.stem.split('-')[0]
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(checkpoint)

    # Print organized list
    for model_name, checkpoints in model_groups.items():
        print(f"\n{model_name}:")
        for checkpoint in sorted(checkpoints):
            print(f"  - {checkpoint.name}")

@generate_app.command()
def generate_samples(
    model_name: Annotated[str, typer.Option(help="Name of the model")] = "pixel_art_diffusion",
    num_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 32,
    output_path: Annotated[str, typer.Option(help="Path to save the generated samples")] = "generated_samples.png",
    custom_checkpoint_path: Annotated[str, typer.Option(help="Optional: Full path to a checkpoint file")] = None,
):
    """Generate samples from a trained PixelArtDiffusion model"""
    # Set up paths
    models_dir = get_models_dir()
    
    # Determine checkpoint path
    if custom_checkpoint_path:
        checkpoint_path = custom_checkpoint_path
    else:
        # Look for checkpoint files matching the model name
        possible_checkpoints = list(models_dir.glob(f"{model_name}*.pt"))
        if not possible_checkpoints:
            print(f"\nError: No checkpoint found for model '{model_name}' in {models_dir}")
            print("\nAvailable models:")
            list_models()
            raise typer.Exit(1)
        
        # Use the latest checkpoint if multiple exist
        checkpoint_path = str(sorted(possible_checkpoints)[-1])
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PixelArtDiffusion(device=DEVICE)
    # Initialize model
    model = PixelArtDiffusion()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)

    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples = model.generate_samples(num_samples=num_samples)
    
    # Visualize and save
    print("Saving visualization...")
    visualize_samples(samples, save_path=output_path)
    
    print("Done!")

if __name__ == "__main__":
    generate_app()