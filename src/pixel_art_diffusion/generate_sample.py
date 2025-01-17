import typer
from typing import Annotated
from .model import PixelArtDiffusion
from .visualize import visualize_samples

generate_app = typer.Typer()

@generate_app.command()
def generate_samples(
    checkpoint_path: Annotated[str, typer.Option(help="Path to the model checkpoint")],
    num_samples: Annotated[int, typer.Option(help="Number of samples to generate")] = 16,
    output_path: Annotated[str, typer.Option(help="Path to save the generated samples")] = "generated_samples.png",
):
    """
    Generate samples from a trained PixelArtDiffusion model
    """
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