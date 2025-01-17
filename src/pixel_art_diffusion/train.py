import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import typer
import wandb
from typing import Annotated, List
from .data import PixelArtDataset
from .model import PixelArtDiffusion
from pathlib import Path

train_app = typer.Typer()


@train_app.command()
def train_model(
    run_name: Annotated[str, typer.Option(help="Name for the training run")] = "pixel_art_diffusion",
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 100,
    label_subset: Annotated[List[int], typer.Option(help="List of label indices to train on (0-4)")] = [3],
):
    """
    Train the PixelArtDiffusion model
    """
    # Fixed hyperparameters
    LEARNING_RATE = 1e-4
    CHECKPOINT_FREQ = 10

    # Initialize W&B
    wandb.init(
        project="pixel-art-diffusion",
        name=run_name,
        config={
            "learning_rate": LEARNING_RATE,
            "num_epochs": num_epochs,
            "label_subset": label_subset,
            "architecture": "PixelArtDiffusion",
        },
    )

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "processed"

    model = PixelArtDiffusion()
    # First time setup - calculate statistics
    dataset = PixelArtDataset(data_path=str(data_path), calculate_stats=True, label_subset=label_subset)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01, eps=1e-8)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=len(dataloader) * num_epochs
    )

    print(f"Starting training for {num_epochs} epochs...")

    # Main training loop
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            clean_images = batch["pixel_values"].to(model.device)

            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=model.device
            )

            # Add noise according to schedule
            noisy_images = model.noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise
            noise_pred = model.model(noisy_images, timesteps).sample

            # Calculate loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            # Optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Log batch metrics
            if batch_idx % 100 == 0:  # Log every 100 batches
                wandb.log(
                    {
                        "batch/loss": loss.item(),
                        "batch/learning_rate": lr_scheduler.get_last_lr()[0],
                        "batch/epoch": epoch,
                        "batch/batch_idx": batch_idx,
                    }
                )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Log epoch metrics
        wandb.log(
            {
                "epoch/average_loss": avg_loss,
                "epoch": epoch,
            }
        )

        # Save intermediate checkpoint
        if epoch % CHECKPOINT_FREQ == 0:
            checkpoint_path = f"{run_name}-checkpoint-epoch-{epoch}.pt"
            model.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

            # Log checkpoint to W&B
            wandb.save(checkpoint_path)

    # Save final checkpoint
    final_checkpoint_path = f"{run_name}-final.pt"
    model.save_checkpoint(final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    wandb.save(final_checkpoint_path)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    train_app()
