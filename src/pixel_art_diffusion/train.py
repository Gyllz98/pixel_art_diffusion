import torch
import torch.nn as nn
import typer
import wandb
import hydra
import os
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from typing import Annotated, List, Optional, Dict, Any, Union
from typing_extensions import TypeAlias
from pathlib import Path
from omegaconf import DictConfig
from pixel_art_diffusion.data import PixelArtDataset
from pixel_art_diffusion.model import PixelArtDiffusion 
from loguru import logger

# Type aliases
BatchType: TypeAlias = Dict[str, torch.Tensor]
WandBConfig: TypeAlias = Dict[str, Union[float, int, List[int], str, Dict[str, Any]]]
OptimizerParamsType: TypeAlias = Dict[str, Any]

train_app = typer.Typer()

def get_models_dir() -> Path:
    """
    Helper function to get the models directory path
    
    Returns:
        Path to the models directory
    """
    return Path(__file__).parent.parent.parent / "models"

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train_model_hydra(cfg: DictConfig) -> None:
    """
    Main training function with Hydra config
    
    Args:
        cfg: Hydra configuration object containing training parameters
    """
    logger.info("Starting training with Hydra configuration")
    logger.debug(f"Configuration: {cfg}")
    
    # Get command line arguments from environment variables or use defaults
    run_name: str = os.getenv("TRAIN_RUN_NAME", "pixel_art_diffusion")
    num_epochs: int = int(os.getenv("TRAIN_NUM_EPOCHS", "100"))
    label_subset: List[int] = [int(x) for x in os.getenv("TRAIN_LABEL_SUBSET", "3").split(",")]

    logger.info(f"Training run: {run_name}, Epochs: {num_epochs}, Label subset: {label_subset}")

    # Set up paths
    models_dir: Path = get_models_dir()
    models_dir.mkdir(exist_ok=True)

    # Initialize W&B
    if cfg.wandb.enabled:
        logger.info("Initializing Weights & Biases")
        wandb_config: WandBConfig = {
            "learning_rate": cfg.optimizer.params.lr,
            "num_epochs": num_epochs,
            "label_subset": label_subset,
            "architecture": "PixelArtDiffusion",
            **cfg.model,
        }
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config=wandb_config,
        )

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {DEVICE}")

    model: PixelArtDiffusion = PixelArtDiffusion(device=DEVICE)
    dataset: PixelArtDataset = PixelArtDataset(
        data_path=cfg.data.root_path,
        calculate_stats=cfg.data.calculate_stats,
        label_subset=label_subset
    )

    # Create dataloader
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers
    )

    # Setup optimizer
    optimizer_params: OptimizerParamsType = cfg.optimizer.params
    optimizer: torch.optim.AdamW = torch.optim.AdamW(
        model.model.parameters(),
        **optimizer_params
    )

    # Calculate total steps for scheduler
    total_steps: int = len(dataloader) * num_epochs

    # Setup scheduler
    lr_scheduler: Any = get_scheduler(
        name=cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.num_warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Starting training for {num_epochs} epochs...")

    # Main training loop
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.model.train()
        total_loss: float = 0.0

        for batch_idx, batch in enumerate(dataloader):
            clean_images: torch.Tensor = batch["pixel_values"].to(model.device)

            # Sample noise and timesteps
            noise: torch.Tensor = torch.randn_like(clean_images)
            timesteps: torch.Tensor = torch.randint(
                0,
                model.noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=model.device
            )

            # Add noise according to schedule
            noisy_images: torch.Tensor = model.noise_scheduler.add_noise(
                clean_images,
                noise,
                timesteps
            )

            # Predict noise
            noise_pred: torch.Tensor = model.model(noisy_images, timesteps).sample

            # Calculate loss
            loss: torch.Tensor = nn.functional.mse_loss(noise_pred, noise)
            total_loss += loss.item()

            # Optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.model.parameters(),
                cfg.training.clip_grad_norm
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Log batch metrics
            if cfg.wandb.enabled and batch_idx % cfg.wandb.log_batch_frequency == 0:
                batch_metrics: Dict[str, Union[float, int]] = {
                    "batch/loss": loss.item(),
                    "batch/learning_rate": lr_scheduler.get_last_lr()[0],
                    "batch/epoch": epoch,
                    "batch/batch_idx": batch_idx,
                }
                wandb.log(batch_metrics)

        avg_loss: float = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Log epoch metrics
        if cfg.wandb.enabled:
            epoch_metrics: Dict[str, Union[float, int]] = {
                "epoch/average_loss": avg_loss,
                "epoch": epoch,
            }
            wandb.log(epoch_metrics)

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            checkpoint_path: Path = models_dir / f"{run_name}-checkpoint-epoch-{epoch}.pt"
            model.save_checkpoint(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")

    logger.success("Training completed successfully!")
    # Save final checkpoint
    final_checkpoint_path: Path = models_dir / f"{run_name}-final.pt"
    model.save_checkpoint(str(final_checkpoint_path))
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    if cfg.wandb.enabled:
        wandb.finish()

@train_app.command()
def train_model(
    run_name: Annotated[str, typer.Option(help="Name for the training run")] = "pixel_art_diffusion",
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 100,
    label_subset: Annotated[List[int], typer.Option(help="List of label indices to train on (0-4)")] = [3],
) -> None:
    """
    CLI wrapper for the training function
    
    Args:
        run_name: Name for the training run
        num_epochs: Number of training epochs
        label_subset: List of label indices to train on (0-4)
    """
    # Set environment variables for Hydra function
    os.environ["TRAIN_RUN_NAME"] = run_name
    os.environ["TRAIN_NUM_EPOCHS"] = str(num_epochs)
    os.environ["TRAIN_LABEL_SUBSET"] = ",".join(map(str, label_subset))

    # Call the Hydra main function
    train_model_hydra()

if __name__ == "__main__":
    train_app()