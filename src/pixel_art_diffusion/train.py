import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from model import PixelArtDiffusion

def train_model(
    dataloader,
    model,
    num_epochs=100,
    learning_rate=1e-4,
    checkpoint_freq=10,
    run_name="pixel_art_diffusion"
):
    """
    Train the PixelArtDiffusion model
    
    Args:
        dataloader: DataLoader containing the training data
        model: PixelArtDiffusion model instance
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        checkpoint_freq: How often to save checkpoints (in epochs)
        run_name: Name for the training run (used in checkpoint naming)
    """
    
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(dataloader) * num_epochs
    )

    print(f"Starting training for {num_epochs} epochs...")
    
    # Main training loop
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.model.train()
        total_loss = 0
        
        for batch in dataloader:
            clean_images = batch["pixel_values"].to(model.device)
            
            # Sample noise and timesteps
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                model.noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=model.device
            )
            
            # Add noise according to schedule
            noisy_images = model.noise_scheduler.add_noise(
                clean_images,
                noise,
                timesteps
            )
            
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
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        
        # Save checkpoint if needed
        if epoch % checkpoint_freq == 0:
            checkpoint_path = f"{run_name}-checkpoint-epoch-{epoch}.pt"
            model.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PixelArtDiffusion model")
    parser.add_argument("--image-size", type=int, default=16, help="Size of the images")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="Checkpoint frequency in epochs")
    parser.add_argument("--run-name", type=str, default="pixel_art_diffusion", help="Name for the training run")
    
    args = parser.parse_args()
    
    # Initialize model
    model = PixelArtDiffusion(
        image_size=args.image_size,
        num_channels=args.num_channels
    )
    
    # Here you would load your dataset and create dataloader
    # dataloader = create_dataloader()  # You need to implement this
    
    # Train the model
    # train_model(
    #     dataloader=dataloader,
    #     model=model,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     checkpoint_freq=args.checkpoint_freq,
    #     run_name=args.run_name
    # )