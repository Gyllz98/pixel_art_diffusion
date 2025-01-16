import torch
from torch import nn
from diffusers.optimization import get_scheduler
from tqdm import tqdm

def train(model, dataloader, run_name="pixel_art_diffusion"):
        """Train the model"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model.learning_rate,
            weight_decay=0.01,
            eps=1e-8
        )
        
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=1000,
            num_training_steps=len(dataloader) * model.num_epochs
        )

        print(f"Starting training for {model.num_epochs} epochs...")
        
        # Main training loop with tqdm
        for epoch in tqdm(range(model.num_epochs), desc="Training epochs"):
            model.train()
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
                noise_pred = model(noisy_images, timesteps).sample
                
                # Calculate loss
                loss = nn.functional.mse_loss(noise_pred, noise)
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            
            if epoch % 10 == 0:
                model.save_checkpoint(f"checkpoint-epoch-{epoch}.pt")