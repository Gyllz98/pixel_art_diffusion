import torch
from diffusers import DDPMScheduler, UNet2DModel

class PixelArtDiffusion:
    def __init__(
        self,
        image_size=16,
        num_channels=3,
        num_train_timesteps=500,
    ):
        self.image_size = image_size
        self.num_channels = num_channels
        
        # Initialize the UNet2DModel - lighter configuration
        self.model = UNet2DModel(
            sample_size=image_size,
            in_channels=num_channels,
            out_channels=num_channels,
            layers_per_block=2,
            block_out_channels=(64, 128),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D")
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=1e-4,
            beta_end=0.02,
            beta_schedule="linear",
            clip_sample=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_samples(self, num_samples=16):
        """Generate new pixel art samples"""
        self.model.eval()
        
        # Start from random noise
        sample = torch.randn(
            (num_samples, self.num_channels, self.image_size, self.image_size),
            device=self.device
        )
        
        # Denoise gradually
        for t in self.noise_scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(sample, t).sample
                step_output = self.noise_scheduler.step(
                    noise_pred,
                    t,
                    sample
                )
                sample = step_output.prev_sample
                
                # Apply pixel art quantization periodically
                if t % 100 == 0:
                    sample = self.quantize_to_pixel_art(sample)
        
        # Final quantization
        sample = self.quantize_to_pixel_art(sample)
        return sample

    def analyze_color_distribution(self, dataloader):
        """Analyze the unique colors used in the dataset."""
        print("Analyzing color distribution in dataset...")
        
        r_values = set()
        g_values = set()
        b_values = set()
        
        for batch in dataloader:
            images = batch["pixel_values"]
            # Convert to [0,1] range if in [-1,1]
            if images.min() < 0:
                images = (images + 1) / 2
                
            # Collect unique values per channel
            r_values.update(images[:, 0, :, :].unique().tolist())
            g_values.update(images[:, 1, :, :].unique().tolist())
            b_values.update(images[:, 2, :, :].unique().tolist())
        
        # Store the actual values
        self.color_levels = {
            'r': sorted(list(r_values)),
            'g': sorted(list(g_values)),
            'b': sorted(list(b_values))
        }
        return self.color_levels

    def quantize_to_pixel_art(self, sample, use_dataset_colors=True):
        """Quantize the sample to match the dataset's color palette."""
        if not hasattr(self, 'color_levels'):
            return self._default_quantize(sample)
            
        # Convert to [0,1] range
        sample = (sample + 1) / 2
        
        # Quantize each channel separately using the analyzed levels
        for i, channel in enumerate(['r', 'g', 'b']):
            levels = torch.tensor(self.color_levels[channel], device=sample.device)
            channel_data = sample[:, i:i+1]
            
            # For each pixel, find the closest value in our color levels
            distances = torch.abs(channel_data.unsqueeze(-1) - levels)
            closest_level_idx = torch.argmin(distances, dim=-1)
            sample[:, i:i+1] = levels[closest_level_idx]
        
        # Convert back to [-1,1] range
        sample = sample * 2 - 1
        return sample

    def _default_quantize(self, sample, num_colors=16):
        """Original quantization method as fallback."""
        sample = (sample + 1) / 2
        sample = torch.floor(sample * (num_colors - 1)) / (num_colors - 1)
        sample = sample * 2 - 1
        return sample

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scheduler_config': self.noise_scheduler.config
        }, path)

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.noise_scheduler = DDPMScheduler.from_config(checkpoint['scheduler_config'])