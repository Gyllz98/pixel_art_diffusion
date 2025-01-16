import torch
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

class PixelArtDiffusion:
    def __init__(
        self,
        image_size=16,
        num_channels=3,
        num_train_timesteps=500,
        batch_size=64,
        num_epochs=100,
        learning_rate=1e-4,
    ):
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Initialize the UNet2DModel - lighter configuration
        self.model = UNet2DModel(
            sample_size=16,           # Your 16x16 images
            in_channels=3,            # RGB
            out_channels=3,
            layers_per_block=2,       # Keeping it light
            block_out_channels=(64, 128),  # Progressive feature scaling but smaller
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
        print(f"Using device: {self.device}")

    def generate_samples(self, num_samples=16):
        """Generate new pixel art samples"""
        self.model.eval()
        
        # Start from random noise
        sample = torch.randn(
            (num_samples, self.num_channels, self.image_size, self.image_size),
            device=self.device
        )
        
        print("Generating samples...")
        # Denoise gradually
        for t in self.noise_scheduler.timesteps:
            if t % 100 == 0:
                print(f"Denoising step {t}")
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
        
        # Create lists to store unique values for each channel
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
        
        print(f"Unique values per channel:")
        print(f"R: {len(r_values)} values")
        print(f"G: {len(g_values)} values")
        print(f"B: {len(b_values)} values")
        
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
            print("Warning: No color analysis performed. Falling back to default quantization.")
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
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.noise_scheduler = DDPMScheduler.from_config(checkpoint['scheduler_config'])
        print(f"Loaded checkpoint from {path}")

    def visualize_samples(self, samples):
        """
        Visualize generated samples in a grid
        
        Args:
            samples: Tensor of shape (batch_size, channels, height, width)
        """
        
        # Ensure the samples are in CPU and correct range
        samples = samples.cpu()
        
        # Create a grid of images
        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
        
        # Convert to numpy and transpose to correct format (H,W,C)
        grid = grid.permute(1, 2, 0).numpy()
        
        # Display using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.show()

    def save_samples(self, samples, path="samples.png"):
        """
        Save generated samples to a file
        
        Args:
            samples: Tensor of shape (batch_size, channels, height, width)
            path: Path to save the image
        """
        
        # Save the grid of images
        save_image(samples, path, nrow=4, normalize=True, value_range=(-1, 1))
        print(f"Saved samples to {path}")