import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualize_samples(samples, save_path=None):
    """
    Visualize and optionally save generated samples in a grid
    
    Args:
        samples: Tensor of shape (batch_size, channels, height, width)
        save_path: Optional path to save the visualization
    """
    
    # Ensure the samples are in CPU and correct range
    samples = samples.cpu()
    
    # Create a grid of images
    grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    
    # Convert to numpy and transpose to correct format (H,W,C)
    grid = grid.permute(1, 2, 0).numpy()
    
    # Invert the images (unnegative)
    grid = 1.0 - grid
    
    # Display using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved visualization to {save_path}")
    
    plt.show()