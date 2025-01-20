import typer
from typing import Optional
import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from loguru import logger
from diffusers.optimization import get_scheduler
from pixel_art_diffusion.model import PixelArtDiffusion
from pixel_art_diffusion.data import PixelArtDataset


def create_profiler(activities, num_steps, logdir, skip_first=0, wait=1, warmup=1):
    return profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=num_steps, repeat=1, skip_first=skip_first),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        with_flops=True,
        on_trace_ready=tensorboard_trace_handler(str(logdir)),
    )


def profile_pipeline(
    cfg: DictConfig,
    num_train_steps: int = 100,
    num_inference_steps: int = 10,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Profile both training and inference pipeline with improved metrics.
    """
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup directories
    timestamp = int(time.time())
    if output_dir:
        run_dir = output_dir / f"profile_run_{timestamp}"
        train_dir = run_dir / "train"
        inference_dir = run_dir / "inference"
        for dir in [train_dir, inference_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # Add TensorBoard writer for additional metrics
        writer = SummaryWriter(str(run_dir))
    else:
        run_dir = None
        writer = None

    # Configure profiler activities
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Model and dataset setup
    logger.info(f"Setting up model and dataset on {device}")
    model = PixelArtDiffusion(device=device)
    dataset = PixelArtDataset(data_path=cfg.data.root_path, calculate_stats=False, label_subset=[3])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers
    )

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.model.parameters(), **cfg.optimizer.params)
    lr_scheduler = get_scheduler(
        name=cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.num_warmup_steps,
        num_training_steps=num_train_steps,
    )

    profiling_results = {"device": device}

    # Training profiling
    logger.info("Starting training profiling...")
    train_prof = create_profiler(activities, num_train_steps, train_dir)

    with train_prof as prof:
        model.model.train()
        train_times = []

        for batch_idx, batch in enumerate(tqdm(dataloader, total=num_train_steps)):
            if batch_idx >= num_train_steps:
                break

            start_time = time.perf_counter()

            clean_images = batch["pixel_values"].to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, model.noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device
            )

            noisy_images = model.noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model.model(noisy_images, timesteps).sample
            loss = nn.functional.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), cfg.training.clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            train_times.append(end_time - start_time)

            if writer:
                writer.add_scalar("Train/Loss", loss.item(), batch_idx)
                writer.add_scalar("Train/Step_Time", train_times[-1], batch_idx)

            prof.step()

    # Inference profiling
    logger.info("Starting inference profiling...")
    inference_prof = create_profiler(activities, num_inference_steps, inference_dir)

    with inference_prof as prof:
        model.model.eval()
        inference_times = []

        with torch.no_grad():
            for i in tqdm(range(num_inference_steps)):
                start_time = time.perf_counter()

                model.generate_samples(1)

                if device == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)

                if writer:
                    writer.add_scalar("Inference/Step_Time", inference_times[-1], i)

                prof.step()

    # Compile detailed results
    profiling_results.update(
        {
            "training": {
                "steps": num_train_steps,
                "avg_step_time": sum(train_times) / len(train_times) if train_times else 0,
                "min_step_time": min(train_times) if train_times else 0,
                "max_step_time": max(train_times) if train_times else 0,
                "memory_stats": {
                    "cpu_memory": str(train_prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
                },
            },
            "inference": {
                "steps": num_inference_steps,
                "avg_step_time": sum(inference_times) / len(inference_times) if inference_times else 0,
                "min_step_time": min(inference_times) if inference_times else 0,
                "max_step_time": max(inference_times) if inference_times else 0,
                "memory_stats": {
                    "cpu_memory": str(
                        inference_prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)
                    )
                },
            },
        }
    )

    if device == "cuda":
        profiling_results["training"]["memory_stats"]["cuda_memory"] = str(
            train_prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
        )
        profiling_results["inference"]["memory_stats"]["cuda_memory"] = str(
            inference_prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
        )

    # Save results
    if run_dir:
        stats_file = run_dir / "pipeline_profile.json"
        with open(stats_file, "w") as f:
            json.dump(profiling_results, f, indent=2)

        logger.info(f"All profiling results saved to {run_dir}")

        if writer:
            writer.close()

    return profiling_results


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Profile model training and inference pipeline."""
    try:
        output_dir = Path("profiling_results")
        output_dir.mkdir(exist_ok=True)

        # Increase steps for more meaningful profiling
        stats = profile_pipeline(
            cfg=cfg,
            num_train_steps=3,  # Increased from 2
            num_inference_steps=2,  # Increased from 2
            output_dir=output_dir,
            device=None,
        )

        # Print summary
        logger.info("\nProfiling Summary:")
        logger.info(f"Device: {stats['device']}")

        for phase in ["training", "inference"]:
            logger.info(f"\n{phase.capitalize()} Profile:")
            logger.info(f"Steps: {stats[phase]['steps']}")
            logger.info(f"Average step time: {stats[phase]['avg_step_time']:.4f}s")
            logger.info(f"Min step time: {stats[phase]['min_step_time']:.4f}s")
            logger.info(f"Max step time: {stats[phase]['max_step_time']:.4f}s")
            logger.info(f"\nTop CPU Memory Usage ({phase.capitalize()}):")
            logger.info(stats[phase]["memory_stats"]["cpu_memory"])

            if "cuda_memory" in stats[phase]["memory_stats"]:
                logger.info(f"\nTop CUDA Memory Usage ({phase.capitalize()}):")
                logger.info(stats[phase]["memory_stats"]["cuda_memory"])

    except Exception as e:
        logger.error(f"Error during profiling: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
