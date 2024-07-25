#!/usr/bin/python3

import os
import argparse

import jax
import wandb
import optax
from jax import random
from torch.utils.data import DataLoader

from .config import model_config, trainer_config, consistency_config
from .training import dataloader
from .models.unet import UNet
from .training.trainer import ConsistencyTrainer

import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float,
                        default=0.0002, help='Learning rate for the optimizer')
    parser.add_argument('--elevation-zip', type=str,
                        default="data/elevation.zip", help='Path to the DEM dataset')
    parser.add_argument('--context-zip', type=str,
                        default="data/context.zip", help='Path to the context dataset')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='checkpoints', help='Path to save the trained model')
    parser.add_argument('--snapshot-dir', type=str,
                        default='snapshots', help='Path to save the generation snapshots')
    parser.add_argument('--eval-dir', type=str,
                        default='eval', help='Path to save evaluation images')
    parser.add_argument('--empty-context-file', type=str,
                        default='empty_context.npy', help='Filename of the empty context tensor (in context-zip)')
    parser.add_argument('--wandb-project-name', type=str,
                        default='terrain_consistency', help='Weights and Biases project name')

    return parser.parse_args()


def preprocess_args(args):
    if not os.path.exists(args.elevation_zip):
        raise FileNotFoundError(
            "The specified elevation source does not exist!")
    if not os.path.exists(args.context_zip):
        raise FileNotFoundError("The specified context source does not exist!")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)


def train(args):
    config = trainer_config
    model = UNet(**model_config)
    optimizer = optax.radam(config["learning_rate"])

    config["reference_dir"] = f"{args.eval_dir}/reference"
    config["synthetic_dir"] = f"{args.eval_dir}/synthetic"
    config["checkpoint_dir"] = args.checkpoint_dir
    config["snapshot_dir"] = args.snapshot_dir
    config["eval_dir"] = args.eval_dir

    train_files, eval_files = dataloader.split_dataset(
        args.elevation_zip, args.context_zip)
    train_dataset = dataloader.ZippedTerrainDataset(
        args.elevation_zip, args.context_zip, args.empty_context_file, files=train_files)
    eval_dataset = dataloader.ZippedTerrainDataset(
        args.elevation_zip, args.context_zip, args.empty_context_file, files=eval_files)

    if not os.path.exists(config["reference_dir"]) or len(os.listdir(config["reference_dir"])) == 0:
        dataloader.save_normalize_eval_dataset(
            eval_dataset, config["reference_dir"])

    config["empty_context"] = train_dataset.empty_context_data

    terrain_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dataloader.numpy_collate)

    wandb.init(
        project=args.wandb_project_name,
        config={
            "model": model_config,
            "trainer": config
        }
    )

    random_key = random.PRNGKey(0)
    trainer = ConsistencyTrainer(random_key,
                                 model=model,
                                 optimizer=optimizer,
                                 dataloader=terrain_dataloader,
                                 img_shape=(64, 64, 1),
                                 num_devices=jax.local_device_count(),
                                 config=config,
                                 consistency_config=consistency_config)

    trainer.train(args.steps)


def main():
    args = parse_args()
    preprocess_args(args)
    train(args)


if __name__ == "__main__":
    main()
