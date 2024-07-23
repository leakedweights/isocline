#!/usr/bin/python3

import os
import argparse

import jax
import optax
from jax import random
from torch.utils.data import DataLoader

from config import model_config, trainer_config, consistency_config
from training import dataloader
from models.unet import UNet
from training.trainer import ConsistencyTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float,
                        default=0.0002, help='Learning rate for the optimizer')
    parser.add_argument('--elevation-zip', type=str,
                        required=True, help='Path to the DEM dataset')
    parser.add_argument('--context-zip', type=str,
                        required=True, help='Path to the context dataset')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='./checkpoints', help='Path to save the trained model')
    parser.add_argument('--snapshot-dir', type=str,
                        default='./snapshots', help='Path to save the generation snapshots')
    parser.add_argument('--eval-dir', type=str,
                        default='./eval', help='Path to save evaluation images')
    parser.add_argument('--empty-context-file', type=str,
                        default='empty_context.npy', help='Filename of the empty context tensor (in context-zip)')

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
    dataset = dataloader.ZippedTerrainDataset(
        args.elevation_zip, args.context_zip, args.empty_context_file)

    terrain_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataloader.numpy_collate,)

    trainer_cfg = trainer_config
    model = UNet(**model_config)
    optimizer = optax.radam(trainer_cfg["learning_rate"])

    random_key = random.PRNGKey(0)
    trainer = ConsistencyTrainer(random_key,
                                 model=model,
                                 optimizer=optimizer,
                                 dataloader=terrain_dataloader,
                                 img_shape=(64, 64, 1),
                                 num_devices=jax.local_device_count(),
                                 config=trainer_cfg,
                                 consistency_config=consistency_config)

    trainer.train(args.steps)


def main():
    args = parse_args()
    preprocess_args(args)
    train(args)


if __name__ == "__main__":
    main()
