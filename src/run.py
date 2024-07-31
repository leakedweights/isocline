#!/usr/bin/python3

import os
import argparse

import jax
import wandb
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from jax import random

from .config import model_config, trainer_config, consistency_config
from .training import tf_data
from .training import dataloader
from .models.unet import UNet
from .training.trainer import ConsistencyTrainer

import warnings
from rasterio.errors import NotGeoreferencedWarning

tf.config.set_visible_devices([], device_type='GPU')
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int,
                        default=256, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float,
                        default=0.0002, help='Learning rate for the optimizer')
    parser.add_argument('--dataset-type', type=str,
                        default="dir", help='Path to the DEM dataset')
    parser.add_argument('--elevation-source', type=str,
                        default="data/elevation", help='Path to the DEM dataset')
    parser.add_argument('--context-source', type=str,
                        default="data/context", help='Path to the context dataset')
    parser.add_argument('--checkpoint-dir', type=str,
                        default='/tmp/checkpoints', help='Path to save the trained model')
    parser.add_argument('--snapshot-dir', type=str,
                        default='snapshots', help='Path to save the generation snapshots')
    parser.add_argument('--eval-dir', type=str,
                        default='eval', help='Path to save evaluation images')
    parser.add_argument('--empty-context-file', type=str,
                        default='empty_context.npy', help='Filename of the empty context tensor (in context-zip)')
    parser.add_argument('--wandb-project-name', type=str,
                        default='terrain_consistency', help='Weights and Biases project name')
    parser.add_argument('--load-ckpt', action=argparse.BooleanOptionalAction, default=False,
                        help='Load latest checkpoint in checkpoint-dir.')
    parser.add_argument('--generate', action=argparse.BooleanOptionalAction, default=False,
                        help='Generate samples. Must specify num-samples.')
    parser.add_argument("--num-samples", type=int, help='Number of samples to save. Must be specified when --generate is true.')

    return parser.parse_args()


def preprocess_args(args):
    if not os.path.exists(args.elevation_source):
        raise FileNotFoundError(
            "The specified elevation source does not exist!")
    if not os.path.exists(args.context_source):
        raise FileNotFoundError("The specified context source does not exist!")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.generate and not args.num_samples:
        raise Exception("When using the --generate flag, you \
                        must specify the number of samples to obtain.")


def train(args):
    config = trainer_config
    model = UNet(**model_config)
    optimizer = optax.radam(config["learning_rate"])

    config["reference_dir"] = f"{args.eval_dir}/reference"
    config["synthetic_dir"] = f"{args.eval_dir}/synthetic"
    config["checkpoint_dir"] = args.checkpoint_dir
    config["snapshot_dir"] = args.snapshot_dir
    config["eval_dir"] = args.eval_dir

    img_shape = (64, 64, 1)

    if args.dataset_type == "dir":
        train_files, eval_files = dataloader.split_dir_dataset(
            args.elevation_source, args.context_source)
        train_dataset = tf_data.get_preheated_dataset(
            batch_size=args.batch_size,
            elevation_dir=args.elevation_source,
            context_dir=args.context_source,
            img_shape=img_shape[:2],
            emb_shape=config["context_dim"],
            included_files=train_files,
            empty_context_filename=args.empty_context_file)
        eval_dataset = dataloader.DirectoryTerrainDataset(
            args.elevation_source, args.context_source, args.empty_context_file, files=eval_files)
    else:
        raise Exception("Invalid dataset type!")

    if not os.path.exists(config["reference_dir"]) or len(os.listdir(config["reference_dir"])) == 0:
        dataloader.save_normalize_eval_dataset(
            eval_dataset, config["reference_dir"])

    config["empty_context"] = np.load(os.path.join(
        args.context_source, args.empty_context_file)).astype(np.float32)

    wandb.init(
        project=args.wandb_project_name,
        id="terrainict",
        config={
            "model": model_config,
            "trainer": config
        },
        resume="allow"
    )

    random_key = random.PRNGKey(0)
    trainer = ConsistencyTrainer(random_key,
                                 model=model,
                                 optimizer=optimizer,
                                 dataloader=tfds.as_numpy(train_dataset),
                                 img_shape=img_shape,
                                 batch_size=args.batch_size,
                                 num_devices=jax.local_device_count(),
                                 config=config,
                                 consistency_config=consistency_config)

    if args.load_ckpt:
        trainer.load_checkpoint()

    if args.generate:
        i = 0
        while i < args.num_samples:
            sample_key, subkey = random.split(sample_key)
            samples = trainer.generate_cfg(sample_key)

            pillow_outputs = [dataloader.reverse_transform(
                output) for output in samples[:min(args.num_samples - i, len(samples))]]
            for idx, output in enumerate(pillow_outputs):
                fpath = f"{args.eval_dir}/{i + idx}.png"
                output.save(fpath)

            i += len(pillow_outputs)
    else:
        trainer.train(args.steps)


def main():
    args = parse_args()
    preprocess_args(args)
    train(args)


if __name__ == "__main__":
    main()
