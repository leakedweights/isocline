import os
import jax
import flax
from jax import random
import jax.numpy as jnp
from flax import linen as nn
from flax.jax_utils import replicate, unreplicate
from flax.training import train_state, checkpoints

from .fid import FID
from ..utils import cast_dim, update_ema
from .dataloader import reverse_transform
from .upload import upload_file
from ..components.consistency_utils import *

import wandb
from tqdm import trange
from functools import partial
from typing import Any, Optional


@flax.struct.dataclass
class TrainState(train_state.TrainState):
    ema_params: Any


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(6, 7, 8))
def train_step(random_key: Any,
               state: train_state.TrainState,
               x: jax.Array,
               context: jax.Array,
               t1: jax.Array,
               t2: jax.Array,
               sigma_data: float,
               sigma_min: float,
               huber_const: float):

    noise_key, dropout_t1, dropout_t2 = random.split(random_key, 3)
    data_dim = jnp.prod(jnp.array(x.shape[1:]))
    c_data = huber_const * jnp.sqrt(data_dim)

    noise = random.normal(noise_key, x.shape)

    @jax.jit
    def loss_fn(params):
        t1_noise_dim = cast_dim(t1, noise.ndim)
        t2_noise_dim = cast_dim(t2, noise.ndim)

        xt1_raw = x + t1_noise_dim * noise
        xt2_raw = x + t2_noise_dim * noise

        _, xt1 = jax.lax.stop_gradient(consistency_fn(
            xt1_raw, context, t1, sigma_data, sigma_min, partial(state.apply_fn, rngs={"dropout": dropout_t1}), params, train=True))
        _, xt2 = consistency_fn(
            xt2_raw, context, t2, sigma_data, sigma_min, partial(state.apply_fn, rngs={"dropout": dropout_t2}), params, train=True)

        loss = pseudo_huber_loss(xt2, xt1, c_data)
        weight = cast_dim((1 / (t2 - t1)), loss.ndim)

        return jnp.mean(weight * loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    loss = jax.lax.pmean(loss, "batch")

    state = state.apply_gradients(grads=grads)
    return state, loss


class ConsistencyTrainer:

    def __init__(self,
                 random_key: Any,
                 model: nn.Module,
                 optimizer: Any,
                 dataloader: Any,
                 img_shape,
                 num_devices: int,
                 batch_size: int,
                 config: dict,
                 consistency_config: dict):

        self.model = model
        self.config = config
        self.checkpoint_step = 0
        self.consistency_config = consistency_config
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.random_key, self.snapshot_key, init_key = random.split(
            random_key, 3)

        assert batch_size % num_devices == 0, "Batch size must be divisible by the number of devices!"
        self.num_devices = num_devices
        device_batch_size = batch_size // num_devices
        self.device_batch_shape = (device_batch_size, *img_shape)

        init_input = jnp.ones(self.device_batch_shape)
        init_context = jnp.ones(
            (device_batch_size, *self.config["context_dim"]))
        init_time = jnp.ones((device_batch_size,))
        model_params = model.init(
            init_key, init_input, init_context, init_time, train=True)

        if self.config["run_evals"]:
            image_shape = self.device_batch_shape[1:]
            self.fid = FID(image_shape, self.config["reference_dir"])
            self.fid.precompute()

        self.state = TrainState.create(
            apply_fn=model.apply, params=model_params, ema_params=model_params, tx=optimizer)

    def train(self, train_steps: int):
        parallel_state = replicate(self.state)
        iterator = iter(self.dataloader)

        if self.config["run_evals"]:
            fid_score = self.run_eval()
            wandb.log({"fid_score": fid_score}, step=self.checkpoint_step)

        with trange(self.checkpoint_step, train_steps, initial=self.checkpoint_step, total=train_steps) as steps:
            cumulative_loss = 0.0
            for step in steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    continue

                x_batch, context_batch = batch
                if x_batch.shape != 4:
                    x_batch = jnp.expand_dims(x_batch, -1)

                _, *data_dim = x_batch.shape
                _, *context_dim = context_batch.shape

                # normalize batch and rescale to [-1, 1]
                if self.config["batch_rescale"]:
                    min_vals = jnp.min(x_batch, axis=tuple(
                        range(1, len(data_dim)+1)), keepdims=True)
                    max_vals = jnp.max(x_batch, axis=tuple(
                        range(1, len(data_dim)+1)), keepdims=True)

                    range_vals = max_vals - min_vals
                    range_vals = jnp.where(range_vals == 0, 1, range_vals)

                    x_batch = (x_batch - min_vals) / range_vals

                x_batch = 2 * x_batch - 1

                x_parallel = x_batch.reshape(self.num_devices, -1, *data_dim)
                context_parallel = context_batch.reshape(
                    self.num_devices, -1, *context_dim)

                self.random_key, schedule_key, *device_keys = random.split(
                    self.random_key, self.num_devices + 2)

                device_keys = jnp.array(device_keys)

                config = self.consistency_config

                N = discretize(
                    step, config["s0"], config["s1"], self.config["max_steps"])

                noise_levels = get_boundaries(
                    N, config["sigma_min"], config["sigma_max"], config["rho"])

                t1, t2 = sample_timesteps(
                    schedule_key, noise_levels, x_parallel.shape[:2], config["p_mean"], config["p_std"])

                parallel_state, parallel_loss = train_step(
                    device_keys,
                    parallel_state,
                    x_parallel,
                    context_parallel,
                    t1, t2,
                    config["sigma_data"],
                    config["sigma_min"],
                    config["huber_const"]
                )

                if self.config["use_ema"]:
                    parallel_state = parallel_state.replace(
                        ema_params=update_ema(
                            parallel_state.ema_params,
                            parallel_state.params,
                            self.config["ema_decay"])
                    )

                loss = unreplicate(parallel_loss)
                steps.set_postfix(loss=loss)
                cumulative_loss += loss
                log_freq = self.config["log_frequency"]

                if ((step + 1) % log_freq == 0) and self.config["log_wandb"]:
                    avg_loss = cumulative_loss / log_freq
                    cumulative_loss = 0
                    wandb.log({"train_loss": avg_loss}, step=step)

                save_checkpoint = (
                    step + 1) % self.config["checkpoint_frequency"] == 0
                save_snapshot = self.config["create_snapshots"] and (
                    step + 1) % self.config["snapshot_frequency"] == 0
                self._save(
                    parallel_state, step, save_checkpoint, save_snapshot)

                run_eval = self.config["run_evals"] and (
                    step + 1) % self.config["eval_frequency"] == 0
                if run_eval:
                    self.state = unreplicate(parallel_state)
                    fid_score = self.run_eval()
                    wandb.log({"fid_score": fid_score}, step=step)

        self._save(
            parallel_state, train_steps, save_checkpoint=True, save_snapshot=True)

    def _save(self, parallel_state, step, save_checkpoint, save_snapshot):
        if not (save_checkpoint or save_snapshot):
            return

        self.state = unreplicate(parallel_state)
        if save_checkpoint:
            self.save_checkpoint(step)
        if save_snapshot:
            self.save_snapshot(step)

    def save_snapshot(self, step):
        outputs = self.generate(self.snapshot_key, context=None)

        pillow_outputs = [reverse_transform(output) for output in outputs]

        os.makedirs(self.config["snapshot_dir"], exist_ok=True)

        for idx, output in enumerate(pillow_outputs[:self.config["samples_to_keep"]]):
            fpath = f"{self.config['snapshot_dir']}/img_it{step+1}_n{idx + 1}.png"
            output.save(fpath)

    def save_checkpoint(self, step):
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        ckpt_file = checkpoints.save_checkpoint(self.config["checkpoint_dir"],
                                                target={
                                                    "state": self.state, "step": step},
                                                step=step,
                                                overwrite=True,
                                                keep=self.config["checkpoints_to_keep"])

        if self.config["upload_to_s3"]:
            upload_file(ckpt_file, self.config["s3_bucket"])

        self.checkpoint_step = step

    def load_checkpoint(self):
        target = {"state": self.state, "step": 0}
        try:
            restored = checkpoints.restore_checkpoint(
                ckpt_dir=self.config["checkpoint_dir"], target=target)
            self.checkpoint_step = int(restored["step"])
            self.state = restored["state"]
        except Exception as e:
            print(f"Unable to load checkpoint: {e}")

    def generate(self, key, context: Optional[jax.Array] = None):
        if self.config["use_ema"]:
            generation_params = self.state.ema_params
        else:
            generation_params = self.state.params

        if context is None:
            empty_context = self.config["empty_context"]
            context = jnp.repeat(jnp.expand_dims(
                empty_context, axis=0), self.device_batch_shape[0], axis=0)

        return sample_single_step(key,
                                  self.state.apply_fn,
                                  generation_params,
                                  self.device_batch_shape,
                                  self.consistency_config["sigma_data"],
                                  self.consistency_config["sigma_min"],
                                  self.consistency_config["sigma_max"],
                                  context=context)

    def generate_cfg(self, key, context):
        cond_key, uncond_key = random.split(key)

        x_cond = self.generate(cond_key, context)
        x_uncond = self.generate(uncond_key)

        return x_cond + self.config["guidance_scale"] * (x_cond - x_uncond)

    def run_eval(self):
        eval_dir = self.config["synthetic_dir"]
        os.makedirs(eval_dir, exist_ok=True)
        num_samples = self.config["num_eval_samples"]

        sample_key = random.key(0)

        i = 0

        while i < num_samples:
            sample_key, subkey = random.split(sample_key)
            samples = self.generate(sample_key)

            pillow_outputs = [reverse_transform(
                output) for output in samples[:min(num_samples - i, len(samples))]]
            for idx, output in enumerate(pillow_outputs):
                fpath = f"{eval_dir}/{i + idx}.png"
                output.save(fpath)

            i += len(pillow_outputs)

        fid_score = self.fid.compute(eval_dir)

        return fid_score
