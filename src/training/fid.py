from tqdm import tqdm
import os
import numpy as np
import jax
from PIL import Image
import jax.numpy as jnp
from functools import partial

from jax_fid import inception, fid as jf


class FID:
    """
    A utility class for using jax-fid.
    Based on https://github.com/matthias-wright/jax-fid/blob/main/jax_fid/fid.py
    """

    def __init__(self, img_size: tuple, reference_dir: str, batch_size: int = 512, scale_to: int = 256):
        self.batch_size = batch_size
        self.img_size = img_size
        self.reference_dir = reference_dir
        self.ref_mu = None
        self.ref_sigma = None
        self.scale_to = scale_to

        rng = jax.random.PRNGKey(0)

        model = inception.InceptionV3(pretrained=True)
        self.params = model.init(rng, jnp.ones((1, scale_to, scale_to, 3)))

        self.apply_fn = jax.jit(partial(model.apply, train=False))

    def load_statistics(self, stats_file):
        if os.path.exists(stats_file):
            stats = np.load(stats_file)
            self.ref_mu = stats['mu']
            self.ref_sigma = stats['sigma']
            print(f"Loaded statistics from {stats_file}")
        else:
            raise FileNotFoundError(f"Statistics file {stats_file} not found.")

    def precompute(self):
        stats_file = os.path.join(self.reference_dir, 'stats')
        try:
            self.load_statistics(stats_file)
        except FileNotFoundError:
            mu, sigma = self.compute_statistics(
                self.reference_dir, self.params, self.apply_fn, self.batch_size, self.img_size)
            self.ref_mu, self.ref_sigma = mu, sigma

            os.makedirs(self.reference_dir, exist_ok=True)
            np.savez(stats_file, mu=mu, sigma=sigma)

    def compute(self, synthetic_dir):

        if self.ref_mu is None or self.ref_sigma is None:
            self.precompute()

        mu2, sigma2 = self.compute_statistics(
            synthetic_dir, self.params, self.apply_fn, self.batch_size, self.img_size)

        fid_score = jf.compute_frechet_distance(
            self.ref_mu, mu2, self.ref_sigma, sigma2, eps=1e-3)

        return fid_score

    def compute_statistics(self, path, params, apply_fn, batch_size=1, img_size=None):
        if path.endswith(".npz"):
            stats = np.load(path)
            mu, sigma = stats["mu"], stats["sigma"]
            return mu, sigma

        images = []
        for f in tqdm(os.listdir(path)):
            try:
                img = Image.open(os.path.join(path, f))
                if img.mode == "L":
                    img = img.convert("RGB")
                img = img.resize(
                    size=(self.scale_to, self.scale_to),
                    resample=Image.BILINEAR,
                )
                img = np.array(img) / 255.0
                if img.shape == (self.scale_to, self.scale_to, 3):
                    images.append(img)
                else:
                    print(f"Skipping file {f}: Unexpected shape {img.shape}")
            except Exception as e:
                print(f"Skipping file {f}: {e}")

        if not images:
            raise ValueError("No valid images found in the directory.")

        num_batches = int(len(images) // batch_size)
        act = []
        for i in tqdm(range(num_batches)):
            x = images[i * batch_size: i * batch_size + batch_size]
            x = np.asarray(x)
            x = 2 * x - 1
            pred = apply_fn(params, jax.lax.stop_gradient(x))
            act.append(pred.squeeze(axis=1).squeeze(axis=1))
        act = jnp.concatenate(act, axis=0)

        if act.shape[0] == 0:
            raise ValueError(
                "No activations computed; check image preprocessing.")

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        if np.any(np.isnan(mu)) or np.any(np.isnan(sigma)):
            print(
                f"NaN detected in computed statistics. {mu: {mu}}; sigma: {sigma}")

        return mu, sigma
