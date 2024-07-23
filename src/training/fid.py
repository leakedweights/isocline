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
    A utility for using jax-fid.
    Based on https://github.com/matthias-wright/jax-fid/blob/main/jax_fid/fid.py
    """

    def __init__(self, img_size: tuple, reference_dir: str, batch_size: int = 128, scale_to: int = 256):
        self.batch_size = batch_size
        self.img_size = img_size
        self.reference_dir = reference_dir
        self.ref_mu = None
        self.ref_sigma = None
        self.scale_to = scale_to

        rng = jax.random.PRNGKey(0)

        model = inception.InceptionV3(pretrained=True)
        self.params = model.init(rng, jnp.ones((1, 256, 256, 3)))

        self.apply_fn = jax.jit(partial(model.apply, train=False))

    def precompute(self, save_dir=None):
        mu, sigma = jf.compute_statistics(
            self.reference_dir, self.params, self.apply_fn, self.batch_size, self.img_size)
        self.ref_mu, self.ref_sigma = mu, sigma

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.savez(os.path.join(save_dir, 'stats'), mu=mu, sigma=sigma)

    def compute(self, synthetic_dir):

        if self.ref_mu is None or self.ref_sigma is None:
            self.precompute()

        mu2, sigma2 = self.compute_statistics(
            synthetic_dir, self.params, self.apply_fn, self.batch_size, self.img_size)
        print(
            f"Ref - mu: {self.ref_mu}, sigma: {self.ref_sigma}; Synth - mu: {mu2}, sigma: {sigma2}")
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
            img = Image.open(os.path.join(path, f))

            if img.mode == "L":
                img = img.convert("RGB")

            img = img.resize(
                size=(self.scale_to, self.scale_to),
                resample=Image.BILINEAR,
            )
            img = np.array(img) / 255.0
            images.append(img)

        num_batches = int(len(images) // batch_size)
        act = []
        for i in tqdm(range(num_batches)):
            x = images[i * batch_size: i * batch_size + batch_size]
            x = np.asarray(x)
            x = 2 * x - 1
            pred = apply_fn(params, jax.lax.stop_gradient(x))
            act.append(pred.squeeze(axis=1).squeeze(axis=1))
        act = jnp.concatenate(act, axis=0)

        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma