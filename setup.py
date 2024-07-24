from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


setup(
    name='terrain_consistency',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'jax',
        'jaxlib',
        'flax',
        'optax',
        'numpy',
        'scipy',
        'rasterio',
        'torch',
        'torchvision',
        'wandb',
        'transformers',
        'jax_fid @ git+https://github.com/matthias-wright/jax-fid',
        'scikit-learn',
        'tqdm',
        'pillow',
        'requests',
        'gdown',
        'boto3'
    ],
)
