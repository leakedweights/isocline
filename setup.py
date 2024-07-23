from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        subprocess.check_call(['python', './data/download.py'])


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
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)
