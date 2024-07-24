import os
from PIL import Image
import torch
import zipfile
from tqdm import tqdm
import rasterio
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import Dataset, default_collate
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Lambda, ToPILImage
from typing import Optional


class ZippedTerrainDataset(Dataset):
    def __init__(self,
                 elevation_zip: str,
                 context_zip: str,
                 empty_context_filename: str,
                 terrain_file_type: str = "tif",
                 context_file_type: str = "npy",
                 files: Optional[list] = None,
                 **kwargs):

        super().__init__(**kwargs)

        self.elevation_zip = zipfile.ZipFile(elevation_zip)
        self.context_zip = zipfile.ZipFile(context_zip)
        self.empty_context_filename = empty_context_filename
        self.terrain_file_type = terrain_file_type
        self.context_file_type = context_file_type

        if files is None:
            self.files = [f for f in self.elevation_zip.namelist(
            ) if f.endswith(self.terrain_file_type)]
        else:
            self.files = files

        with self.context_zip.open(empty_context_filename) as file:
            self.empty_context_data = np.load(file)

    def __len__(self):
        return len(self.files)

    def open_tif(self, bytes_data):
        with rasterio.MemoryFile(bytes_data) as memfile:
            with memfile.open() as src:
                data = src.read(1)
        return data

    def __getitem__(self, idx: int):
        target_elevation_file = self.files[idx]
        target_context_file = target_elevation_file.replace(
            self.terrain_file_type, self.context_file_type)

        with self.elevation_zip.open(target_elevation_file) as file:
            elevation_data = file.read()
            elevation_array = self.open_tif(elevation_data)

        try:
            with self.context_zip.open(target_context_file) as file:
                context_data = np.load(file)
        except KeyError:
            context_data = self.empty_context_data

        if elevation_array.shape != 3:
            elevation_array = np.expand_dims(elevation_array, -1)

        return elevation_array, context_data

    def __del__(self):
        self.elevation_zip.close()
        self.context_zip.close()


def split_dataset(elevation_zip: str, context_zip: str, test_ratio: float = 0.1):
    with zipfile.ZipFile(elevation_zip) as ezip, zipfile.ZipFile(context_zip) as czip:
        terrain_files = [f for f in ezip.namelist() if f.endswith('tif')]
        context_files = {f.replace('npy', 'tif')
                         for f in czip.namelist() if f.endswith('npy')}

        files_with_context = [f for f in terrain_files if f in context_files]
        files_without_context = [
            f for f in terrain_files if f not in context_files]

        total_files = len(terrain_files)
        test_size = int(total_files * test_ratio)
        train_size = total_files - test_size

        if len(files_with_context) <= train_size:
            train_files = files_with_context
            remaining_train_slots = train_size - len(train_files)

            additional_train_files, eval_files = train_test_split(
                files_without_context, test_size=test_size)
            train_files.extend(additional_train_files[:remaining_train_slots])
        else:
            train_files, eval_files_with_context = train_test_split(
                files_with_context, train_size=train_size)
            eval_files = eval_files_with_context + \
                files_without_context[:test_size -
                                      len(eval_files_with_context)]

        return train_files, eval_files


def numpy_collate(batch):
    batch = default_collate(batch)
    batch = tree_map(lambda x: np.asarray(x), batch)
    return batch


def save_normalize_eval_dataset(dataset: Dataset, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for idx, (elevation_array, _) in enumerate(tqdm(dataset, desc="Saving normalized dataset")):
        abs_max = np.max(np.abs(elevation_array))
        if abs_max != 0:
            normalized_data = elevation_array / abs_max
        else:
            normalized_data = elevation_array

        if normalized_data.ndim == 3 and normalized_data.shape[2] == 1:
            normalized_data = normalized_data.squeeze(axis=2)

        normalized_data = (normalized_data * 255).astype(np.uint8)

        output_path = os.path.join(output_dir, f"normalized_{idx}.png")
        Image.fromarray(normalized_data).save(output_path)


reverse_transform = Compose([
    Lambda(lambda x: torch.from_numpy(np.asarray(x))),
    Lambda(lambda x: x.permute(2, 0, 1)),
    Lambda(lambda x: 0.5 * (x - 1.0)),
    ToPILImage()
])
