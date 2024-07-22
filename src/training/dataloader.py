import torch
import zipfile
import rasterio
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import default_collate, Dataset
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage


import zipfile
from torch.utils.data import Dataset
import io


class ZippedTerrainDataset(Dataset):
    def __init__(self,
                 elevation_zip: str,
                 context_zip: str,
                 empty_context_filename: str,
                 terrain_file_type: str = "tif",
                 context_file_type: str = "npy",
                 **kwargs):

        super().__init__(**kwargs)

        self.elevation_zip = zipfile.ZipFile(elevation_zip)
        self.context_zip = zipfile.ZipFile(context_zip)
        self.empty_context_filename = empty_context_filename
        self.terrain_file_type = terrain_file_type
        self.context_file_type = context_file_type

        self.files = [f for f in self.elevation_zip.namelist(
        ) if f.endswith(self.terrain_file_type)]

        with self.context_zip.open(empty_context_filename) as file:
            self.empty_context_data = file.read()

    def __len__(self):
        return len(self.files)

    def open_tif(self, bytes_data):
        with rasterio.open(io.BytesIO(bytes_data)) as src:
            data = src.read(1)
            data = np.ma.masked_where(data == src.nodata, data).compressed()
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
                context_data = file.read()
        except KeyError:
            context_data = self.empty_context_data

        return elevation_array, io.BytesIO(context_data)

    def __del__(self):
        self.elevation_zip.close()
        self.context_zip.close()


def numpy_collate(batch):
    batch = default_collate(batch)
    batch = tree_map(lambda x: np.asarray(x), batch)
    return batch


transform = Compose([
    ToTensor(),
    Lambda(lambda x: x.permute(1, 2, 0)),
    Lambda(lambda x: x * 2 - 1),
])

reverse_transform = Compose([
    Lambda(lambda x: torch.from_numpy(np.asarray(x))),
    Lambda(lambda x: x.permute(2, 0, 1)),
    Lambda(lambda x: 0.5 * (x - 1)),
    ToPILImage()
])
