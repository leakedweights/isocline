"""
Download and prepare dataset.

DEM dataset: NASADEM Merged DEM Global 1 arc second
Source: https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.032021.4326.2
"""

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm

import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

import os
import shutil
import random
import json
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def list_tif_files(bucket_name, prefix, s3_resource):
    bucket = s3_resource.Bucket(bucket_name)
    tif_files = []
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith('.tif'):
            tif_files.append(obj.key)
    return tif_files


def download_selected_files(file_list, bucket_name, local, s3_resource):
    bucket = s3_resource.Bucket(bucket_name)
    for file_key in tqdm(file_list):
        target = os.path.join(local, os.path.relpath(file_key, 'NASADEM'))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        bucket.download_file(file_key, target)


def save_file_list(file_list, filename):
    with open(filename, 'w') as f:
        json.dump(file_list, f)


def load_file_list(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# download randomly selected .tif files
def download(limit: Optional[int] = None, destination: str = './data/'):
    endpoint_url = 'https://opentopography.s3.sdsc.edu'
    s3 = boto3.resource('s3', config=Config(
        signature_version=UNSIGNED), endpoint_url=endpoint_url)

    source_bucket = 'raster'
    source_prefix = 'NASADEM'
    file_list_filename = 'dem_samples.json'

    if not os.path.exists(destination):
        os.makedirs(destination)

    tif_files = list_tif_files(source_bucket, source_prefix, s3)

    if limit is not None:
        selected_files = random.sample(tif_files, limit)
    else:
        selected_files = tif_files

    save_file_list(selected_files, file_list_filename)
    download_selected_files(selected_files, source_bucket, destination, s3)

    print(f"Downloaded {len(selected_files)} files to {destination}")


def redownload(destination: str = './data/'):
    endpoint_url = 'https://opentopography.s3.sdsc.edu'
    s3 = boto3.resource('s3', config=Config(
        signature_version=UNSIGNED), endpoint_url=endpoint_url)

    source_bucket = 'raster'
    file_list_filename = 'selected_files.json'

    if not os.path.exists(destination):
        os.makedirs(destination)

    selected_files = load_file_list(file_list_filename)
    download_selected_files(selected_files, source_bucket, destination, s3)

    print(f"Redownloaded {len(selected_files)} files to {destination}")


# preprocessing
def load_and_smooth_elevation(tif_file, sigma=1):
    with rasterio.open(tif_file) as src:
        data = src.read(1)
        data = np.ma.masked_where(data == src.nodata, data)
        data_smoothed = gaussian_filter(data, sigma=sigma)
    return data_smoothed


def slice_dem(data, slice_size):
    slices = []
    rows, cols = data.shape
    for i in range(0, rows, slice_size):
        for j in range(0, cols, slice_size):
            slice_data = data[i:i+slice_size, j:j+slice_size]
            if slice_data.shape == (slice_size, slice_size):
                slices.append(slice_data)
    return slices


def save_slices(slices, output_dir, base_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, slice_data in enumerate(slices):
        output_path = os.path.join(
            output_dir, f"{base_filename}_slice_{idx}.tif")
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=slice_data.shape[0],
            width=slice_data.shape[1],
            count=1,
            dtype=slice_data.dtype
        ) as dst:
            dst.write(slice_data, 1)


def process_slices(sample_paths, output_dir, slice_size):
    for sample_path in tqdm(sample_paths):
        elevation_data = load_and_smooth_elevation(sample_path, sigma=2)
        sliced_frames = slice_dem(elevation_data, slice_size)
        save_slices(sliced_frames, output_dir, os.path.splitext(
            os.path.basename(sample_path)))


def filter_nonzero(tif_file):
    try:
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            data = np.ma.masked_where(data == src.nodata, data)
        return tif_file if data.max() > 0 else None
    except Exception as e:
        print(f"Error processing file {tif_file}: {e}")
        return None


def copy_file(src, dest_dir):
    try:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy(src, dest_dir)
    except Exception as e:
        print(f"Error copying file {src} to {dest_dir}: {e}")


def filter_and_copy_files(tif_paths, dest_dir, num_workers=8):
    filtered_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(
            filter_nonzero, tif_file): tif_file for tif_file in tif_paths}
        for future in tqdm(as_completed(futures), total=len(futures)):
            tif_file = future.result()
            if tif_file:
                filtered_files.append(tif_file)
                copy_file(tif_file, dest_dir)
    return filtered_files

def calculate_entropy(tif_file: str) -> Tuple[str, float]:
    try:
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            data = np.ma.masked_where(data == src.nodata, data).compressed()
        
        hist, bin_edges = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        
        return tif_file, entropy
    except Exception as e:
        print(f"Error processing file {tif_file}: {e}")
        return tif_file, 0

def rank_dem_samples_by_entropy(tif_files: List[str], num_workers: int = 8) -> List[Tuple[str, float]]:
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(calculate_entropy, tif_file): tif_file for tif_file in tif_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result[1] > 0:
                results.append(result)
    return sorted(results, key=lambda x: x[1], reverse=True)

# run script
if __name__ == "__main__":
    data_dir = "./raw_data"
    sample_dir = f"{data_dir}/NASADEM_be"
    slice_dir = "./slices"
    filtered_directory = './filtered_dems'
    dataset_directory = "./dataset_samples"
    slice_size = 256
    num_files = 2048
    dataset_size = 100_000

    download(limit=num_files, destination=data_dir)

    samples = os.listdir(sample_dir)
    sample_paths = [os.path.join(sample_dir, sample) for sample in samples]

    process_slices(sample_paths, slice_dir, slice_size)

    slices = os.listdir(slice_dir)
    slice_paths = [os.path.join(slice_dir, slice) for slice in slices]
    filtered_slice_paths = filter_and_copy_files(
        slice_paths, filtered_directory)

    files_by_entropy = rank_dem_samples_by_entropy(filtered_slice_paths)
    save_file_list(files_by_entropy[:dataset_size], dataset_directory)
