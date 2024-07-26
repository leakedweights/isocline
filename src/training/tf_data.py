import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import rasterio


class TextToTerrainDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for elevations and embeddings dataset."""
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, elevation_dir, embedding_dir, elevation_shape, embedding_shape, file_list, empty_context_filename, **kwargs):
        self.elevation_dir = elevation_dir
        self.embedding_dir = embedding_dir
        self.elevation_shape = elevation_shape
        self.embedding_shape = embedding_shape
        self.empty_context_path = os.path.join(
            embedding_dir, empty_context_filename)
        self.file_list = file_list
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "Dataset consisting of elevation maps in .tif format and corresponding text embeddings."),
            features=tfds.features.FeaturesDict({
                'elevation': tfds.features.Tensor(shape=self.elevation_shape, dtype=tf.float32),
                'embedding': tfds.features.Tensor(shape=self.embedding_shape, dtype=tf.float32),
            }),
            supervised_keys=('elevation', 'embedding'),
        )

    def _split_generators(self, dl_manager):
        return [
            tfds.core.SplitGenerator(
                name='full_dataset',
                gen_kwargs={
                    "elevation_dir": self.elevation_dir,
                    "embedding_dir": self.embedding_dir,
                    "file_list": self.file_list,
                },
            ),
        ]

    def _generate_examples(self, elevation_dir, embedding_dir, file_list):
        for file_name in file_list:
            elevation_path = os.path.join(elevation_dir, file_name)
            embedding_path = os.path.join(
                embedding_dir, file_name.replace('.tif', '.npy'))

            with rasterio.open(elevation_path) as src:
                elevation = src.read(1).astype(np.float32)

            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path).astype(np.float32)
            else:
                embedding = np.load(self.empty_context_path).astype(np.float32)

            yield file_name, {
                'elevation': elevation,
                'embedding': embedding,
            }


def load_and_preprocess(example):
    elevation = example['elevation']
    embedding = example['embedding']
    return elevation, embedding


def get_preheated_dataset(batch_size, elevation_dir, context_dir, img_shape, emb_shape, included_files, empty_context_filename):

    builder = TextToTerrainDataset(elevation_dir=elevation_dir, embedding_dir=context_dir,
                              elevation_shape=img_shape, embedding_shape=emb_shape, file_list=included_files, empty_context_filename=empty_context_filename)
    builder.download_and_prepare()

    ds = builder.as_dataset(split='full_dataset')
    ds = ds.map(load_and_preprocess,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    ds = ds.with_options(options)

    return ds
