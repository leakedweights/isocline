# Terrain Consistency Models

## Dataset

The model is trained on samples from the [NASADEM Merged DEM Global 1 arc second](https://lpdaac.usgs.gov/products/nasadem_hgtv001/) digital elevation model dataset.

From the available 14520 `.tif` files in the merged NASADEM dataset, 2048 were randomly selected. The samples were further cut into `256x256` slices. The slices were filtered to exclude samples that contained no nonzero elevation. The remaining ~330K slices were sorted in decreasing order of Shannon entropy and the first 100K samples were selected. Since the original tiles covered 1 arc second (30x30m) on 3601x3601 pixels, the new tiles cover approximately 420x420m areas.
