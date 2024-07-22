# Terrain Consistency Models

## Dataset

The model is trained on samples from the [NASADEM Merged DEM Global 1 arc second](https://lpdaac.usgs.gov/products/nasadem_hgtv001/) digital elevation model dataset.

From the available 14520 `.tif` files in the merged NASADEM dataset, 2048 were randomly selected. The samples were further cut into `256x256` slices. The slices were filtered to exclude samples that contained no nonzero elevation. The remaining ~330K slices were sorted in decreasing order of entropy (calculated from a 256-bin histogram of the sample) and the first 100K samples were selected. Names of the `.tif` files that made it into the dataset can be found in `./src/data/ranked_dataset_samples.json`

### Example 
    The terrain of Tshuapa, Democratic Republic of the Congo, features undulating hills and subtle ridges that suggest varying elevations, interspersed with intricate patterns of erosion. The landscape is characterized by extensive networks of stream beds and plateaus, indicating a dynamic interplay of fluvial processes and geological formations.

![An example textured slice from the NASADEM dataset](assets/NASADEM_HGT_n00e023_slice_512_2560.png)

<!-- Recalculate!!

Since the original tiles covered 1 arc second (30x30m) on 3601x3601 pixels, the new tiles cover approximately 420x420m areas.

-->
