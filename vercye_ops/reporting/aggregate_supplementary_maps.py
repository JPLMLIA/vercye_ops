"""
Aggregate supplementary maps (LAI, cropmask) and shapefiles from per-region outputs.

Yield map merging is handled separately by mosaic_and_reproject.py.
"""

import logging
import os
import os.path as op

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.merge import merge

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def get_file_path(base_dir, region_name, file_suffix):
    return op.join(base_dir, region_name, f"{region_name}{file_suffix}")


def validate_consistency_across_products(file_groups):
    """Validate that CRS and resolution are consistent across product groups."""
    logger.info("Validating consistency across products...")

    reference_crs = None
    reference_res = None

    for label, file_paths in file_groups.items():
        logger.info(f"Validating {label} TIF files...")

        with rio.open(file_paths[0]) as first_ds:
            product_crs = first_ds.crs
            product_res = first_ds.transform[0]

            if reference_crs is None:
                reference_crs = product_crs
                reference_res = product_res
            else:
                if product_crs != reference_crs:
                    raise ValueError(f"CRS mismatch detected in {label} TIF files: {product_crs} vs {reference_crs}")
                if product_res != reference_res:
                    raise ValueError(
                        f"Resolution mismatch detected in {label} TIF files: {product_res} vs {reference_res}"
                    )

        for file_path in file_paths:
            with rio.open(file_path) as ds:
                if ds.crs != product_crs:
                    raise ValueError(f"CRS mismatch within {label} TIF files: {file_path}")
                if ds.transform[0] != product_res:
                    raise ValueError(f"Resolution mismatch within {label} TIF files: {file_path}")

    logger.info("All products validated successfully.")


def merge_tifs(tif_files, label):
    """Merge multiple TIF files into a single array and profile."""
    logger.info(f"Merging {label} TIF files...")
    datasets = [rio.open(tif) for tif in tif_files]

    if len(datasets) == 0:
        raise ValueError(f"No datasets to merge for {label}")

    src_nodata = datasets[0].nodata
    if src_nodata is None and np.issubdtype(datasets[0].dtypes[0], np.floating):
        src_nodata = float("nan")

    merged_array, merged_transform = merge(datasets, nodata=src_nodata)

    band_names = [
        (f"{label}_{datasets[0].descriptions[i]}" if datasets[0].descriptions[i] else f"{label}_Band_{i+1}")
        for i in range(datasets[0].count)
    ]
    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        transform=merged_transform,
        count=0,
        compress="lzw",
        nodata=src_nodata,
    )

    for ds in datasets:
        ds.close()

    return {"array": merged_array, "band_names": band_names, "profile": profile}


def save_aggregated_map(output_path, map_array, band_names, profile):
    """Save merged arrays as a multi-band GeoTIFF."""
    profile.update(count=map_array.shape[0])

    with rio.open(output_path, "w", **profile) as dst:
        for i in range(map_array.shape[0]):
            band_idx = i + 1
            dst.write(map_array[i, :, :], band_idx)
            dst.set_band_description(band_idx, band_names[i])

    logger.info(f"Aggregated map saved to: {output_path}")


def merge_shapefiles(shapefile_paths, region_names):
    """Merge multiple shapefiles into a single GeoDataFrame."""
    logger.info("Merging shapefiles...")
    gdfs = [gpd.read_file(shp) for shp in shapefile_paths]

    if "cleaned_region_name_vercye" not in gdfs[0].columns:
        for gdf, region_name in zip(gdfs, region_names):
            gdf["cleaned_region_name_vercye"] = region_name

    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return merged_gdf


def is_valid_region_dir(base_dir, region_name):
    if not op.isdir(op.join(base_dir, region_name)):
        return False

    for suffix in ["_LAI_MAX.tif", "_cropmask_constrained.tif", ".geojson"]:
        if not op.exists(get_file_path(base_dir, region_name, suffix)):
            return False

    return True


@click.command()
@click.option(
    "--roi_base_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the directory containing region subdirectories.",
)
@click.option(
    "--yield_estimates_fpath",
    required=True,
    type=click.Path(),
    help="Path to the CSV containing the yield estimates per region.",
)
@click.option(
    "--val_fpath",
    required=False,
    type=click.Path(),
    help="Path to the csv containing the validation data per region.",
    default=None,
)
@click.option(
    "--output_lai_tif_fpath",
    required=False,
    type=click.Path(),
    help="Path to save the aggregated LAI GeoTIFF.",
)
@click.option(
    "--output_cropmask_tif_fpath",
    required=False,
    type=click.Path(),
    help="Path to save the aggregated cropmask GeoTIFF.",
)
@click.option(
    "--output_shapefile_fpath",
    required=False,
    type=click.Path(),
    help="Path to save the aggregated GeoJSONs.",
)
def cli(
    roi_base_dir,
    yield_estimates_fpath,
    val_fpath,
    output_lai_tif_fpath=None,
    output_cropmask_tif_fpath=None,
    output_shapefile_fpath=None,
):
    """Aggregate supplementary maps (LAI, cropmask) and shapefiles from per-region outputs."""
    logger.setLevel(logging.INFO)
    logger.info(f"Starting supplementary map aggregation for regions in {roi_base_dir}...")

    output_fpaths = {
        "LAI": output_lai_tif_fpath or op.join(roi_base_dir, "aggregated_LAI_MAX.tif"),
        "cropmask": output_cropmask_tif_fpath or op.join(roi_base_dir, "aggregated_cropmask.tif"),
        "shapefile": output_shapefile_fpath or op.join(roi_base_dir, "aggregated_region_boundaries.geojson"),
    }

    regions = [d for d in os.listdir(roi_base_dir) if is_valid_region_dir(roi_base_dir, d)]

    lai_files = [get_file_path(roi_base_dir, region, "_LAI_MAX.tif") for region in regions]
    cropmask_files = [get_file_path(roi_base_dir, region, "_cropmask_constrained.tif") for region in regions]

    file_groups = {"LAI": lai_files, "cropmask": cropmask_files}

    validate_consistency_across_products(file_groups)

    # Merge LAI and cropmask TIFs
    for label, file_group in file_groups.items():
        result = merge_tifs(file_group, label)
        save_aggregated_map(output_fpaths[label], result["array"], result["band_names"], result["profile"])

    # Merge shapefiles and join yield estimates
    merged_gdf = merge_shapefiles([get_file_path(roi_base_dir, region, ".geojson") for region in regions], regions)

    yield_estimates = pd.read_csv(yield_estimates_fpath)
    yield_estimates["region"] = yield_estimates["region"].astype(str)

    yield_estimates.rename(
        columns={
            "mean_yield_kg_ha": "estimated_mean_yield_kg_ha",
            "total_production_kg": "estimated_production_kg",
            "median_yield_kg_ha": "estimated_median_yield_kg_ha",
        },
        inplace=True,
    )

    merge_cols = ["region"]
    for col in [
        "estimated_mean_yield_kg_ha",
        "estimated_median_yield_kg_ha",
        "estimated_production_kg",
        "total_cropland_area_ha",
    ]:
        if col in yield_estimates.columns:
            merge_cols.append(col)

    merged_gdf = merged_gdf.merge(
        yield_estimates[merge_cols],
        left_on="cleaned_region_name_vercye",
        right_on="region",
        how="left",
    )

    if val_fpath:
        val_data = pd.read_csv(val_fpath)
        val_data["region"] = val_data["region"].astype(str)
        merged_gdf = merged_gdf.merge(val_data, left_on="cleaned_region_name_vercye", right_on="region", how="left")

    merged_gdf.to_file(output_fpaths["shapefile"], driver="GeoJSON")

    logger.info("Supplementary map aggregation complete.")


if __name__ == "__main__":
    cli()
