import os
import os.path as op
import logging
import click
import numpy as np
import geopandas as gpd
import rasterio as rio
from rasterio.merge import merge
import pandas as pd

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

def get_file_path(base_dir, region_name, file_suffix):
    """
    Generate a file path for a specific region and suffix.

    Parameters
    ----------
    base_dir : str
        Base directory containing region subdirectories.
    region_name : str
        Name of the region.
    file_suffix : str
        File suffix, e.g., '_LAI_MAX.tif'.

    Returns
    -------
    str
        Full path to the file.
    """
    return op.join(base_dir, region_name, f'{region_name}{file_suffix}')

def validate_consistency_across_products(file_groups):
    """
    Validate that CRS and resolution are consistent across different product groups.

    Parameters
    ----------
    file_groups : dict
        Dictionary where keys are product labels (e.g., 'yield', 'LAI', 'cropmask') and
        values are lists of file paths for each product.

    Raises
    ------
    ValueError
        If CRS or resolution is inconsistent across products or files.
    """
    logger.info('Validating consistency across products...')

    reference_crs = None
    reference_res = None

    for label, file_paths in file_groups.items():
        logger.info(f'Validating {label} TIF files...')

        with rio.open(file_paths[0]) as first_ds:
            product_crs = first_ds.crs
            product_res = first_ds.transform[0]

            if reference_crs is None:
                reference_crs = product_crs
                reference_res = product_res
            else:
                if product_crs != reference_crs:
                    raise ValueError(f'CRS mismatch detected in {label} TIF files: {product_crs} vs {reference_crs}')
                if product_res != reference_res:
                    raise ValueError(f'Resolution mismatch detected in {label} TIF files: {product_res} vs {reference_res}')

        for file_path in file_paths:
            with rio.open(file_path) as ds:
                if ds.crs != product_crs:
                    raise ValueError(f'CRS mismatch within {label} TIF files: {file_path}')
                if ds.transform[0] != product_res:
                    raise ValueError(f'Resolution mismatch within {label} TIF files: {file_path}')

    logger.info("All products validated successfully.")

def merge_tifs(tif_files, label):
    """
    Merge multiple TIF files into a single array and profile.

    Parameters
    ----------
    tif_files : list of str
        List of file paths to TIF files.

    Returns
    -------
    tuple
        Merged array and updated profile.
    """
    logger.info(f'Merging {label} TIF files...')
    datasets = [rio.open(tif) for tif in tif_files]

    if len(datasets) == 0:
        raise ValueError(f'No datasets to merge for {label}')

    merged_array, merged_transform = merge(datasets)

    band_names = [f'{label}_{datasets[0].descriptions[i]}' if datasets[0].descriptions[i] else f'{label}_Band_{i+1}' for i in range(datasets[0].count)]
    profile = datasets[0].profile.copy()
    profile.update(
        driver='GTiff',
        height=merged_array.shape[1],
        width=merged_array.shape[2],
        transform=merged_transform,
        count=0,  # Keep as 0 for now
        compress='lzw'
    )

    for ds in datasets:
        ds.close()

    logger.info(f'Merged {label} TIF files successfully.')
    logger.info(f'Merged array shape: {merged_array.shape}')

    return {
        'array': merged_array,
        'band_names': band_names, 
        'profile': profile
    }

def save_aggregated_map(output_path, map, band_names, profile):
    """
    Save the merged arrays as a multi-band GeoTIFF.

    Parameters
    ----------
    output_path : str
        Path to save the output file.
    arrays : list of ndarray
        List of arrays to save as bands.
    profile : dict
        Raster profile.
    """
    profile.update(count=map.shape[0])

    with rio.open(output_path, 'w', **profile) as dst:
        for i in range(map.shape[0]):
            band_idx = i + 1
            dst.write(map[i, :, :], band_idx)  # Write each band individually
            dst.set_band_description(band_idx, band_names[i])

    logger.info(f'Aggregated map saved to: {output_path}')


def merge_shapefiles(shapefile_paths):
    """
    Merge multiple shapefiles into a single GeoDataFrame.

    Parameters
    ----------
    shapefile_paths : list of str
        List of file paths to shapefiles.

    Returns
    -------
    GeoDataFrame
        Merged GeoDataFrame.
    """
    logger.info('Merging shapefiles...')
    gdfs = [gpd.read_file(shp) for shp in shapefile_paths]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    return merged_gdf

@click.command()
@click.option('--roi_base_dir', required=True, type=click.Path(exists=True), help='Path to the directory containing region subdirectories.')
@click.option('--output_lai_tif_fpath', required=False, type=click.Path(), help='Path to save the aggregated lai GeoTIFF.')
@click.option('--output_yield_tif_fpath', required=False, type=click.Path(), help='Path to save the aggregated simulated yield GeoTIFF.')
@click.option('--output_cropmask_tif_fpath', required=False, type=click.Path(), help='Path to save the aggregated cropmask GeoTIFF.')
@click.option('--output_shapefile_fpath', required=False, type=click.Path(), help='Path to save the aggregated GeoJsons.')
def cli(roi_base_dir, output_lai_tif_fpath=None, output_yield_tif_fpath=None, output_cropmask_tif_fpath=None, output_shapefile_fpath=None):
    """
    Command-line interface to aggregate maps from regions.

    Parameters
    ----------
    roi_base_dir : str
        Directory containing subdirectories for each region.
    output_fpath : str
        Path to save the aggregated GeoTIFF.
    """
    logger.setLevel(logging.INFO)
    logger.info(f'Starting map aggregation for regions in {roi_base_dir}...')

    output_fpaths = {
        'yield': output_yield_tif_fpath or op.join(roi_base_dir, f'aggregated_yield.tif'),
        'LAI': output_lai_tif_fpath or op.join(roi_base_dir, f'aggregated_LAI_MAX.tif'),
        'cropmask': output_cropmask_tif_fpath or op.join(roi_base_dir, f'aggregated_cropmask.tif'),
        'shapefile': output_shapefile_fpath or op.join(roi_base_dir, 'aggregated.geojson')
    }

    region_dirs = [d for d in os.listdir(roi_base_dir) if op.isdir(op.join(roi_base_dir, d))]

    yield_files = [get_file_path(roi_base_dir, region, '_yield_map.tif') for region in region_dirs]
    lai_files = [get_file_path(roi_base_dir, region, '_LAI_MAX.tif') for region in region_dirs]
    cropmask_files = [get_file_path(roi_base_dir, region, '_cropmask_constrained.tif') for region in region_dirs]

    file_groups = {
        'yield': yield_files,
        'LAI': lai_files,
        'cropmask': cropmask_files
    }

    # Validate CRS and resolution consistency across products
    # Currently checking that all products have same CRS and Res but can also loosen this constraint if required
    validate_consistency_across_products(file_groups)

    # Merge TIF files and save aggregated map
    logger.info('Merging TIF files...')

    # for label, file_group in file_groups.items():
    #     result = merge_tifs(file_group, label)
    #     save_aggregated_map(output_fpaths[label], result['array'], result['band_names'], result['profile'])
    
    # Merge shapefiles and save
    merged_gdf = merge_shapefiles([get_file_path(roi_base_dir, region, '.geojson') for region in region_dirs])
    merged_gdf.to_file(output_fpaths['shapefile'], driver='GeoJSON')

if __name__ == '__main__':
    cli()
