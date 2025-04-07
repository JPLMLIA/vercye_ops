import click
from collections import defaultdict
from glob import glob

import rasterio as rio
from rasterio.warp import reproject, Resampling


def find_largest_extent_LAI_file(lai_dir: str, lai_region: str, lai_resolution: int):
    lai_files = sorted(glob(f"{lai_dir}/{lai_region}_{lai_resolution}*.tif"))
    max_extent = None
    max_extent_file = None
    extent_frequencies = defaultdict(int)
    for lai_file in lai_files:
        with rio.open(lai_file) as lai_ds:
            lai_width = lai_ds.width
            lai_height = lai_ds.height

            if max_extent is None:
                max_extent = (lai_width, lai_height)
                max_extent_file = lai_file
            elif lai_width * lai_height > max_extent[0] * max_extent[1]:
                max_extent = (lai_width, lai_height)
                max_extent_file = lai_file

            extent_frequencies[(lai_width, lai_height)] += 1

    most_frequent_extent = max(extent_frequencies, key=extent_frequencies.get)
    if most_frequent_extent != max_extent:
        raise Exception(
            f"LAI file with the largest extent ({max_extent_file}) is not the most frequent one. "
            "Please manually identify which LAI file to use as a reference for resolution and extent."
        )

    return max_extent_file


@click.command()
@click.argument('mask_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path())
@click.option('--lai_dir', type=click.Path(exists=True), default=None, help='Directory where LAI data is saved. Needs to be used with --lai_region.')
@click.option('--lai_region', type=str, default=None, help='Region of LAI data to use. Needs to be used with --lai_dir.')
@click.option('--lai_resolution', type=int, help='Resolution of LAI data to use. Needs to be used if providing --lai_dir')
@click.option('--lai_path', type=click.Path(exists=True), default=None, help='Path to a specific LAI file. Mutually exclusive with --lai_dir/region.')
def main(mask_path, out_path, lai_dir, lai_region, lai_resolution, lai_path):
    """Reprojects a crop mask to the LAI raster
    
    mask_path is reprojected to match the projection, extent, and resolution of
    LAI_path. It is then saved as out_path.

    Since the extent of LAI files may vary, it is reccomended to use --lai_dir/region to identify the file with the largest extent.
    """

    if not (lai_dir and lai_region and lai_resolution) and not lai_path:
        raise Exception("Please either specify lai_dir and lai_region or specify lai_path.")

    if (lai_dir or lai_region) and lai_path:
        raise Exception("Please either specify lai_dir and lai_region OR lai_path.")

    if lai_dir and lai_region:
        lai_path = find_largest_extent_LAI_file(lai_dir, lai_region, lai_resolution)

    with rio.open(mask_path) as mask_ds:
        # Mask's CRS
        mask_crs = mask_ds.crs
        mask_transform = mask_ds.transform

        with rio.open(lai_path) as lai_ds:
            lai_crs = lai_ds.crs
            lai_transform = lai_ds.transform
            lai_width = lai_ds.width
            lai_height = lai_ds.height

            # Write reprojected
            with rio.open(
                out_path,
                'w',
                driver='GTiff',
                height=lai_height,
                width=lai_width,
                count=1,
                dtype=rio.uint8,
                crs=lai_crs,
                transform=lai_transform
            ) as dst_ds:
                reproject(
                    source=rio.band(mask_ds, 1),
                    destination=rio.band(dst_ds, 1),
                    src_transform=mask_transform,
                    src_crs=mask_crs,
                    dst_transform=lai_transform,
                    dst_crs=lai_crs,
                    resampling=Resampling.nearest
                )

if __name__ == "__main__":
    main()