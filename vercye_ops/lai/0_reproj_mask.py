import click
from collections import defaultdict
from glob import glob

import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin


def find_union_extent_LAI_info(lai_dir: str, lai_region: str, lai_resolution: int, lai_file_ext: rio.coords.BoundingBox):
    lai_files = sorted(glob(f"{lai_dir}/{lai_region}_{lai_resolution}*.{lai_file_ext}"))

    if not lai_files:
        raise Exception(f"No LAI files found in {lai_dir} for region {lai_region} with resolution {lai_resolution} and extension {lai_file_ext}")

    union_bounds = None
    lai_crs = None
    resolution = None
    res_x = None
    res_y = None

    for lai_file in lai_files:
        with rio.open(lai_file) as ds:
            if union_bounds is None:
                union_bounds = ds.bounds
                lai_crs = ds.crs
                res_x, res_y = ds.res
            else:
                if ds.res != (res_x, res_y):
                    raise Exception(f"LAI files have different resolutions: {ds.res} vs {(res_x, res_y)}.")

                b = ds.bounds
                union_bounds = rio.coords.BoundingBox(
                    left=min(union_bounds.left, b.left),
                    bottom=min(union_bounds.bottom, b.bottom),
                    right=max(union_bounds.right, b.right),
                    top=max(union_bounds.top, b.top),
                )

    return union_bounds, lai_crs, (res_x, res_y)


def handle_identify_extent_reproject(lai_dir, lai_region, lai_resolution, lai_file_ext, mask_path, out_path):
    union_bounds, lai_crs, (res_x, res_y) = find_union_extent_LAI_info(lai_dir, lai_region, lai_resolution, lai_file_ext)

    # Calculate transform and dimensions
    width = int((union_bounds.right - union_bounds.left) / res_x)
    height = int((union_bounds.top - union_bounds.bottom) / abs(res_y))
    transform = from_origin(union_bounds.left, union_bounds.top, res_x, abs(res_y))

    with rio.open(mask_path) as mask_ds:
        mask_crs = mask_ds.crs
        mask_transform = mask_ds.transform

        with rio.open(
            out_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=rio.uint8,
            crs=lai_crs,
            transform=transform
        ) as dst_ds:
            reproject(
                source=rio.band(mask_ds, 1),
                destination=rio.band(dst_ds, 1),
                src_transform=mask_transform,
                src_crs=mask_crs,
                dst_transform=transform,
                dst_crs=lai_crs,
                resampling=Resampling.nearest
            )
    return

def handle_reprojection_to_specified_raster(lai_path, mask_path, out_path):
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


@click.command()
@click.argument('mask_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path())
@click.option('--lai_dir', type=click.Path(exists=True), default=None, help='Directory where LAI data is saved. Needs to be used with --lai_region.')
@click.option('--lai_region', type=str, default=None, help='Region of LAI data to use. Needs to be used with --lai_dir.')
@click.option('--lai_resolution', type=int, help='Resolution of LAI data to use. Needs to be used if providing --lai_dir')
@click.option('--lai_file_ext', type=click.Choice(['tif', 'vrt']), help='File extension of the LAI files. Usage with --lai_dir', default='tif')
@click.option('--lai_path', type=click.Path(exists=True), default=None, help='Path to a specific LAI file. Mutually exclusive with --lai_dir/region.')
def main(mask_path, out_path, lai_dir, lai_region, lai_resolution, lai_file_ext, lai_path):
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
        handle_identify_extent_reproject(lai_dir, lai_region, lai_resolution, lai_file_ext, mask_path, out_path)
    else:
        handle_reprojection_to_specified_raster(lai_path, mask_path, out_path)

    

if __name__ == "__main__":
    main()