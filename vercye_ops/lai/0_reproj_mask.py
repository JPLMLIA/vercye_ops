import click

import rasterio as rio
from rasterio.warp import reproject, Resampling


@click.command()
@click.argument('mask_path', type=click.Path(exists=True))
@click.argument('lai_path', type=click.Path(exists=True))
@click.argument('out_path', type=click.Path())
def main(mask_path, lai_path, out_path):
    """Reprojects a crop mask to the LAI raster
    
    mask_path is reprojected to match the projection, extent, and resolution of
    LAI_path. It is then saved as out_path.
    """

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