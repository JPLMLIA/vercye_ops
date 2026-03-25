import click
import geopandas as gpd
import rasterio as rio
from pyproj import CRS
from rasterio.mask import mask


@click.command()
@click.argument("mask_path", type=click.Path(exists=True))
@click.argument("shp_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def main(mask_path, shp_path, out_path):
    """Crop and zero's out cropmask outside of shp geometry"""

    # Load the input shapefile/geojson and original mask
    shp = gpd.read_file(shp_path)

    with rio.open(mask_path) as src:
        expected_crs = CRS.from_epsg(4326)

        # Validate that both inputs are in EPSG:4326
        if src.crs is None or not CRS(src.crs).equals(expected_crs):
            raise ValueError(
                f"Cropmask raster CRS is {src.crs}, expected EPSG:4326. "
                "Reproject the cropmask before running this step."
            )
        if shp.crs is None or not CRS(shp.crs).equals(expected_crs):
            raise ValueError(
                f"Geometry CRS is {shp.crs}, expected EPSG:4326. " "Reproject the geometry before running this step."
            )

        # Zero out pixels outside the input geometry
        masked_src, masked_transform = mask(src, shp.geometry, crop=True, nodata=0, indexes=1)

        # Get metadata from the source file
        out_meta = src.meta.copy()

    # Update the metadata with new dimensions and transform
    out_meta.update(
        {
            "driver": "GTiff",
            "height": masked_src.shape[0],
            "width": masked_src.shape[1],
            "transform": masked_transform,
            "compress": "lzw",
        }
    )

    # Write the masked array to a new raster file
    with rio.open(out_path, "w", **out_meta) as dest:
        dest.write(masked_src, 1)


if __name__ == "__main__":
    main()
