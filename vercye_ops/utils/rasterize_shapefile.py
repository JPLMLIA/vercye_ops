#!/usr/bin/env python3
import click
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd


@click.command()
@click.option("--shapefile", required=True, help="Input shapefile (GeoJSON/ .shp)")
@click.option("--reference", required=True, help="Reference GeoTIFF to match.")
@click.option("--out", "out_raster", required=True, help="Output raster path.")
def rasterize_shapefile(shapefile, reference, out_raster):
    """Rasterize a GeoJSON onto the grid of a reference GeoTIFF as a binary mask (1/0)"""
    # Read reference raster metadata
    with rasterio.open(reference) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        crs = ref.crs
        height, width = ref.height, ref.width

    # Read and prepare vector data
    gdf = gpd.read_file(shapefile)
    if gdf.empty:
        raise click.ClickException("GeoJSON contains no features.")

    if gdf.crs is None:
        raise click.ClickException("GeoJSON has no CRS. Assign or fix before use.")

    if crs is None:
        raise click.ClickException("Reference raster has no CRS.")

    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    gdf = gdf[gdf.geometry.notnull() & gdf.is_valid]
    if gdf.empty:
        raise click.ClickException("All geometries are empty or invalid.")
    else:
        shapes = [(geom, 1) for geom in gdf.geometry]

    # Choose dtype and nodata
    out_dtype = np.uint8
    nodata = 0
    out = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype=out_dtype,
    )

    # Write output
    profile.update(

        dtype=out_dtype,
        count=1,
        nodata=nodata,
        compress="lzw",
        tiled=True,
        driver="GTiff",
        blockxsize=min(512, width),
        blockysize=min(512, height),
    )

    with rasterio.open(out_raster, "w", **profile) as dst:
        dst.write(out, 1)

    click.echo(f"Saved rasterized output to {out_raster}")


if __name__ == "__main__":
    rasterize_shapefile()
