import glob
import os
import re
from datetime import datetime

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.features import rasterize
from shapely.wkt import loads

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def read_met_file(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

    lat = None
    lon = None
    for line in lines:
        if line.lower().startswith("latitude"):
            lat = float(line.split("=")[1].strip().split()[0])
        elif line.lower().startswith("longitude"):
            lon = float(line.split("=")[1].strip().split()[0])

    header_line = next(i for i, line in enumerate(lines) if line.startswith("year"))
    data_lines = lines[header_line + 2 :]  # Skip header and unit lines

    data = []
    for line in data_lines:
        parts = line.split()
        data.append(
            {
                "date": datetime.strptime(f"{parts[0]}-{int(parts[1]):03}", "%Y-%j"),
                "year": int(parts[0]),
                "day": int(parts[1]),
                "radn": float(parts[2]) if float(parts[2]) != -999 else 0,
                "meant": float(parts[3]) if float(parts[3]) != -999 else 0,
                "maxt": float(parts[4]) if float(parts[3]) != -999 else 0,
                "mint": float(parts[5]) if float(parts[4]) != -999 else 0,
                "rain": float(parts[6]) if float(parts[5]) != -999 else 0,
                "wind": float(parts[7]) if float(parts[6]) != -999 else 0,
            }
        )

    df = pd.DataFrame(data)
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def plot_map(gdf, column, cmap, title, legend_label, pdf_pages):
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    gdf.boundary.plot(ax=ax, linewidth=1, edgecolor="black")
    plot = gdf.plot(column=column, ax=ax, cmap=cmap, alpha=0.6, edgecolor="k")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%")
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=gdf[column].min(), vmax=gdf[column].max())
    )
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(legend_label, fontsize=14)

    ax.set_title(title, fontsize=18, loc="left", pad=20)
    plt.xlabel("Longitude", fontsize=14)
    plt.ylabel("Latitude", fontsize=14)

    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.set_facecolor("white")

    pdf_pages.savefig(fig)
    plt.close(fig)


def aggregate_data(data):
    """Helper function to aggregate meteorological stats."""
    data.loc[:, "year"] = data["date"].dt.year
    annual_rainfall = (
        data.groupby(["latitude", "longitude", "year"])["rain"]
        .sum()
        .groupby(level=[0, 1])
        .mean()
        .reset_index()
    )

    aggregated_data = (
        data.groupby(["latitude", "longitude"])
        .agg(
            {
                "radn": "mean",
                "maxt": "mean",
                "mint": "mean",
                "meant": "mean",
                "wind": "mean",
            }
        )
        .reset_index()
    )

    # Merging the average annual rainfall with the aggregated data
    aggregated_data = pd.merge(
        aggregated_data, annual_rainfall, on=["latitude", "longitude"]
    )

    # Converting wind speed from m/s to km/h
    aggregated_data["wind"] = aggregated_data["wind"] * 3.6

    return aggregated_data


def aggregate_met_stats(regions_base_dir, year, num_last_years):
    """
    Aggregate meteorological stats from multiple regions within a directory.

    Parameters
    ----------
    regions_base_dir : str
        Path to the directory (e.g., year, timepoint) containing region subdirectories.

    year: int
        Specific year to consider for aggregation.

    num_last_years : int
        Number of last years to consider for additional aggregation for comparison.

    Returns
    -------
    gdf : GeoDataFrame
        A GeoDataFrame containing the aggregated meteorological stats.
    """
    file_paths = glob.glob(os.path.join(regions_base_dir, "*", "*.met"))
    dfs = [read_met_file(filepath) for filepath in file_paths]
    all_data = pd.concat(dfs, ignore_index=True)

    latest_date = all_data["date"].max()
    cutoff_date = latest_date - pd.DateOffset(years=num_last_years)

    last_n_years_data = all_data[all_data["date"] >= cutoff_date].copy()
    single_year_data = all_data[all_data["date"].dt.year == year].copy()

    aggregated_last_n_years = aggregate_data(last_n_years_data)
    aggregated_single_year = aggregate_data(single_year_data)

    # Get geometries of the regions
    all_geometry_files = glob.glob(os.path.join(regions_base_dir, "*", "*.geojson"))
    pattern = re.compile(r".*/([^/]+)/\1\.geojson$")  # Ensures the same wildcard value
    valid_gemoetry_files = [f for f in all_geometry_files if pattern.match(f)]

    # Load precomputed centroids to match with metfile
    geometries = []
    for geometry_file in valid_gemoetry_files:
        gdf = gpd.read_file(geometry_file)
        centroid_geom = loads(
            gdf["centroid"].iloc[0]
        )  # Extract the first (and only) centroid
        gdf["latitude"] = centroid_geom.y
        gdf["longitude"] = centroid_geom.x
        geometries.append(gdf)

    gdf_polygons = pd.concat(geometries, ignore_index=True)

    # Entries in metfile are rounded to 2 digits, so need to match
    gdf_polygons["latitude"] = gdf_polygons["latitude"].round(2)
    gdf_polygons["longitude"] = gdf_polygons["longitude"].round(2)

    gdf_last_n_years = gdf_polygons.merge(
        aggregated_last_n_years, on=["latitude", "longitude"]
    )
    gdf_single_year = gdf_polygons.merge(
        aggregated_single_year, on=["latitude", "longitude"]
    )

    return {"multiyear": gdf_last_n_years, "single_year": gdf_single_year}


def plot_stats_and_save(
    single_year_data, multi_year_data, year, num_last_years, out_file_path
):
    with PdfPages(out_file_path) as pdf_pages:
        plot_map(
            single_year_data,
            "rain",
            "Blues",
            f"Year {year}: Average Annual Rainfall (mm)",
            "Annual Rainfall (mm)",
            pdf_pages,
        )
        plot_map(
            single_year_data,
            "meant",
            "coolwarm",
            f"Year {year}: Average Mean Temperature (°C)",
            "Mean Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            single_year_data,
            "maxt",
            "OrRd",
            f"Year {year}: Average Max Temperature (°C)",
            "Max Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            single_year_data,
            "mint",
            "Blues",
            f"Year {year}: Average Min Temperature (°C)",
            "Min Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            single_year_data,
            "radn",
            "YlOrBr",
            f"Year {year}: Average Radiation (MJ/m^2)",
            "Radiation (MJ/m^2)",
            pdf_pages,
        )
        plot_map(
            single_year_data,
            "wind",
            "PuBu",
            f"Year {year}: Average Wind Speed (km/h)",
            "Wind Speed (km/h)",
            pdf_pages,
        )

        plot_map(
            multi_year_data,
            "rain",
            "Blues",
            f"Last {num_last_years} years: Average Annual Rainfall (mm)",
            "Annual Rainfall (mm)",
            pdf_pages,
        )
        plot_map(
            multi_year_data,
            "meant",
            "coolwarm",
            f"Last {num_last_years} years: Average Mean Temperature (°C)",
            "Mean Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            multi_year_data,
            "maxt",
            "OrRd",
            f"Last {num_last_years} years: Average Max Temperature (°C)",
            "Max Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            multi_year_data,
            "mint",
            "Blues",
            f"Last {num_last_years} years: Average Min Temperature (°C)",
            "Min Temperature (°C)",
            pdf_pages,
        )
        plot_map(
            multi_year_data,
            "radn",
            "YlOrBr",
            f"Last {num_last_years} years:Average Radiation (MJ/m^2)",
            "Radiation (MJ/m^2)",
            pdf_pages,
        )
        plot_map(
            multi_year_data,
            "wind",
            "PuBu",
            f"Last {num_last_years} years: Average Wind Speed (km/h)",
            "Wind Speed (km/h)",
            pdf_pages,
        )
        pdf_pages.close()


def get_reference_metadata(reference_tif):
    with rasterio.open(reference_tif) as src:
        return {
            "transform": src.transform,
            "crs": src.crs,
            "shape": (src.height, src.width),
        }


def polygon_to_raster(gdf, value_column, out_shape, transform):
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[value_column]))
    raster = rasterize(shapes, out_shape=out_shape, transform=transform, fill=np.nan)
    return raster


def create_multi_band_tif(
    single_year_data, multi_year_data, year, num_last_years, reference_tif, out_tif_path
):
    # Get metadata from reference TIFF
    ref_meta = get_reference_metadata(reference_tif)
    transform, crs, (height, width) = (
        ref_meta["transform"],
        ref_meta["crs"],
        ref_meta["shape"],
    )

    variables = ["rain", "meant", "maxt", "mint", "radn", "wind"]
    descriptions = [
        "Annual Rainfall",
        "Mean Temperature",
        "Max Temperature",
        "Min Temperature",
        "Radiation",
        "Wind Speed",
    ]

    layers = []
    layer_names = []

    # Process single-year data
    for var in variables:
        raster = polygon_to_raster(single_year_data, var, (height, width), transform)
        layers.append(raster)
        layer_names.append(f"Year_{year}_{var}")

    # Process multi-year data
    for var in variables:
        raster = polygon_to_raster(multi_year_data, var, (height, width), transform)
        layers.append(raster)
        layer_names.append(f"Last_{num_last_years}years_{var}")

    layers = np.stack(layers)
    save_as_tif(layers, layer_names, out_tif_path, transform, crs)


def save_as_tif(layers, layer_names, out_tif_path, transform, crs):
    bands, height, width = layers.shape

    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=rasterio.float32,
        transform=transform,
        crs=crs,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        for i, (layer, name) in enumerate(zip(layers, layer_names), start=1):
            dst.write(layer, i)
            dst.set_band_description(i, name)


@click.command()
@click.option(
    "--regions_base_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the directory (e.g., year, timepoint) containing region subdirectories.",
)
@click.option(
    "--output_pdf_path",
    required=True,
    type=click.Path(),
    help="Path to save the aggregated met data plots (.pdf).",
)
@click.option("--year", type=int, help="Specific year to consider for aggregation.")
@click.option(
    "--num_last_years",
    default=20,
    type=int,
    help="Number of last years to consider for additional aggregation for comparison.",
)
@click.option(
    "--reference_tif_path",
    required=False,
    type=click.Path(),
    default=None,
    help="Path to a tif file containing the target crs and resolution.",
)
@click.option(
    "--output_tif_path",
    required=False,
    type=click.Path(),
    default=None,
    help="Path to the output .tif file.",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(
    regions_base_dir,
    output_pdf_path,
    year,
    num_last_years,
    reference_tif_path,
    output_tif_path,
    verbose,
):
    """Aggregate meteorological stats from multiple regions within a directory."""
    if verbose:
        logger.setLevel("INFO")
    else:
        logger.setLevel("WARNING")

    if (reference_tif_path and not output_tif_path) or (
        output_tif_path and not reference_tif_path
    ):
        raise Exception(
            "'reference_tif_path' and 'output_tif_path' must be both set or not set at all."
        )

    logger.info(f"Processing directory: {regions_base_dir}")
    aggregated_stats = aggregate_met_stats(regions_base_dir, year, num_last_years)
    single_year_stats = aggregated_stats.get("single_year")
    multiyear_stats = aggregated_stats.get("multiyear")

    if not aggregated_stats:
        raise Exception("Failed to aggregate met stats.")

    logger.info(f"Saving aggregated met stats to: {output_pdf_path}")
    plot_stats_and_save(
        single_year_stats, multiyear_stats, year, num_last_years, output_pdf_path
    )

    if output_tif_path and reference_tif_path:
        logger.info(f"Saving additional tif output to {output_tif_path}.")
        create_multi_band_tif(
            single_year_stats,
            multiyear_stats,
            year,
            num_last_years,
            reference_tif_path,
            output_tif_path,
        )


if __name__ == "__main__":
    cli()
