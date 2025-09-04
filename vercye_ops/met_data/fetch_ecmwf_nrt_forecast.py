from pathlib import Path

import click
import ee
import pandas as pd

from vercye_ops.met_data.fetch_era5 import (
    build_ee_geometry,
    clean_era5,
    error_checking_function,
    postprocess_for_apsim,
    write_met_data_to_csv,
)
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def fetch_nrt_forecast(forecasting_date, lon, lat, polygon_path):
    year = forecasting_date.year
    month = forecasting_date.month
    day = forecasting_date.day
    logger.info(f"Fetching forecast from {year}, {month}, {day}")

    geometry, geo_geotype = build_ee_geometry(lon, lat, polygon_path)

    era5 = (
        ee.ImageCollection("ECMWF/NRT_FORECAST/IFS/OPER")
        .filter(ee.filter.Filter.eq("creation_year", year))
        .filter(ee.filter.Filter.eq("creation_month", month))
        .filter(ee.filter.Filter.eq("creation_day", day))
        .filterBounds(geometry)
        .select(
            [
                "total_precipitation_sfc",
                "temperature_2m_sfc",
                "surface_solar_radiation_downwards_sfc",
                "u_component_of_wind_10m_sfc",
                "v_component_of_wind_10m_sfc",
            ]
        )
        .sort("forecast_hours")
    )

    def aggregate_daily(imgcol):
        # min/max temps
        temp_min = imgcol.select("temperature_2m_sfc").reduce(ee.Reducer.min())
        temp_max = imgcol.select("temperature_2m_sfc").reduce(ee.Reducer.max())

        # wind means
        u_mean = imgcol.select("u_component_of_wind_10m_sfc").mean()
        v_mean = imgcol.select("v_component_of_wind_10m_sfc").mean()

        # cumulative -> daily diff needed
        precip = imgcol.select("total_precipitation_sfc")
        precip_diff = precip.max().subtract(precip.min())

        solar = imgcol.select("surface_solar_radiation_downwards_sfc")
        solar_diff = solar.max().subtract(solar.min())

        return (
            temp_min.rename("temperature_2m_min")
            .addBands(temp_max.rename("temperature_2m_max"))
            .addBands(u_mean.rename("u_component_of_wind_10m"))
            .addBands(v_mean.rename("v_component_of_wind_10m"))
            .addBands(precip_diff.rename("total_precipitation_sum"))
            .addBands(solar_diff.rename("surface_solar_radiation_downwards_sum"))
        )

    # make sequence in steps of 24 for daily
    total_hours = era5.aggregate_max("forecast_hours").getInfo()
    hour_steps = ee.List.sequence(0, total_hours - 1, 24)

    # build daily aggregates in required format
    def daily_chunk(offset):
        offset = ee.Number(offset)  # ensure it's an ee.Number
        img = aggregate_daily(era5.filter(ee.Filter.rangeContains("forecast_hours", offset, offset.add(23))))
        return img.set("forecast_offset", offset)

    daily_imgs = hour_steps.map(daily_chunk)
    daily_ic = ee.ImageCollection(daily_imgs)

    def extract(image):
        # compute valid date = forecasting_date + forecast_offset (in hours)
        offset_hours = ee.Number(image.get("forecast_offset"))
        valid_date = ee.Date(forecasting_date).advance(offset_hours, "hour")
        date_str = valid_date.format("YYYY-MM-dd")

        reducer = ee.Reducer.first() if geo_geotype == "point" else ee.Reducer.mean()
        values = image.reduceRegion(reducer, geometry, 1000)
        return ee.Feature(None, values.set("date", date_str))

    features = daily_ic.map(extract)
    feature_collection = ee.FeatureCollection(features)
    result = feature_collection.getInfo()
    records = [f["properties"] for f in result["features"]]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["forecast"] = True

    return df


@click.command()
@click.option(
    "--forecast_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for data collection in YYYY-MM-DD format.",
)
@click.option(
    "--forecast_date_from_existing",
    is_flag=True,
    help="Instead of using the fixed forecast_date, derive the forecast date as the last date in an existing output_file.",
)
@click.option(
    "--lon",
    type=float,
    required=False,
    help="Longitude of the location. Must be provided if --polygon_path is not set and met_agg_method is centroid.",
)
@click.option(
    "--lat",
    type=float,
    required=False,
    help="Latitude of the location. Must be provided if --polygon_path is not set and met_agg_method is centroid.",
)
@click.option(
    "--polygon_path",
    type=click.Path(file_okay=True, dir_okay=False),
    required=False,
    help="Path to the regions polygon if using mean aggregation method.",
    default=None,
)
@click.option(
    "--met_agg_method",
    type=click.Choice(["mean", "centroid"], case_sensitive=False),
    help="Method to aggregate meteorological data in a ROI.",
)
@click.option(
    "--ee_project",
    type=str,
    required=False,
    help="Name of the Earth Engine Project in which to run the ERA5 processing. Only required when using --met_source era5.",
)
@click.option(
    "--output_file",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
    help="File where the forecasts should be saved. Will append by default if existing.",
)
@click.option(
    "--overwrite", is_flag=True, help="If the output_file alraedy exists, data will not be appended but overwritten."
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def cli(
    forecast_date,
    forecast_date_from_existing,
    lon,
    lat,
    polygon_path,
    met_agg_method,
    ee_project,
    output_file,
    overwrite,
    verbose,
):
    """Fetches / appends ECMWF near real time meteorological forecasts in the required APSIM format."""
    if verbose:
        logger.setLevel("INFO")

    if ee_project is None:
        raise Exception("Setting --ee_project required when using ERA5 as the meteorological data source.")

    if (not forecast_date and not forecast_date_from_existing) or (forecast_date_from_existing and forecast_date):
        raise Exception("Must EITHER set forecast_date or forecast_date_from_existing.")

    if forecast_date_from_existing and not Path(output_file).exists():
        raise Exception(
            f"Forecast_date_from_existing chosen, but no file to derive the last date exists under {output_file}."
        )

    logger.info("Initializing google earth engine.")
    ee.Initialize(project=ee_project)

    # Identify date from which to do the forecast
    df_existing = None
    if Path(output_file).exists() and not overwrite:
        # Choose last valid date to do the forecast from
        df_existing = pd.read_csv(output_file, index_col=0, parse_dates=True)
        forecast_date = df_existing.index.max()

    if met_agg_method == "mean":
        if not polygon_path:
            raise ValueError("Must provided polygon_path when using mean aggregation.")
        df = fetch_nrt_forecast(forecasting_date=forecast_date, lon=None, lat=None, polygon_path=polygon_path)
    elif met_agg_method == "centroid":
        if not lon or not lat:
            raise ValueError("Must provide lat and lon when using centroid aggregation.")
        df = fetch_nrt_forecast(forecasting_date=forecast_date, lon=lon, lat=lat, polygon_path=None)
    else:
        raise Exception("Invalid agg method provided")

    df = postprocess_for_apsim(df, end_date=None, is_forecast=True)

    # Load existing and append
    if df_existing is not None:
        df = df[df.index > forecast_date]
        df = pd.concat([df_existing, df])

    # Clean the data
    df = clean_era5(df)
    error_checking_function(df)

    # Save to csv
    write_met_data_to_csv(df, output_file)
    logger.info("Data successfully written to %s", output_file)


if __name__ == "__main__":
    cli()
