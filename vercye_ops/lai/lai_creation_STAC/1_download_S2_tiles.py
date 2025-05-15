import multiprocessing
import os
import xml.etree.ElementTree as ET
from functools import partial

import click
import geopandas as gpd
import numpy as np
import rasterio as rio
from vercye_ops.lai.lai_creation_STAC.stac_downloader import run_pipeline


def get_s2_geometry_data(metadata_xml):
    """
    Get sentinel-2 geometry-related data from granule metadata xml.
    This is rather hardcoded for currenlty planned applications.
    """

    # Parse XML and extract azimuth and zenith angles
    xml_root = ET.fromstring(metadata_xml)
    azimuth_angle_el = xml_root.findall(".//Mean_Sun_Angle/AZIMUTH_ANGLE")[0]
    azimuth_angle_units = azimuth_angle_el.attrib["unit"]
    if azimuth_angle_units != "deg":
        raise Exception(f"azimuth_angle_units must be 'deg', but it is {azimuth_angle_units}.")
    azimuth_angle = float(azimuth_angle_el.text)

    zenith_angle_el = xml_root.findall(".//Mean_Sun_Angle/ZENITH_ANGLE")[0]
    zenith_angle_units = zenith_angle_el.attrib["unit"]
    if zenith_angle_units != "deg":
        raise Exception(f"zenith_angle_units must be 'deg', but it is {zenith_angle_units}.")
    zenith_angle = float(zenith_angle_el.text)

    # Extract mean viewing incidence angles for band 8A
    b8a_incidence_angle_el = xml_root.findall(
        ".//Mean_Viewing_Incidence_Angle_List/Mean_Viewing_Incidence_Angle[@bandId='8']"
    )

    if not b8a_incidence_angle_el:
        raise Exception("Could not find Mean_Viewing_Incidence_Angle for band 8.")

    b8a_incidence_angle_el = b8a_incidence_angle_el[0]

    mean_incidence_azimuth_angle_b8a_el = b8a_incidence_angle_el.find("AZIMUTH_ANGLE")
    if mean_incidence_azimuth_angle_b8a_el.attrib["unit"] != "deg":
        raise Exception(f"mean_incidence_azimuth_angle_b8a must be in degrees.")
    mean_incidence_azimuth_angle_b8a = float(mean_incidence_azimuth_angle_b8a_el.text)

    mean_incidence_zenith_angle_b8a_el = b8a_incidence_angle_el.find("ZENITH_ANGLE")
    if mean_incidence_zenith_angle_b8a_el.attrib["unit"] != "deg":
        raise Exception(f"mean_incidence_zenith_angle_b8a must be in degrees.")
    mean_incidence_zenith_angle_b8a = float(mean_incidence_zenith_angle_b8a_el.text)

    return {
        "azimuth_angle": azimuth_angle,
        "zenith_angle": zenith_angle,
        "mean_incidence_azimuth_angle_b8a": mean_incidence_azimuth_angle_b8a,
        "mean_incidence_zenith_angle_b8a": mean_incidence_zenith_angle_b8a,
    }


def compute_cos_angles(
    zenith_angle, azimuth_angle, mean_incidence_zenith_angle_b8a, mean_incidence_azimuth_angle_b8a
):
    cos_vza = np.uint16(np.cos(np.deg2rad(mean_incidence_zenith_angle_b8a)) * 10000)
    cos_sza = np.uint16(np.cos(np.deg2rad(zenith_angle)) * 10000)
    # Converting to int16 to match GEE script
    cos_raa = np.int16(np.cos(np.deg2rad(azimuth_angle - mean_incidence_azimuth_angle_b8a)) * 10000)
    return {
        "cos_vza": cos_vza,
        "cos_sza": cos_sza,
        "cos_raa": cos_raa,
    }


def create_geometry_bands(item, cos_angles, metadata, output_folder, blocksize=256):
    geometry_band_paths = {}

    # Create each geometry band
    for angle_name, angle_value in cos_angles.items():
        geo_dtype = np.int16
        # Create empty array with same dimensions as other bands
        band_data = np.full((metadata["height"], metadata["width"]), angle_value, dtype=geo_dtype)

        # Save the geometry band
        output_path = os.path.join(
            output_folder, f"{item.id}_{angle_name}_{metadata['resolution']}m.tif"
        )

        with rio.open(
            output_path,
            "w",
            driver="GTiff",
            height=metadata["height"],
            width=metadata["width"],
            count=1,
            crs=metadata["crs"],
            transform=metadata["transform"],
            nodata=metadata["nodata"],
            compress="LZW",
            dtype=geo_dtype,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
        ) as dst:
            dst.write(band_data, 1)

        geometry_band_paths[angle_name] = output_path
        print(f"Created geometry band {angle_name}")

    return geometry_band_paths


def add_geometry_bands(
    item, band_paths, band_names, resolution, mask, metadata_file_paths, output_folder
):
    print("Computing geometry bands")
    graunle_metadata_file = metadata_file_paths["granule_metadata"]

    # Read the metadata XML file
    metadata_xml = None
    with open(graunle_metadata_file, "r") as f:
        metadata_xml = f.read()

    geometry_data = get_s2_geometry_data(metadata_xml)
    cos_angles = compute_cos_angles(
        geometry_data["zenith_angle"],
        geometry_data["azimuth_angle"],
        geometry_data["mean_incidence_zenith_angle_b8a"],
        geometry_data["mean_incidence_azimuth_angle_b8a"],
    )

    # Get the reference metadata from the first band
    # Expects all bands to have the same dimensions
    first_band_path = band_paths[band_names[0]]
    with rio.open(first_band_path) as src:
        reference_metadata = {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "resolution": resolution,
        }

    # Create the geometry bands using the reference metadata
    geometry_band_paths = create_geometry_bands(item, cos_angles, reference_metadata, output_folder)

    # Ensure correct bandorder for output
    band_paths["cos_vza"] = geometry_band_paths["cos_vza"]
    band_paths["cos_sza"] = geometry_band_paths["cos_sza"]
    band_paths["cos_raa"] = geometry_band_paths["cos_raa"]
    band_names = ["cos_vza", "cos_sza", "cos_raa"] + band_names

    return band_paths, band_names


def s2_mask_processor(
    maskbands, resolution, item, output_folder, scl_keep_classes, cloud_thresh, snowprob_thresh
):
    mask = None

    scl_band_meta, scl_band = maskbands["scl"]
    s2cloudless_band_meta, s2cloudless_band = maskbands["cloud"]
    snowprob_band_meta, snowprob_band = maskbands["snow"]
    mask = np.ones_like(scl_band)  # Start with all valid (1)

    # Invalidate pixels based on SCL
    mask = np.where(np.isin(scl_band, scl_keep_classes), mask, 0)

    # Invalidate pixels based on S2Cloudless
    mask = np.where(s2cloudless_band >= cloud_thresh, 0, mask)

    # Invalidate pixels based on Snowprob
    mask = np.where(snowprob_band >= snowprob_thresh, 0, mask)

    new_metadata = {}

    return new_metadata, mask


def build_s2_mask_processor(
    cloud_thresh,
    snowprob_thresh,
    scl_keep_classes=[4, 5],
):
    # Factory function that returns a Sentinel-2 mask processing function.
    # Parameters like scl_keep_classes, cloud_thresh, and snowprob_thresh are fixed at creation time.
    # SCL default classes to keep are [4, 5] (vegetation and non-vegetation)

    return partial(
        s2_mask_processor,
        scl_keep_classes=scl_keep_classes,
        cloud_thresh=cloud_thresh,
        snowprob_thresh=snowprob_thresh,
    )


@click.command()
@click.option(
    "--satellite",
    type=click.Choice(["S2"], case_sensitive=False),
    default="S2",
    help="Satellite source to use. Currently supported 'S2' (Sentinel-2)",
)
@click.option("--start-date", type=str, required=True, help="Start date in YYYY-MM-DD format")
@click.option("--end-date", type=str, required=True, help="End date in YYYY-MM-DD format")
@click.option("--resolution", type=int, help="Target resolution in meters.")
@click.option(
    "--geojson-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the GeoJSON/shapefile defining the area of interest",
)
@click.option(
    "--output-folder",
    type=click.Path(),
    required=True,
    help="Folder where output files will be saved",
)
@click.option(
    "--max-cloud-cover",
    type=int,
    default=70,
    help="Maximum cloud cover percentage (0-100) of a tile",
)
@click.option(
    "--cloudprob-thresh",
    type=int,
    default=25,
    help="Threshold for pixels to be considered clouds (S2Cloudless).",
)
@click.option(
    "--snowprob-thresh",
    type=int,
    default=15,
    help="Threshold for pixel to be considered as snow.",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of parallel workers (defaults to CPU count - 1)",
)
@click.option(
    "--save-mask",
    is_flag=True,
    default=False,
    help="Save the mask as an own band. Will be the last band.",
)
def main(
    satellite,
    start_date,
    end_date,
    resolution,
    geojson_path,
    output_folder,
    max_cloud_cover,
    cloudprob_thresh,
    snowprob_thresh,
    num_workers,
    save_mask,
):
    print("Starting satellite data download pipeline.")

    # Define satellite specific setup
    satellite = satellite.lower()
    if satellite == "s2":
        stac_collection_name = "sentinel-2-c1-l2a"
        stack_catalog_url = "https://earth-search.aws.element84.com/v1"
        metadata_asset_names = ["granule_metadata"]
        mask_bands = ["scl", "cloud", "snow"]
        mask_processor = build_s2_mask_processor(
            cloud_thresh=cloudprob_thresh, snowprob_thresh=snowprob_thresh
        )
        band_processor = add_geometry_bands

        # 'cosVZA', 'cosSZA', 'cosRAA' will be prepended by the custom band processor
        if resolution == 10:
            # ['B2', 'B3', 'B4', 'B8']
            bands = ["blue", "green", "red", "nir"]
        else:
            # ['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
            bands = [
                "green",
                "red",
                "rededge1",
                "rededge2",
                "rededge3",
                "nir08",
                "swir16",
                "swir22",
            ]

    else:
        raise Exception("Currently only supporting S2 download.")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set up parallel processing
    if num_workers is None:
        num_workers = max(1, int(multiprocessing.cpu_count() - 1))
    print(f"Using {num_workers} workers out of {multiprocessing.cpu_count()} available cores")

    # Read area of interest
    print("Reading GeoDataFrame")
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.to_crs(epsg=4326)

    run_pipeline(
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
        roi_gdf=gdf,
        bands=bands,
        output_folder=output_folder,
        stack_catalog_url=stack_catalog_url,
        stac_collection_name=stac_collection_name,
        max_cloud_cover=max_cloud_cover,
        num_workers=num_workers,
        metadata_asset_names=metadata_asset_names,
        mask_bands=mask_bands,
        save_mask=save_mask,
        mask_band_processor=mask_processor,
        custom_band_processor=band_processor,
    )
    print("Download Pipeline completed successfully.")


if __name__ == "__main__":
    main()
