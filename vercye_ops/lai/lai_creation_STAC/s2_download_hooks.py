from datetime import datetime, timezone
import os
import xml.etree.ElementTree as ET
from functools import partial
from typing import Dict, List

import numpy as np
import rasterio as rio
from pystac.item import Item as pyStacItem


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
        raise Exception(
            f"azimuth_angle_units must be 'deg', but it is {azimuth_angle_units}."
        )
    azimuth_angle = float(azimuth_angle_el.text)

    zenith_angle_el = xml_root.findall(".//Mean_Sun_Angle/ZENITH_ANGLE")[0]
    zenith_angle_units = zenith_angle_el.attrib["unit"]
    if zenith_angle_units != "deg":
        raise Exception(
            f"zenith_angle_units must be 'deg', but it is {zenith_angle_units}."
        )
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
        raise Exception("mean_incidence_azimuth_angle_b8a must be in degrees.")
    mean_incidence_azimuth_angle_b8a = float(mean_incidence_azimuth_angle_b8a_el.text)

    mean_incidence_zenith_angle_b8a_el = b8a_incidence_angle_el.find("ZENITH_ANGLE")
    if mean_incidence_zenith_angle_b8a_el.attrib["unit"] != "deg":
        raise Exception("mean_incidence_zenith_angle_b8a must be in degrees.")
    mean_incidence_zenith_angle_b8a = float(mean_incidence_zenith_angle_b8a_el.text)

    return {
        "azimuth_angle": azimuth_angle,
        "zenith_angle": zenith_angle,
        "mean_incidence_azimuth_angle_b8a": mean_incidence_azimuth_angle_b8a,
        "mean_incidence_zenith_angle_b8a": mean_incidence_zenith_angle_b8a,
    }


def compute_cos_angles(
    zenith_angle,
    azimuth_angle,
    mean_incidence_zenith_angle_b8a,
    mean_incidence_azimuth_angle_b8a,
):
    cos_vza = np.uint16(np.cos(np.deg2rad(mean_incidence_zenith_angle_b8a)) * 10000)
    cos_sza = np.uint16(np.cos(np.deg2rad(zenith_angle)) * 10000)
    # Converting to int16 to match GEE script
    cos_raa = np.int16(
        np.cos(np.deg2rad(azimuth_angle - mean_incidence_azimuth_angle_b8a)) * 10000
    )
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
        band_data = np.full(
            (metadata["height"], metadata["width"]), angle_value, dtype=geo_dtype
        )

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

    return geometry_band_paths

def build_geometry_band_adder(granule_mtd_asset_name):
    return partial(
        add_geometry_bands,
        granule_mtd_asset_name = granule_mtd_asset_name
    )


def add_geometry_bands(
    item: pyStacItem,
    band_paths: Dict[str, str],
    band_names: List[str],
    mask: np.ndarray,
    file_asset_paths: Dict[str, str],
    resolution: float,
    output_folder: str,
    granule_mtd_asset_name = "granule_metadata"
):
    if not file_asset_paths or granule_mtd_asset_name not in file_asset_paths:
        raise ValueError(f"File asset paths must contain '{granule_mtd_asset_name}'.")

    graunle_metadata_file = file_asset_paths[granule_mtd_asset_name]

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
    geometry_band_paths = create_geometry_bands(
        item, cos_angles, reference_metadata, output_folder
    )

    # Ensure correct bandorder for output
    band_paths["cos_vza"] = geometry_band_paths["cos_vza"]
    band_paths["cos_sza"] = geometry_band_paths["cos_sza"]
    band_paths["cos_raa"] = geometry_band_paths["cos_raa"]
    band_names = ["cos_vza", "cos_sza", "cos_raa"] + band_names

    return band_paths, band_names


def build_s2_masking_hook(
    cloud_thresh=None,
    snowprob_thresh=None,
    scl_keep_classes=[4, 5],
    scl_bandname='scl',
    cloudprob_bandname='cloud',
    snowprob_bandname='snow'
):
    # Factory function that returns a Sentinel-2 mask processing function.
    # Parameters like scl_keep_classes, cloud_thresh, and snowprob_thresh are fixed at creation time.
    # SCL default classes to keep are [4, 5] (vegetation and non-vegetation)

    return partial(
        s2_mask_processor,
        scl_keep_classes=scl_keep_classes,
        cloud_thresh=cloud_thresh,
        snowprob_thresh=snowprob_thresh,
        scl_bandname=scl_bandname,
        cloudprob_bandname=cloudprob_bandname,
        snowprob_bandname=snowprob_bandname
    )


def s2_mask_processor(maskbands, scl_keep_classes, cloud_thresh, snowprob_thresh, scl_bandname='scl', cloudprob_bandname='cloud', snowprob_bandname='snow'):
    mask = None

    scl_band_meta, scl_band = maskbands[scl_bandname]
    mask = np.ones_like(scl_band)  # Start with all valid (1)

    # Invalidate pixels based on SCL
    mask = np.where(np.isin(scl_band, scl_keep_classes), mask, 0)

    # Invalidate pixels based on cloud probability
    # Currently we are fallingback on S2A-L2A non-collection-1, for 2022/23
    # This is temporary however it does not include the S2Cloudless band
    if "cloud" in maskbands:
        s2cloudless_band_meta, s2cloudless_band = maskbands[cloudprob_bandname]
        mask = np.where(s2cloudless_band >= cloud_thresh, 0, mask)

    # Invalidate pixels based on snow probability
    if "snow" in maskbands:
        snowprob_band_meta, snowprob_band = maskbands[snowprob_bandname]
        mask = np.where(snowprob_band >= snowprob_thresh, 0, mask)

    new_metadata = {}

    return new_metadata, mask

def s2_harmonization_processor(raster: np.ndarray, raster_profile: dict, item: pyStacItem):
    # Harmonizes scenes freom before 2022-01-25 to match the baseline >= 4.0 format
    # Adds a shift of +1000 to match the ESA introduced shift
    tile_date = item.datetime

    if tile_date < datetime(2022, 1, 25, tzinfo=timezone.utc):
        nodata_val = raster_profile['nodata']
        raster = np.where(raster != nodata_val, raster + 1000, nodata_val)

    return raster, raster_profile