import os
import numpy as np
import pystac_client
import geopandas as gpd
import time
import multiprocessing
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import concurrent.futures
from datetime import datetime
import subprocess
import xml.etree.ElementTree as ET
from rasterio.windows import from_bounds

import requests

# This script fetches all required Sentinel-2 tiles that intersect our ROI
# These are then resampled to the target resolution and and saved to disk
# We are also fetching the SCL band to mask clouds
# Additionally, we create geometry bands for each tile
# We create a VRT for each tile compising all bands
    
def apply_scl_mask(data, scl_data, nodata_val, valid_scl_classes=None):
    if valid_scl_classes is None:
        # Default: Keep vegetation (4), non-vegetated (5), water (6), unclassified (7)
        valid_scl_classes = [4, 5, 6, 7]

    if data.shape != scl_data.shape:
        raise ValueError(f"Shape mismatch: data shape {data.shape} vs scl shape {scl_data.shape}")

    mask = np.isin(scl_data, valid_scl_classes)
    
    nodata_val = data.dtype.type(nodata_val)
    masked_data = np.where(mask, data, nodata_val)

    # Harmonize by subtracting 1000 from the data, but only where not nodata
    # Is used to align with GEE harmonized data 
    # DOING THIS IN LAI SCRIPT NOW
    # masked_data = np.where(masked_data != nodata_val, masked_data - 1000, nodata_val)
    
    return masked_data

def save_band(metadata, data, item, band_name, resolution, output_folder):
    output_path = os.path.join(output_folder, f"{item.id}_{band_name}_{resolution}m.tif")
    
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1,
        crs=metadata['crs'],
        transform=metadata['transform'],
        nodata=metadata['nodata'],
        compress='LZW',
        dtype=metadata['dtype'],
        tiled=True
    ) as dst:
        dst.write(data, 1)

    return output_path

def resample_to_resolution(tile_path, resolution, item, band_name, bbox=None, scl_ref=None, output_folder=None):
    """
    Resample the downloaded tile to the desired resolution (10m or 20m) entirely in memory.
    Uses SCL at the target resolution as reference for the extent if provided.
    """
    if resolution not in [10, 20]:
        raise ValueError(f"Resolution must be either 10 or 20, got {resolution}")
    
    # Band resolutions defined as before
    band_resolutions = {
        'red': 10, 'green': 10, 'blue': 10, 'nir08': 20,
        'rededge1': 20, 'rededge2': 20, 'rededge3': 20,
        'swir16': 20, 'swir22': 20, 'scl': 20 
    }

    scl_keep_classes = [4, 5, 6, 7]
    
    native_resolution = band_resolutions.get(band_name)
    if not native_resolution:
        raise ValueError(f"Unknown band resolution for {band_name}")

    t0 = time.time()
    with rasterio.open(tile_path) as src:

        
        if bbox:
            minx, miny, maxx, maxy = bbox
            minx = max(minx, src.bounds.left)
            maxx = min(maxx, src.bounds.right)
            miny = max(miny, src.bounds.bottom)
            maxy = min(maxy, src.bounds.top)
            src_window = from_bounds(minx, miny, maxx, maxy, src.transform)
            window_transform = src.window_transform(src_window)
            src_data = src.read(1, window=src_window)
            window_height, window_width = src_data.shape
        else:
            src_data = src.read(1)
            window_transform = src.transform
            window_height, window_width = src.height, src.width
        
        # Skip resampling if already at target resolution
        if native_resolution == resolution:
            # Create metadata with the correct transform
            metadata = {
                'transform': window_transform,
                'crs': src.crs,
                'width': window_width,
                'height': window_height,
                'nodata': src.nodata,
                'resolution': resolution,
                'dtype': src_data.dtype
            }
            
            # Apply masking if needed
            if band_name != 'scl' and scl_ref is not None:
                data = apply_scl_mask(src_data, scl_ref, src.nodata, scl_keep_classes)
            else:
                data = src_data
            
            print(f"Read and processed {band_name} in {time.time() - t0:.2f} seconds")
            if band_name == 'scl':
                return data
            else:
                t1 = time.time()
                out_path = save_band(metadata, data, item, band_name, resolution, output_folder)
                print(f"Saved {band_name} in {time.time() - t1:.2f} seconds")
                return band_name, out_path
        
        # Find reference band for target resolution
        ref_band_url = None
        for asset_name, asset in item.assets.items():
            if asset_name in band_resolutions and band_resolutions[asset_name] == resolution:
                ref_band_url = asset.href
                break
        
        if not ref_band_url:
            raise ValueError(f"Could not find a reference band with {resolution}m resolution")
        
        # Open reference band
        with rasterio.open(ref_band_url) as ref_src:
            if bbox:
                minx, miny, maxx, maxy = bbox
                minx = max(minx, ref_src.bounds.left)
                maxx = min(maxx, ref_src.bounds.right)
                miny = max(miny, ref_src.bounds.bottom)
                maxy = min(maxy, ref_src.bounds.top)

                ref_window = from_bounds(minx, miny, maxx, maxy, ref_src.transform)
                dst_transform = ref_src.window_transform(ref_window)
                
                #TODO need to refactor this as we dont want to always read the reference band again!
                data = ref_src.read(1, window=ref_window)
                dst_height, dst_width = data.shape
            else:
                dst_transform = ref_src.transform
                dst_width = ref_src.width
                dst_height = ref_src.height
                
            dst_nodata = ref_src.nodata
            
        # Initialize destination array
        dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)
        if src.nodata is not None:
            dst_data.fill(src.nodata)
        
        # Reproject with careful attention to transforms
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=window_transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=ref_src.crs,
            src_nodata=src.nodata,
            dst_nodata=dst_nodata,
            resampling=Resampling.nearest,
            num_threads=4
        )
        
        metadata = {
            'transform': dst_transform,
            'crs': ref_src.crs,
            'width': dst_width,
            'height': dst_height,
            'nodata': dst_nodata,
            'resolution': resolution,
            'dtype': dst_data.dtype
        }

        # Apply SCL mask if needed
        if band_name != 'scl' and scl_ref is not None:
            t2 = time.time()
            data = apply_scl_mask(dst_data, scl_ref, dst_nodata, scl_keep_classes)
            print(f"Applied SCL mask for band {band_name} in {time.time() - t2:.2f} seconds")
        else:
            data = dst_data
        
        if band_name == 'scl':
            return data
        else:
            # Save with correct transform
            t1 = time.time()
            out_path = save_band(metadata, data, item, band_name, resolution, output_folder)
            print(f"Saved {band_name} in {time.time() - t1:.2f} seconds")
            return band_name, out_path

def create_bandstacked_vrt(output_file_path, band_paths):
    # use gdal to create a VRT

    band_paths_str = ' '.join(band_paths)

    # Want to update gdal to use better blocksize
    command = f"gdalbuildvrt -separate -overwrite {output_file_path} {band_paths_str}"
    subprocess.run(command, shell=True, check=True)

    # Additionally create an actual geotiff with the stacked bands for quick debugging
    # output_file_path_gtiff = output_file_path.replace('.vrt', '.tif')
    # command = f"gdal_translate -of GTiff -co TILED=YES -co compress=LZW {output_file_path} {output_file_path_gtiff}"
    # subprocess.run(command, shell=True, check=True)

    return output_file_path

def get_geometry_data(metadata_xml_url):
    """Get geometry-related data from granule metadata xml."""
    # Download metadata xml

    response = requests.get(metadata_xml_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download metadata XML, status code: {response.status_code}")


    xml_root = ET.fromstring(response.content)
    azimuth_angle_el= xml_root.findall('.//Mean_Sun_Angle/AZIMUTH_ANGLE')[0]
    azimuth_angle_units = azimuth_angle_el.attrib['unit']
    if azimuth_angle_units != 'deg':
        raise Exception(f"azimuth_angle_units must be 'deg', but it is {azimuth_angle_units}.")
    azimuth_angle = float(azimuth_angle_el.text)
    
    zenith_angle_el = xml_root.findall('.//Mean_Sun_Angle/ZENITH_ANGLE')[0]
    zenith_angle_units = zenith_angle_el.attrib['unit']
    if zenith_angle_units != 'deg':
        raise Exception(f"zenith_angle_units must be 'deg', but it is {zenith_angle_units}.")
    zenith_angle = float(zenith_angle_el.text)

    b8a_incidence_angle_el = xml_root.findall(".//Mean_Viewing_Incidence_Angle_List/Mean_Viewing_Incidence_Angle[@bandId='9']")

    if not b8a_incidence_angle_el:
        raise Exception("Could not find Mean_Viewing_Incidence_Angle for band 9.")

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

def compute_cos_angles(zenith_angle, azimuth_angle, mean_incidence_zenith_angle_b8a, mean_incidence_azimuth_angle_b8a):
    cos_vza = np.cos(np.deg2rad(mean_incidence_zenith_angle_b8a)) * 10000
    cos_sza = np.cos(np.deg2rad(zenith_angle)) * 10000
    cos_raa = np.cos(np.deg2rad(azimuth_angle - mean_incidence_azimuth_angle_b8a)) * 10000
    return {
        "cos_vza": cos_vza,
        "cos_sza": cos_sza,
        "cos_raa": cos_raa,
    }

def create_geometry_bands(item, cos_angles, scl_ref, metadata, output_folder):
    geometry_band_paths = {}
    
    # Create each geometry band
    for angle_name, angle_value in cos_angles.items():
        # Create empty array with same dimensions as other bands
        band_data = np.full((metadata['height'], metadata['width']), 
                           angle_value, 
                           dtype=metadata['dtype'])
        
        # Apply SCL mask
        scl_keep_classes = [4, 5, 6, 7]  # Keep vegetation, non-vegetated, water, unclassified
        masked_data = apply_scl_mask(band_data, scl_ref, metadata['nodata'], scl_keep_classes)
        
        # Save the geometry band
        output_path = os.path.join(output_folder, f"{item.id}_{angle_name}_{metadata['resolution']}m.tif")
        
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=metadata['height'], width=metadata['width'],
            count=1,
            crs=metadata['crs'],
            transform=metadata['transform'],
            nodata=metadata['nodata'],
            compress='LZW',
            dtype=metadata['dtype'],
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as dst:
            dst.write(masked_data, 1)
        
        geometry_band_paths[angle_name] = output_path
        print(f"Created geometry band {angle_name}")
    
    return geometry_band_paths

def process_scene(item, bands, gdf, resolution, output_folder):

    gdf = gdf.to_crs(item.properties['proj:code'])
    bbox = gdf.geometry.values[0].bounds

    t0 = time.time()
    scl_band_path = item.assets['scl'].href
    print(f"Processing SCL band {scl_band_path}")
    scl_ref = resample_to_resolution(scl_band_path, resolution, item, 'scl', bbox)
    print(f"Resampled SCL band in {time.time() - t0:.2f} seconds")

    # Now process all other bands using SCL reference
    print(f"Processing remaining bands with SCL masking")
    t1 = time.time()
    band_paths = {}

    for band in bands:
        if band == 'scl':
            continue

        band_path = item.assets[band].href
        print(f"Processing band {band} at {band_path}")
        band_name, out_path = resample_to_resolution(band_path, resolution, item, band, bbox, scl_ref, output_folder=output_folder)
        band_paths[band_name] = out_path

    # with concurrent.futures.ThreadPoolExecutor(max_workers=len(bands)) as executor:
    #     band_futures = {}
    #     for band in bands:
    #         if band == 'scl':
    #             continue

    #         band_path = item.assets[band].href
    #         future = executor.submit(resample_to_resolution, band_path, resolution, item, band, bbox, scl_ref, output_folder=output_folder)
    #         band_futures[future] = (item.id, item, band)
        
    #     # Wait for all futures to complete
    #     for future in concurrent.futures.as_completed(band_futures):
    #         result = future.result()
    #         if result:
    #             band_name, band_path = result
    #             band_paths[band_name] = band_path
    
    print(f"Resampled and masked in {time.time() - t1:.2f} seconds")

    # Step 3: Compute geometry bands        
    print('Computing geometry bands')
    t3 = time.time()
    metadata_xml_url = item.assets['granule_metadata'].href
    geometry_data = get_geometry_data(metadata_xml_url)
    cos_angles = compute_cos_angles(
        geometry_data['zenith_angle'],
        geometry_data['azimuth_angle'],
        geometry_data['mean_incidence_zenith_angle_b8a'],
        geometry_data['mean_incidence_azimuth_angle_b8a']
    )

    # Get the reference metadata from the first band
    first_band_path = band_paths[bands[0]]
    with rasterio.open(first_band_path) as src:
        reference_metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'nodata': src.nodata,
            'resolution': resolution,
            'dtype': np.uint16
        }
    
    # Create the geometry bands using the reference metadata
    geometry_band_paths = create_geometry_bands(item, cos_angles, scl_ref, reference_metadata, output_folder)
    print(f"Created geometry bands in {time.time() - t3:.2f} seconds")
    
    # Add geometry bands to band paths
    band_paths.update(geometry_band_paths)

    # Ensure order of bands: cosVZA, cosSZA, cosRAA, followed by the other bands in order of input
    band_paths_sorted = [band_paths[band] for band in ['cos_vza', 'cos_sza', 'cos_raa'] + bands]

    # Step 4: Create VRT
    t2 = time.time()
    print(f'VRT paths {band_paths_sorted}')

    
    vrt_path = os.path.join(output_folder, f"{item.id}_{resolution}m_{item.datetime.strftime('%Y-%m-%d')}.vrt")
    create_bandstacked_vrt(vrt_path, band_paths_sorted)
    print(f"Created VRT in {time.time() - t2:.2f} seconds")
    
    return vrt_path

def download_process_parallel(items, bands, gdf, resolution, output_folder, num_workers):
    output_paths = []
    
    #max_workers = int(num_workers / len(bands))
    #print(f"Using {max_workers} workers for downloading and processing")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for item in items:
            future = executor.submit(process_scene, item, bands, gdf, resolution, output_folder)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                output_paths.append(result)
    
    return output_paths


def main():
    print("Starting satellite data processing pipeline")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Open the STAC catalog
    print("Opening catalog")
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    
    # Define bands to process
    bands = ['green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir08', 'swir16', 'swir22']
    # Note: 'scl' will be added automatically in the process_tile_data function

    # TODOs
    # - Remove / mask pixels outside geometry
    # - Improve parallelization
    # - Use Cloud and Snow prob bands additionally for masking
    
    # Set target resolution (10 or 20 meters)
    resolution = 10
    
    # Set paths
    geojson_path = '/gpfs/data1/cmongp2/sawahnr/nasa-harvest/vercye/playground/maroc_adm_0_demo_data/mar_admbnda_adm0_hcp_20230925.shp'
    output_folder = '/gpfs/data1/cmongp2/sawahnr/data/misc/vercye_dl_tiles_moroc_test'
    os.makedirs(output_folder, exist_ok=True)
    
    # Set up parallel processing
    num_workers = max(1, int(multiprocessing.cpu_count()))
    print(f"Using {num_workers} workers out of {multiprocessing.cpu_count()} available cores")
    
    # Read area of interest
    print("Reading GeoDataFrame")
    gdf = gpd.read_file(geojson_path)
    
    # Set up search parameters
    year = 2020
    start_date = f"{year}-01-01"
    end_date = f"{year}-01-31"
    
    # Search for items
    print(f"Searching for items from {start_date} to {end_date}")
    t0 = time.time()
    search = catalog.search(
        collections=["sentinel-2-c1-l2a"],
        intersects=gdf.geometry.values[0],
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 70}},
    )

    # Get items from search
    items = search.item_collection()
    print(f"Found {len(items)} items")
    print(f"Search took {time.time() - t0:.2f} seconds")
    
    # Process all items
    t1 = time.time()
    output_paths = download_process_parallel(items, bands, gdf, resolution, output_folder, num_workers)
    
    print(f"\nCreated {len(output_paths)} VRT files")
    print(output_paths)
    print(f"\nTotal runtime: {time.time() - t1:.2f} seconds")


if __name__ == '__main__':
    main()