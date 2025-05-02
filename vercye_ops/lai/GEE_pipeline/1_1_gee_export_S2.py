import os
import ee
import json
import click
from pathlib import Path
import time
from datetime import datetime, timedelta
from glob import glob
import unicodedata
import concurrent.futures

import geopandas as gpd
from gdrive_download_helpers import get_drive_service, download_files_from_drive, delete_files_from_drive, find_files_in_drive


def json_to_fc(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return ee.FeatureCollection(json_data)

def shp_to_fc(shp_path):
    shape = gpd.read_file(shp_path)
    shape_json = json.loads(shape.to_json())
    return ee.FeatureCollection(shape_json)

def validate_date_format(date_string):
    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def maskLowQA(image):
    qaBand = 'cs'
    clearThreshold = 0.60
    mask = image.select(qaBand).gte(clearThreshold)
    return image.updateMask(mask)

def maskSnow(image):
    snowBand = 'MSK_SNWPRB'
    snowThreshold = 50
    mask = image.select(snowBand).lt(snowThreshold)
    return image.updateMask(mask)

def addGeometry(image):
    return (image.addBands(image.metadata("MEAN_INCIDENCE_ZENITH_ANGLE_B8A").multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(['cosVZA']))
                .addBands(image.metadata("MEAN_SOLAR_ZENITH_ANGLE").multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(['cosSZA']))
                .addBands(image.metadata("MEAN_SOLAR_AZIMUTH_ANGLE").subtract(image.metadata("MEAN_INCIDENCE_AZIMUTH_ANGLE_B8A")).multiply(3.1415).divide(180).cos().multiply(10000).toInt16().rename(['cosRAA'])))

def clean_name(name):
    # Remove special characters to avoid GEE export issues
    return ''.join(
        c for c in unicodedata.normalize('NFKD', name)
        if not unicodedata.combining(c)
    )

def process_gdrive_download(drive_service, folder_name, file_description, download_folder):
    time.sleep(30)
            
    drive_files = find_files_in_drive(drive_service, folder_name, file_description)
    
    if drive_files:
        print(f"Found {len(drive_files)} files for export task")
        
        # Download all the files
        downloaded = download_files_from_drive(
            drive_service, 
            drive_files, 
            os.path.join(download_folder, folder_name),
            file_description
        )
        
        # Delete the files from Drive if downloads were successful
        if downloaded:
            file_ids = [file_id for file_id, _ in downloaded]
            delete_files_from_drive(drive_service, file_ids)
        else:
            file_ids = [file_id for file_id, _ in downloaded]
            raise RuntimeError(f"Failed to download files: {file_ids}")
    else:
        raise RuntimeError(f"Could not find exported file in Google Drive.")

@click.command()
@click.option('--project', help='GEE Project Name')
@click.option('--library', default="library/", type=click.Path(file_okay=False), help='Local Path to the library folder')
@click.option('--region', help='Region name (without apostrophes)')
@click.option('--shpfile', type=click.Path(exists=True), help='Local Path to the shapefile to override region')
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), help='Start date for the image collection')
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), help='End date for the image collection')
@click.option('--resolution', type=int, default=20, help='Spatial resolution in meters per pixel.')
@click.option('--export-mode', type=click.Choice(['gdrive', 'gcs']), default='gdrive', help='Export mode: Google Drive or Google Cloud Storage. Attention: GCS will come with egress costs!')
@click.option('--export-bucket', type=str, help='Google Cloud Storage bucket name', required=False)
@click.option('--gcs-folder-path', type=str, help='Google Cloud Storage folder path in bucket', required=False)
@click.option('--gdrive-credentials', type=click.Path(exists=True), help='Path to Google Drive credentials JSON file.', required=False)
@click.option('--download-folder', type=click.Path(), help='Path where to save the downloaded S2 data.', required=False)
def main(project, library=None, region=None, shpfile=None, start_date="2021-09-01", end_date="2021-10-01", resolution=20, export_mode='drive', export_bucket=None, gcs_folder_path=None, gdrive_credentials=None, download_folder=None):

    if export_mode == 'gcs' and (export_bucket is None or gcs_folder_path is None):
        raise ValueError("Export bucket must be specified for GCS export mode.")

    if export_mode == 'gcs':
        print("GCS export mode is selected. This may incur egress costs. Please check the GCS documentation for pricing details.")

    # Initialize Earth Engine
    ee.Initialize(project=project)

    drive_service = None
    download_executor = None
    if export_mode == 'gdrive' and gdrive_credentials is not None:
        drive_service = get_drive_service(gdrive_credentials)
        print("Successfully connected to Google Drive API")
        
        max_workers = 4
        download_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    # Start timing run
    all_start = time.time()

    if shpfile is None and region is None:
        raise ValueError("Either a shapefile or administrative division (region) should be specified.")
    elif shpfile is not None and region is not None:
        raise ValueError("Only one of shapefile or crop mask should be specified.")

    # Get geometry 
    if region is not None:
        # Look for geojson in the library
        region_geojsons = glob(f"{library}/{region}.geojson")

        if len(region_geojsons) == 0:
            raise FileNotFoundError(f"{library}/{region}.geojson not found.")
        
        shp = json_to_fc(region_geojsons[0])
        print(f"Loaded geometry from {region_geojsons[0]}")
    elif shpfile is not None:
        # Override with shp file
        shp = shp_to_fc(shpfile)
        print(f"Loaded geometry from {shpfile}")
        
    geometry = shp.geometry()
    if region is not None:
        geometry_name = region
    else:
        geometry_name = Path(shpfile).stem

    geometry_name = clean_name(geometry_name)


    # Get the Sentinel-2 image collection
    S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

    # Start iterating through date pairs
    current_date = start_date
    while current_date < end_date:
        next_date = current_date + timedelta(days=1)

        current_datestr = current_date.strftime("%Y-%m-%d")
        next_datestr = next_date.strftime("%Y-%m-%d")
        # Filter on geometry and date
        S2_filtered = (S2.filterBounds(geometry).filterDate(current_datestr, next_datestr))

        # Filter on cloudy percentage and cloud score and snow percentage
        CLOUDY_THRESHOLD = 80
        CS_THRESHOLD = 0.6
        SNOW_THRESHOLD = 50
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        csPlusBands = csPlus.first().bandNames()
        S2_filtered = (S2_filtered.linkCollection(csPlus, csPlusBands)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUDY_THRESHOLD))
                        .map(lambda image: image.updateMask(image.select('cs').gte(CS_THRESHOLD)))
                        .map(lambda image: image.updateMask(image.select('MSK_SNWPRB').lt(SNOW_THRESHOLD)))
        )

        if S2_filtered.size().getInfo() == 0:
            print(f"No images found for {current_datestr}. Skipping...")
            current_date = next_date
            continue

        # Add geometry bands
        S2_filtered = S2_filtered.map(addGeometry)

        # Mosaic
        S2_mosaic = ee.Image(S2_filtered.mosaic())
        
        # Select bands
        if resolution == 10:
            BAND_NAMES = ['cosVZA', 'cosSZA', 'cosRAA', 'B2', 'B3', 'B4', 'B8']
        else:
            BAND_NAMES = ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

        S2_mosaic = S2_mosaic.select(ee.List(BAND_NAMES))
        S2_mosaic = S2_mosaic.toInt16()

        file_description = f"{geometry_name}_{str(resolution)}m_{current_datestr}"
        folder_name = f"{geometry_name}_{str(resolution)}m"            

        if export_mode == 'gdrive':
            print(f"Exporting {current_datestr} to Google Drive...")

            task = ee.batch.Export.image.toDrive(
                image=S2_mosaic,
                description = f"{geometry_name}_{str(resolution)}m_{current_datestr}",
                folder = f"{geometry_name}_{str(resolution)}m",
                scale=resolution,
                fileFormat='GeoTIFF',
                maxPixels=1e13,
                region=geometry,
                #fileDimensions=6144,
                skipEmptyTiles=True)

        elif export_mode == 'gcs':
            print(f"Exporting {current_datestr} to Googles Cloud Storage...")
            task = ee.batch.Export.image.toCloudStorage(image=S2_mosaic,
                        description=file_description,
                        fileNamePrefix=f"{gcs_folder_path}/{folder_name}/{file_description}",
                        bucket=export_bucket,
                        scale=resolution,
                        fileFormat='GeoTIFF',
                        maxPixels=1e13,
                        region=geometry,
                        #fileDimensions=6144,
                        skipEmptyTiles=True)
        else:
            raise ValueError("Invalid export mode. Choose either 'drive' or 'gcs'.")

        start = time.time()
        task.start()
        
        # Wait for the task to complete - only want to have one task running at a time per region
        while(task.active()):
            print(f"Task {task.id} is {task.status()['state']} ({time.time()-start:.2f}s elapsed)")
            time.sleep(3)
        
        task_status = task.status()
        print(f"Task {task.id} is {task.status()['state']} in {time.time()-start:.2f}s")

        # Check if the task was successful and download the file if in GDrive mode to free up space in GDrive
        # While this will prevent the next task from starting, this is a good way to avoid running out of space in GDrive
        # If this is too slow, we can do the downloading with a seperate background process
        if task_status['state'] == 'COMPLETED' and drive_service is not None:
            print(f"Task completed successfully, launching download task from Google Drive...")
            download_executor.submit(
                process_gdrive_download,
                drive_service,
                folder_name,
                file_description,
                download_folder
            )

        current_date = next_date
    
    print(f"Completed in {time.time()-all_start:.2f}s")

if __name__ == "__main__":
    # Authenticate to Earth Engine
    ee.Authenticate()

    main()
