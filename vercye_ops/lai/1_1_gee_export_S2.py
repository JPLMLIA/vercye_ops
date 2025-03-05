import ee
import json
import click
from pathlib import Path
import time
from datetime import datetime, timedelta
from glob import glob

import geopandas as gpd

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

@click.command()
@click.option('--project', help='GEE Project Name')
@click.option('--library', default="library/", type=click.Path(file_okay=False), help='Local Path to the library folder')
@click.option('--region', help='Region name (without apostrophes)')
@click.option('--shpfile', type=click.Path(exists=True), help='Local Path to the shapefile to override region')
@click.option('--start-date', type=click.DateTime(formats=["%Y-%m-%d"]), help='Start date for the image collection')
@click.option('--end-date', type=click.DateTime(formats=["%Y-%m-%d"]), help='End date for the image collection')
def main(project, library=None, region=None, shpfile=None, start_date="2021-09-01", end_date="2021-10-01"):

    # Initialize Earth Engine
    ee.Initialize(project=project)

    # Start timing run
    all_start = time.time()

    if shpfile is None and region is None:
        raise ValueError("Either a shapefile or oblast should be specified.")
    elif shpfile is not None and region is not None:
        raise ValueError("Only one of shapefile or crop mask should be specified.")

    # Get geometry 
    if region is not None:
        # Look for geojson in the library
        oblast_geojsons = glob(f"{library}/{region}.geojson")

        if len(oblast_geojsons) == 0:
            raise FileNotFoundError(f"{library}/{region}.geojson not found.")
        
        shp = json_to_fc(oblast_geojsons[0])
        print(f"Loaded geometry from {oblast_geojsons[0]}")
    elif shpfile is not None:
        # Override with shp file
        shp = shp_to_fc(shpfile)
        print(f"Loaded geometry from {shpfile}")
        
    geometry = shp.geometry()
    if region is not None:
        geometry_name = region
    else:
        geometry_name = Path(shpfile).stem

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
        BAND_NAMES = ['cosVZA', 'cosSZA', 'cosRAA', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
        S2_mosaic = S2_mosaic.select(ee.List(BAND_NAMES))
        S2_mosaic = S2_mosaic.toInt16()

        # Export to Google Drive
        print(f"Exporting {current_datestr} to Google Drive...")

        task = ee.batch.Export.image.toDrive(image=S2_mosaic,
                    description=f"{geometry_name}_{current_datestr}_10m",
                    folder=f"{geometry_name}_10m",
                    scale=10,
                    fileFormat='GeoTIFF',
                    maxPixels=1e13,
                    region=geometry,
                    #fileDimensions=6144,
                    skipEmptyTiles=True)
        start = time.time()
        task.start()
        
        while(task.active()):
            print(f"Task {task.id} is {task.status()['state']} ({time.time()-start:.2f}s elapsed)")
            time.sleep(3)
        
        print(f"Task {task.id} is {task.status()['state']} in {time.time()-start:.2f}s")
        current_date = next_date
    
    print(f"Completed in {time.time()-all_start:.2f}s")

if __name__ == "__main__":

    # Authenticate to Earth Engine
    ee.Authenticate()

    main()
