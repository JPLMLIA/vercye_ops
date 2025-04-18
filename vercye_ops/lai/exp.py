from datetime import datetime
import os
import os.path as op
from pathlib import Path
from glob import glob
import time
from collections import Counter
import click
import subprocess

import numpy as np
import torch
import torch.nn as nn
import rasterio as rio

from functools import partial
from rasterio.warp import calculate_default_transform, reproject, Resampling
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


class Scale2d(nn.Module):
    def __init__(self, n_ch):
        super(Scale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1,n_ch,1,1))
        self.bias = nn.Parameter(torch.Tensor(1,n_ch,1,1))

    def forward(self, x):
        return x * self.weight + self.bias

class UnScale2d(nn.Module):
    def __init__(self, n_ch):
        super(UnScale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1,n_ch,1,1))
        self.bias = nn.Parameter(torch.Tensor(1,n_ch,1,1))

    def forward(self, x):
        return (x - self.bias) / self.weight

class LAI_CNN(nn.Module):
    def __init__(self, in_ch, h1_dim, out_ch):
        super(LAI_CNN, self).__init__()
        self.input = Scale2d(in_ch)
        self.h1 = nn.Conv2d(in_ch, h1_dim, 1, 1, 0, bias=True)
        self.h2 = nn.Conv2d(h1_dim, out_ch, 1, 1, 0, bias=True)
        self.output = UnScale2d(out_ch)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input(x)
        x = self.h1(x)
        x = self.tanh(x)
        x = self.h2(x)
        x = self.output(x)
        return x
    
def is_within_date_range(vf, start_date, end_date):
    # files have pattern f"{s2_dir}/{region}_{resolution}m_{date}.vrt"
    date = Path(vf).stem.split("_")[-1]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date

def get_most_common_crs(vrt_files):
    """Identify the most common CRS from all VRT files"""
    crs_counts = {}
    for vf in vrt_files:
        with rio.open(vf) as src:
            if src.crs is None:
                raise ValueError(f"CRS is None for {vf}")
                
            crs = src.crs
            if crs in crs_counts:
                crs_counts[crs][0] += 1
            else:
                crs_counts[crs] = [1, vf]
    
    # Get the most common CRS
    most_common_el = max(crs_counts.items(), key=lambda x: x[1][0])
    return most_common_el[0], most_common_el[1][1]

def determine_target_resolution(vrt_file, target_crs='EPSG:4326'):
    """
    Sample one file to determine appropriate resolution for the target CRS
    that maintains similar level of detail
    """
    # Use first non-empty file for sampling
    with rio.open(vrt_file) as src:
        #if src.count > 0 and not np.all(src.read(1) == 0):
            # Calculate transform to target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        # Calculate resolution in target CRS units
        x_res = (src.bounds.right - src.bounds.left) / src.width
        y_res = (src.bounds.top - src.bounds.bottom) / src.height
        
        # Get sample points for resolution calculation
        left, bottom, right, top = rio.warp.transform_bounds(
            src.crs, target_crs, *src.bounds)
        
        # Calculate equivalent resolution in target CRS
        x_res_target = (right - left) / dst_width
        y_res_target = (top - bottom) / dst_height
        
        print(f"Sample file: {Path(vrt_file).name}")
        print(f"Original resolution: {x_res:.8f}, {y_res:.8f} in {src.crs}")
        print(f"Target resolution: {x_res_target:.8f}, {y_res_target:.8f} in {target_crs}")
        src = None  # Explicitely Close the file
    src = None
    return (x_res_target, y_res_target)


def process_single_file(vf, model, lai_dir, target_resolution, target_crs):
    """Process a single VRT file with the provided model and return the output filename"""
    print(f"Processing {vf}")
    # Load the image
    t0 = time.time()
    print(f"Loading {vf}")
    s2_ds = rio.open(vf)
    print('Opened')
    print(s2_ds.crs)
    s2_array = s2_ds.read()
    print(f"Loaded {vf} in {time.time()-t0:.2f} seconds")
    original_crs = s2_ds.crs
    print(f"Dataload for {Path(vf).name} in {time.time()-t0:.2f} seconds")
    filename = op.join(lai_dir, Path(vf).stem + "_LAI_tile.tif")
    s2_ds.close()
    return filename

@click.command()
@click.argument('S2_dir', type=click.Path(exists=True))
@click.argument('LAI_dir', type=click.Path(exists=True))
@click.argument('region', type=str)
@click.argument('resolution', type=int)
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='Start date', required=False, default=None)
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='End date', required=False, default=None)
@click.option('--model_weights', type=click.Path(exists=True), default='models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth', help='Local Path to the model weights')
def main(s2_dir, lai_dir, region, resolution, start_date, end_date, model_weights="models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth"):
    """ Main LAI batch prediction function

    S2_dir: Local Path to the .vrt Sentinel-2 images

    LAI_dir: Local Path to the LAI estimates

    region: Name of the region. Used to match file names beginning with region_

    resolution: Spatial resolution in meters. Used to match file names beginning with region_resolution

    This pipeline does the following:
    1. Looks for Sentinel-2 images in the specified directory in the format {geometry_name}_{resolution}m_{date}.vrt
    2. Uses the pytorch model to predict LAI
    3. Exports the LAI estimate to the specified directory in the format {geometry_name}_{resolution}m_{date}_LAI.tif
    """

    start = time.time()
    num_processes = 4
    print(f"Using {num_processes} parallel workers")

    # Get all the VRT files
    vrt_files = sorted(glob(f"{s2_dir}/*_{resolution}m_*.vrt"))

    if start_date is not None and end_date is not None:
        vrt_files = [vf for vf in vrt_files if is_within_date_range(vf, start_date, end_date)]

    print(f"Found {len(vrt_files)} VRT files for {region} at {resolution}m in {s2_dir}")

    t0 = time.time()
    print("Identifying most common CRS...")
    target_crs = 'EPSG:4326'
    most_common_crs, most_common_crs_file = get_most_common_crs(vrt_files)
    print(f"Most common CRS: {most_common_crs}")
    print(f"Target CRS: {target_crs}")
    
    print("Determining target resolution...")
    target_resolution = determine_target_resolution(most_common_crs_file, 'EPSG:4326')
    print(f"Using target resolution: {target_resolution[0]:.8f}, {target_resolution[1]:.8f}")
    print(f"Time taken to identify most common CRS: {time.time()-t0:.2f} seconds")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        for f in vrt_files:
            futures.append(executor.submit(process_single_file, f, None, None, None, None))

        # Wait for all futures to complete
        all_results = []
        for future in as_completed(futures):
            _ = future.result()  # can handle result if needed



if __name__ == "__main__":
    main()