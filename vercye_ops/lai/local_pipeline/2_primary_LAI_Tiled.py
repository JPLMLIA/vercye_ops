from datetime import datetime
import os
import os.path as op
from pathlib import Path
from glob import glob
import time
from collections import Counter
import click
import subprocess

import concurrent
import numpy as np
import torch
import torch.nn as nn
import rasterio as rio

from functools import partial
from rasterio.warp import calculate_default_transform, reproject, Resampling


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
    # TODO Use first non-empty file for sampling
    with rio.open(vrt_file) as src:
        # if src.count > 0 and not np.all(src.read(1) == 0):
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
    src = None # Explicitely Close the file 
    return (x_res_target, y_res_target)

#OLD
def process_vrt_file(vf, model_weights, lai_dir, target_resolution, target_crs):
    """Process a single VRT file with the model and return the output filename"""
    # Load the pytorch model
    model = LAI_CNN(11, 5, 1)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    try:
        # Load the image
        t0 = time.time()
        s2_ds = rio.open(vf)
        s2_array = s2_ds.read()
        original_crs = s2_ds.crs
        print(f"Dataload in {time.time()-t0:.2f} seconds")

        # If the last band of the image is all zeros, skip
        if np.all(s2_array[-1] == 0):
            print(f"Skipping {Path(vf).name} because it is all zeros")
            s2_ds.close()
            return None, None
        else:
            print(f"Processing {Path(vf).name}")

        left, bottom, right, top = rio.warp.transform_bounds(original_crs, target_crs, *s2_ds.bounds)
            
        x_res, y_res = target_resolution
        width = int((right - left) / x_res)
        height = int((top - bottom) / y_res)
        
        # Create the transformation for output
        dst_transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

        # Built-in scaling
        s2_array = (s2_array * 0.0001)
        
        # subtract 0.1 where the array is not nodata
        s2_array[s2_array != s2_ds.nodata] -= 0.1

        # Input
        t1 = time.time()
        s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)

        # Run model
        LAI_estimate = model(s2_tensor)
        LAI_estimate = LAI_estimate.cpu().squeeze(0).squeeze(0).detach().numpy()
        print(f"Model prediction in {time.time()-t1:.2f} seconds")

        # set NODATA to nan
        nodata_val = s2_ds.nodata
        LAI_estimate[s2_array[-1] == nodata_val] = np.nan

        profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'float32',
                'crs': target_crs,
                'transform': dst_transform,
                'compress': 'lzw',
                'nodata': np.nan
            }
            
        # Create in-memory array for reprojection
        t2 = time.time()
        dst_array = np.zeros((height, width), dtype=np.float32)
        dst_array.fill(np.nan)  # Fill with NaN initially
        
        # Reproject the LAI estimate to the target CRS with consistent resolution
        reproject(
            LAI_estimate,
            dst_array,
            src_transform=s2_ds.transform,
            src_crs=original_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan
        )
        print(f"Reprojection in {time.time()-t2:.2f} seconds")
        
        # Write the reprojected data
        filename = op.join(lai_dir, Path(vf).stem + "_LAI_tile.tif")
        with rio.open(filename, 'w', **profile) as dst:
            dst.write(dst_array, 1)
            # Set band description to estimateLAI
            dst.set_band_description(1, "estimateLAI")
    
        print(f"Exported {filename}")
        s2_ds.close()
        return filename
    
    except Exception as e:
        print(f"Error processing {vf}: {e}")
        return None, None

def worker_process_files(worker_id, file_batch, model_weights, lai_dir, target_resolution, target_crs):
    """Worker function that processes a batch of files with a single model instance"""
    print(f"Worker {worker_id} starting, processing {len(file_batch)} files")
    
    # Load the model once per worker
    model = LAI_CNN(11, 5, 1)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    print('Model loaded')
    
    output_files = []
    
    for vf in file_batch:
        output_file = process_single_file(vf, model, lai_dir, target_resolution, target_crs)
        if output_file:
            output_files.append(output_file)
    
    print(f"Worker {worker_id} finished processing {len(file_batch)} files")
    return output_files

def process_single_file(vf, model, lai_dir, target_resolution, target_crs):
    """Process a single VRT file with the provided model and return the output filename"""
    print(f"Processing ... {vf}")
    # Load the image
    t0 = time.time()
    print(f"Loading {vf}")
    s2_ds = rio.open(vf)
    print('Opened')
    s2_array = s2_ds.read()
    original_crs = s2_ds.crs
    print(f"Dataload for {Path(vf).name} in {time.time()-t0:.2f} seconds")

    # If the last band of the image is all zeros, skip
    if np.all(s2_array[-1] == 0):
        print(f"Skipping {Path(vf).name} because it is all zeros")
        s2_ds.close()
        return None
    else:
        print(f"Processing {Path(vf).name}")

    left, bottom, right, top = rio.warp.transform_bounds(original_crs, target_crs, *s2_ds.bounds)

    x_res, y_res = target_resolution
    width = int((right - left) / x_res)
    height = int((top - bottom) / y_res)
    
    # Create the transformation for output
    dst_transform = rio.transform.from_bounds(left, bottom, right, top, width, height)

    # Built-in scaling
    s2_array = (s2_array * 0.0001)
    
    # subtract 0.1 where the array is not nodata (Due to baseline >= 5 and need t match GEE harmonized collection)
    nodata_val = s2_ds.nodata
    s2_array[s2_array != nodata_val] -= 0.1

    # Input
    t1 = time.time()
    s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)

    # Run model
    with torch.no_grad():
        LAI_estimate = model(s2_tensor)
    LAI_estimate = LAI_estimate.cpu().squeeze(0).squeeze(0).numpy()
    print(f"Model prediction for {Path(vf).name} in {time.time()-t1:.2f} seconds")

    # set NODATA to nan
    LAI_estimate[s2_array[-1] == nodata_val] = np.nan

    profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'float32',
            'crs': target_crs,
            'transform': dst_transform,
            'compress': 'lzw',
            'nodata': np.nan
        }
        
    # Create in-memory array for reprojection
    dst_array = np.zeros((height, width), dtype=np.float32)
    dst_array.fill(np.nan)
    
    # Reproject the LAI estimate to the target CRS with consistent resolution
    reproject(
        LAI_estimate,
        dst_array,
        src_transform=s2_ds.transform,
        src_crs=original_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )
    
    # Write the reprojected data
    filename = op.join(lai_dir, Path(vf).stem + "_LAI_tile.tif")
    with rio.open(filename, 'w', **profile) as dst:
        dst.write(dst_array, 1)
        # Set band description to estimateLAI
        dst.set_band_description(1, "estimateLAI")

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
    num_processes = 32
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

    # Divide files into batches for each worker
    batch_size = len(vrt_files) // num_processes
    file_batches = []
    
    for i in range(num_processes):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_processes - 1 else len(vrt_files)
        file_batches.append(vrt_files[start_idx:end_idx])
    
    # Create a process pool with fixed number of workers

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        
        for i, file_batch in enumerate(file_batches):
            futures.append(executor.submit(worker_process_files, i, file_batch, model_weights, lai_dir, target_resolution, target_crs))

        # Wait for all futures to complete
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Worker {i} processed {len(result)} files successfully")
                    all_results.append(result)
            except Exception as e:
                print(f"Error in worker {i}: {e}")
    
    # Flatten list of lists to get all output files
    output_files = [file for batch_result in all_results for file in batch_result if file is not None]
    print(f"Processed {len(output_files)} files successfully")

    # Group output_files by date
    output_files_by_date = {}
    for file in output_files:
        date = Path(file).stem.split("_")[-3]
        if date not in output_files_by_date:
            output_files_by_date[date] = []
        output_files_by_date[date].append(file)

    # Create daily mosaics (VRT)
    for date, files in output_files_by_date.items():
        mosaic_filename = op.join(lai_dir, f"{region}_{resolution}m_{date}_LAI.vrt")
        print(f"Creating mosaic for {date}...")
        cmd = ['gdalbuildvrt', '-overwrite', mosaic_filename] + files
        subprocess.run(cmd)
        print(f"Created mosaic {mosaic_filename}")

    # TODO Clean up S2 files

    print(f"Finished in {time.time()-start:.2f} seconds")

if __name__ == "__main__":
    main()