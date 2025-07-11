import concurrent
import os
import os.path as op
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import numpy as np
import rasterio as rio
import torch

import xml.etree.ElementTree as ET

from vercye_ops.lai.model.model import load_model


import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def is_within_date_range(vf, start_date, end_date):
    # files have pattern f"{s2_dir}/{tileID}_{resolution}m_{date}.vrt"
    date = Path(vf).stem.split("_")[-1]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date


def delete_vrt_and_linked_tifs(vrt_path):
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    source_files = []
    for elem in root.iter():
        if elem.tag.endswith("SourceFilename"):
            filepath = elem.text
            if filepath:
                filepath = filepath.strip()
                if elem.attrib.get("relativeToVRT", "0") == "1":
                    filepath = os.path.join(os.path.dirname(vrt_path), filepath)
                source_files.append(filepath)

    source_files.append(vrt_path)

    # TODO add metadata path for deletion aswell

    for f in source_files:
        if os.path.exists(f) and not os.path.isdir(f):
            logger.info(f"Deleting: {f}")
            os.remove(f)
        else:
            logger.info(f"Not found: {f}")


def worker_process_files(
    worker_id, file_batch, lai_dir, remove_original, sateillite, resolution
):
    """Worker function that processes a batch of files with a single model instance"""
    logger.info(f"Worker {worker_id} starting, processing {len(file_batch)} files")

    # Load the model once per worker
    model = load_model(sateillite, resolution)
    model.eval()
    logger.info("Model loaded")

    output_files = []

    for vf in file_batch:
        output_file = process_single_file(vf, model, lai_dir, remove_original)
        if output_file:
            output_files.append(output_file)

    logger.info(f"Worker {worker_id} finished processing {len(file_batch)} files")
    return output_files


def process_single_file(vrt_path, model, lai_dir, remove_original):
    """Process a single VRT file with the provided model and return the output filename"""
    logger.info(f"Processing ... {vrt_path}")

    # Load the image
    with rio.open(vrt_path) as s2_ds:
        s2_array = s2_ds.read()
        profile = s2_ds.profile

        # Validate that correct number of input bands is provided.
        if not s2_array.shape[0] == model.num_in_ch:
            raise ValueError(
                f"Number of bands in {vrt_path} does not match the number of input channels. Expected {model.num_in_ch} but got {s2_array.shape[0]}"
            )

        # If the last band of the image is all zeros, skip
        if np.all(s2_array[-1] == 0):
            logger.info(f"Skipping {Path(vrt_path).name} because it is all zeros")
            s2_ds.close()
            return None
        else:
            logger.info(f"Processing {Path(vrt_path).name}")

        # Built-in scaling
        s2_array = s2_array * 0.0001

        # Input
        t1 = time.time()
        s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)

        # Run model
        with torch.no_grad():
            LAI_estimate = model(s2_tensor)
        LAI_estimate = LAI_estimate.cpu().squeeze(0).squeeze(0).numpy()
        logger.info(f"Model prediction for {Path(vrt_path).name} in {time.time()-t1:.2f} seconds")

        # Set NODATA to nan
        nodata_val = s2_ds.nodata
        LAI_estimate[s2_array[-1] == nodata_val] = np.nan

    # Write the LAI data
    filename = op.join(lai_dir, Path(vrt_path).stem + "_LAI_tile.tif")

    if os.path.exists(filename):
        os.remove(filename)

    profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan, driver="GTiff", blockxsize=256, blockysize=256)
    with rio.open(filename, "w", **profile) as dst:
        dst.write(LAI_estimate, 1)
        # Set band description to estimateLAI
        dst.set_band_description(1, "estimateLAI")

    if remove_original:
       # Accumulate all files linked to the VRT
       delete_vrt_and_linked_tifs(vrt_path)

    return filename


@click.command()
@click.argument("S2-dir", type=click.Path(exists=True))
@click.argument("LAI-dir", type=click.Path(exists=True))
@click.argument("resolution", type=int)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date",
    required=False,
    default=None,
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date",
    required=False,
    default=None,
)
@click.option(
    "--num-cores",
    type=int,
    default=64,
    help="Number of workers (cores) to use.",
)
@click.option(
    "--remove-original",
    is_flag=True,
    help="Remove original VRT files AND linked tifs after processing",
    default=False,
)

def main(s2_dir, lai_dir, resolution, start_date, end_date, num_cores, model_weights, remove_original):
    """
    Main function to process Sentinel-2 VRT files and generate LAI estimates.
    
    """

    start = time.time()
    logger.info(f"Using {num_cores} parallel workers")

    # Currently only supporting Sentinel-2 yet
    sateillite = 'S2'

    # Get all the VRT files
    vrt_files = sorted(glob(f"{s2_dir}/*_{resolution}m_*.vrt"))

    if start_date is not None and end_date is not None:
        vrt_files = [vf for vf in vrt_files if is_within_date_range(vf, start_date, end_date)]

    logger.info(f"Found {len(vrt_files)} VRT files at {resolution}m in {s2_dir}")

    # Divide files into batches for each worker
    file_batches = []
    n = len(vrt_files)
    base_size  = n // num_cores
    remainder  = n % num_cores

    start = 0
    for i in range(num_cores):
        # each of the first `remainder` batches gets one extra file
        this_batch_size = base_size + (1 if i < remainder else 0)
        end = start + this_batch_size
        file_batches.append(vrt_files[start:end])
        start = end

    # Create a process pool with fixed number of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []

        for i, file_batch in enumerate(file_batches):
            futures.append(
                executor.submit(
                    worker_process_files,
                    i,
                    file_batch,
                    lai_dir,
                    remove_original,
                    sateillite,
                    resolution
                )
            )

        # Wait for all futures to complete
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    logger.info(f"Worker {i} processed {len(result)} files successfully")
                    all_results.append(result)
            except Exception as e:
                logger.info(f"Error in worker {i}: {e}")

    output_files = [
        file for batch_result in all_results for file in batch_result if file is not None
    ]

    logger.info(f"Processed {len(output_files)} files successfully")
    logger.info(f"Finished in {time.time()-start:.2f} seconds")


if __name__ == "__main__":
    main()
