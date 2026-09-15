import concurrent
import os
import os.path as op
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from glob import glob
from pathlib import Path

import click
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import torch

from vercye_ops.lai.model.model import load_model
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()
logger.setLevel("INFO")


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


def worker_process_files(worker_id, file_batch, lai_dir, remove_original, satellite, resolution):
    """Worker function that processes a batch of files with a single model instance"""
    logger.info(f"Worker {worker_id} starting, processing {len(file_batch)} files")

    # Load the model once per worker
    model = load_model(satellite, resolution)
    model.eval()
    logger.info("Model loaded")

    output_files = []

    for vf in file_batch:
        output_file = process_single_file(vf, model, lai_dir, remove_original)
        if output_file:
            output_files.append(output_file)

    logger.info(f"Worker {worker_id} finished processing {len(file_batch)} files")
    return output_files


# Rows per model forward pass. 1098 divides a 10980-row S2 tile exactly into 10 stripes and
# keeps one stripe's float32 buffer near 0.5 GB for an 11-band tile.
STRIPE_ROWS = 1098


def process_single_file(vrt_path, model, lai_dir, remove_original):
    """Process a single VRT file with the provided model and return the output filename"""
    logger.info(f"Processing ... {vrt_path}")

    # Load the image
    with rio.open(vrt_path) as s2_ds:
        profile = s2_ds.profile
        nodata_val = s2_ds.nodata

        if nodata_val is None:
            raise ValueError(f"Received tif with no nodata value set. Can't process {vrt_path}.")

        # Validate that correct number of input bands is provided.
        if not s2_ds.count == model.num_in_ch:
            raise ValueError(
                f"Number of bands in {vrt_path} does not match the number of input channels. Expected {model.num_in_ch} but got {s2_ds.count}"
            )

        # If the last band of the image is all nodata, skip
        if np.all(s2_ds.read(s2_ds.count) == nodata_val):
            logger.info(f"Skipping {Path(vrt_path).name} because it is all zeros")
            s2_ds.close()
            return None
        else:
            logger.info(f"Processing {Path(vrt_path).name}")

        # Run the model over horizontal stripes rather than the whole tile at once.
        #
        # The LAI network (lai/model/model.py) is Scale2d -> Conv2d(k=1) -> Tanh ->
        # Conv2d(k=1) -> UnScale2d: every layer is 1x1, so an output pixel depends only on
        # the input pixel at the same position. Striping is therefore exactly equivalent to
        # a whole-tile forward pass, not an approximation -- but it holds one stripe instead
        # of a full 11 x 10980 x 10980 tile, cutting peak RSS from ~38 GB to ~3 GB and
        # letting many more workers run on a box with no swap.
        t1 = time.time()
        height, width = s2_ds.height, s2_ds.width
        LAI_estimate = np.empty((height, width), dtype=np.float32)

        for row0 in range(0, height, STRIPE_ROWS):
            nrows = min(STRIPE_ROWS, height - row0)
            window = Window(0, row0, width, nrows)
            stripe = s2_ds.read(window=window)

            # Set NODATA to nan. Casting to float32 rather than letting np.where upcast to
            # float64 is bit-identical (int16 -> float32 is exact) and halves the buffer.
            mask = stripe == nodata_val
            stripe = stripe.astype(np.float32)
            stripe[mask] = np.nan

            # Built-in scaling
            # Now handling in model directly stripe = stripe * 0.0001

            # from_numpy shares the buffer instead of copying it (stripe is already float32)
            stripe_tensor = torch.from_numpy(stripe).unsqueeze(0)

            with torch.no_grad():
                stripe_out = model(stripe_tensor)
            LAI_estimate[row0 : row0 + nrows, :] = stripe_out.cpu().squeeze(0).squeeze(0).numpy()

        logger.info(f"Model prediction for {Path(vrt_path).name} in {time.time()-t1:.2f} seconds")

    # Write the LAI data
    filename = op.join(lai_dir, Path(vrt_path).stem + "_LAI_tile.tif")

    if os.path.exists(filename):
        os.remove(filename)

    profile.update(
        count=1,
        dtype="float32",
        compress="lzw",
        nodata=np.nan,
        driver="GTiff",
        blockxsize=256,
        blockysize=256,
        tiled=True,
    )
    with rio.open(filename, "w", **profile) as dst:
        dst.write(LAI_estimate, 1)
        # Set band description to estimateLAI
        dst.set_band_description(1, "estimateLAI")

    if remove_original:
        # Accumulate all files linked to the VRT
        delete_vrt_and_linked_tifs(vrt_path)

    return filename


@click.command()
@click.argument("imagery-dir", type=click.Path(exists=True))
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
    "--satellite",
    type=str,
    default="S2",
    help="Imagery from Satellite type: S2 for Sentinel-2, HLS_S30 for HLS Sentinel Version, HLS_L30 for HLS Landsat version.",
)
@click.option(
    "--remove-original",
    is_flag=True,
    help="Remove original VRT files AND linked tifs after processing",
    default=False,
)
def main(imagery_dir, lai_dir, resolution, start_date, end_date, num_cores, satellite, remove_original):
    """
    Main function to process Sentinel-2 VRT files and generate LAI estimates.

    """

    start_time = time.time()
    logger.info(f"Using {num_cores} parallel workers")

    # Get all the VRT files
    vrt_files = sorted(glob(f"{imagery_dir}/*_{resolution}m_*.vrt"))
    print(vrt_files)

    if start_date is not None and end_date is not None:
        vrt_files = [vf for vf in vrt_files if is_within_date_range(vf, start_date, end_date)]
    print(vrt_files)
    logger.info(f"Found {len(vrt_files)} VRT files at {resolution}m in {imagery_dir}")

    # Divide files into batches for each worker
    file_batches = []
    n = len(vrt_files)
    base_size = n // num_cores
    remainder = n % num_cores

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
                    satellite,
                    resolution,
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
                traceback.print_exc()
                raise e

    output_files = [file for batch_result in all_results for file in batch_result if file is not None]

    logger.info(f"Processed {len(output_files)} files successfully")
    logger.info(f"Finished in {time.time()-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
