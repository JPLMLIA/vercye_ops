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
import torch.nn as nn

import xml.etree.ElementTree as ET

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class Scale2d(nn.Module):
    def __init__(self, n_ch):
        super(Scale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))

    def forward(self, x):
        return x * self.weight + self.bias


class UnScale2d(nn.Module):
    def __init__(self, n_ch):
        super(UnScale2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, n_ch, 1, 1))

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

    for f in source_files:
        if os.path.exists(f) and not os.path.isdir(f):
            logger.info(f"Deleting: {f}")
            os.remove(f)
        else:
            logger.info(f"Not found: {f}")


def worker_process_files(
    worker_id, file_batch, model_weights, lai_dir, remove_original
):
    """Worker function that processes a batch of files with a single model instance"""
    logger.info(f"Worker {worker_id} starting, processing {len(file_batch)} files")

    # Load the model once per worker
    model = LAI_CNN(11, 5, 1)
    model.load_state_dict(torch.load(model_weights))
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

        # If the last band of the image is all zeros, skip
        if np.all(s2_array[-1] == 0):
            logger.info(f"Skipping {Path(vrt_path).name} because it is all zeros")
            s2_ds.close()
            return None
        else:
            logger.info(f"Processing {Path(vrt_path).name}")

        # Built-in scaling
        s2_array = s2_array * 0.0001

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
        logger.info(f"Model prediction for {Path(vrt_path).name} in {time.time()-t1:.2f} seconds")

        # set NODATA to nan
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
    "--model-weights",
    type=click.Path(exists=True),
    default="../trained_models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth",
    help="Local Path to the model weights",
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

    if resolution != 20:
        logger.warning(
            f"Currently only the 20m model is implemented in the STAC pipeline. If you are sure you want to use {resolution}m, remove this code.."
        )
        return

    # Get all the VRT files
    vrt_files = sorted(glob(f"{s2_dir}/*_{resolution}m_*.vrt"))

    if start_date is not None and end_date is not None:
        vrt_files = [vf for vf in vrt_files if is_within_date_range(vf, start_date, end_date)]

    logger.info(f"Found {len(vrt_files)} VRT files at {resolution}m in {s2_dir}")

    # Divide files into batches for each worker
    batch_size = max(1, len(vrt_files) // num_cores)
    file_batches = []

    for i in range(num_cores):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < num_cores - 1 else len(vrt_files)
        file_batches.append(vrt_files[start_idx:end_idx])

    # Create a process pool with fixed number of workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []

        for i, file_batch in enumerate(file_batches):
            futures.append(
                executor.submit(
                    worker_process_files,
                    i,
                    file_batch,
                    model_weights,
                    lai_dir,
                    remove_original
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
