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

from vercye_ops.lai.model import load_model_from_weights, load_model

# See https://code.earthengine.google.com/?accept_repo=users/rfernand387/LEAFToolboxModules for details
default_model_weights = {
    "S2": {
        10: {
            "weights_path": "../trained_models/s2_sl2p_weiss_or_prosail_10m_NNT1_Single_0_1_LAI.pth",
            "channels": ["cosVZA", "cosSZA", "cosRAA", "B2", "B3", "B4", "B8"],
        },
        20: {
            "weights_path": "../trained_models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth",
            "channels": [
                "cosVZA",
                "cosSZA",
                "cosRAA",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8A",
                "B11",
                "B12",
            ],
        },
    }
}


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
    # files are expected to have the pattern f"{s2_dir}/{region}_{resolution}m_{date}.vrt"
    date = Path(vf).stem.split("_")[-1]
    date = datetime.strptime(date, "%Y-%m-%d")
    return start_date <= date <= end_date


@click.command()
@click.argument("S2_dir", type=click.Path(exists=True))
@click.argument("LAI_dir", type=click.Path(exists=True))
@click.argument("region", type=str)
@click.argument("resolution", type=int)
@click.option(
    "--start_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date",
    required=False,
    default=None,
)
@click.option(
    "--end_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date",
    required=False,
    default=None,
)
@click.option(
    "--model_weights",
    type=click.Path(exists=True),
    default=None,
    help="Local Path to the model weights. Default values for 10 and 20m resolution available.",
)
@click.option(
    "--channels",
    type=str,
    default=None,
    help="Input channels to use. String with comma separetes band names e.g cosVZA,cosRZA,cosSZA,B3,B4....",
    required=False,
)
def main(
    s2_dir,
    lai_dir,
    region,
    resolution,
    start_date,
    end_date,
    model_weights="../trained_models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth",
    channels=None,
):
    """Main LAI batch prediction function

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

    if model_weights is not None and not channels:
        raise ValueError("channels must be specified if model_weights is provided.")

    if model_weights is not None and channels is not None:
        channels = [ch.strip() for ch in channels.split(",")]
        num_in_ch = len(channels)

        model = load_model_from_weights(model_weights, channels)
    else:
        sateillite = 'S2'
        model = load_model(sateillite, resolution)
        model.eval()

    # Get all the VRT files
    vrt_files = sorted(glob(f"{s2_dir}/{region}_{resolution}m_*.vrt"))

    if start_date is not None and end_date is not None:
        vrt_files = [vf for vf in vrt_files if is_within_date_range(vf, start_date, end_date)]

    print(f"Found {len(vrt_files)} VRT files for {region} at {resolution}m in {s2_dir}")

    for vf in vrt_files:
        # Load the image
        s2_ds = rio.open(vf)
        s2_array = s2_ds.read()

        if not s2_array.shape[0] == model.num_in_ch:
            raise ValueError(
                f"Number of bands in {vf} does not match the number of input channels. Expected {num_in_ch} but got {s2_array.shape[0]}"
            )

        # If the last band of the image is all zeros, skip
        # The first three bands are geometry values
        if np.all(s2_array[-1] == 0):
            print(f"Skipping {Path(vf).name} because it is all zeros")
            continue
        else:
            print(f"Processing {Path(vf).name}")

        # Built-in scaling
        s2_array = s2_array * 0.0001

        # Input
        s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)

        # Run model
        LAI_estimate = model(s2_tensor)
        LAI_estimate = LAI_estimate.cpu().squeeze(0).squeeze(0).detach().numpy()
        # set NODATA to nan
        LAI_estimate[s2_array[-1] == 0] = np.nan

        # Export as GeoTIFF
        filename = op.join(lai_dir, Path(vf).stem + "_LAI.tif")
        profile = s2_ds.profile
        profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan, driver="GTiff")
        with rio.open(filename, "w", **profile) as dst:
            dst.write(LAI_estimate, 1)
            dst.set_band_description(1, "estimateLAI")
        print(f"Exported {filename}")
        s2_ds.close()

    print(f"Finished in {time.time()-start:.2f} seconds")


if __name__ == "__main__":
    main()
