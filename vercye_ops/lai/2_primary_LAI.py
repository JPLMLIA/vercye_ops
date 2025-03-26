import os.path as op
from pathlib import Path
from glob import glob
import time
import click

import numpy as np
import torch
import torch.nn as nn
import rasterio as rio

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

@click.command()
@click.argument('S2_dir', type=click.Path(exists=True))
@click.argument('LAI_dir', type=click.Path(exists=True))
@click.argument('region', type=str)
@click.argument('resolution', type=int)
@click.option('--model_weights', type=click.Path(exists=True), default='models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth', help='Local Path to the model weights')
def main(s2_dir, lai_dir, region, resolution, model_weights="models/s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth"):
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

    # Load the pytorch model
    model = LAI_CNN(11, 5, 1)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    # Get all the VRT files
    vrt_files = sorted(glob(f"{s2_dir}/{region}_{resolution}m_*.vrt"))
    print(f"Found {len(vrt_files)} VRT files for {region} at {resolution}m in {s2_dir}")

    for vf in vrt_files:
        # Load the image
        s2_ds = rio.open(vf)
        s2_array = s2_ds.read()

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
        #s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)
        s2_tensor = torch.tensor(s2_array, dtype=torch.float32).unsqueeze(0)
        # Run model
        LAI_estimate = model(s2_tensor)
        LAI_estimate = LAI_estimate.cpu().squeeze(0).squeeze(0).detach().numpy()
        # set NODATA to nan
        LAI_estimate[s2_array[-1] == 0] = np.nan

        # Export as GeoTIFF
        filename = op.join(lai_dir, Path(vf).stem + "_LAI.tif")
        profile = s2_ds.profile
        profile.update(count=1, dtype='float32', compress='lzw', nodata=np.nan, driver='GTiff')
        with rio.open(filename, 'w', **profile) as dst:
            dst.write(LAI_estimate, 1)
            # Set band description to estimateLAI
            dst.set_band_description(1, "estimateLAI")
        print(f"Exported {filename}")
        s2_ds.close()
        
    print(f"Finished in {time.time()-start:.2f} seconds")

if __name__ == "__main__":
    main()
