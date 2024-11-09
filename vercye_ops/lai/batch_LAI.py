import os.path as op
from pathlib import Path
from datetime import datetime, timedelta
import time
import click
import csv

import numpy as np
from scipy.interpolate import CubicSpline
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
@click.argument('geometry_name', type=str)
@click.option('--adjustment', options=["none", "wheat", "maize"], default="none", help='Adjustment to apply to the LAI estimate')
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='Start date for the image collection')
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), help='End date for the image collection')
@click.option('--model_weights', type=click.Path(exists=True), default='s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth', help='Local Path to the model weights')
def main(s2_dir, lai_dir, geometry_name, adjustment, start_date, end_date, model_weights="s2_sl2p_weiss_or_prosail_NNT3_Single_0_1_LAI.pth"):
    """ Main LAI batch prediction function

    S2_dir: Local Path to the Sentinel-2 images
    LAI_dir: Local Path to the LAI estimates
    geometry_name: Name of the geometry

    This pipeline does the following:
    1. For each date starting from start_date to end_date
        2. Looks for Sentinel-2 images in the specified directory in the format {geometry_name}_{date}.tif
            2a. If the image is not found, it add a blank row to the CSV output list and continues to the next date
        3. Loads the Sentinel-2 image
            3a. If the image only has zeros, it adds a blank row to the CSV output and continues to the next date
        4. Uses the pytorch model to predict LAI
        6. Exports the LAI estimate to the specified directory in the format {geometry_name}_{date}_LAI.tif
        7. Calculates a running maximum of the LAI estimate across all dates
        8. Adds a row to the CSV output list with the Date, mean, median, min, max, and stddev of all non-nan pixels in the LAI estimate
    9. Exports the CSV output list to the specified directory in the format {geometry_name}_{start_date}_{end_date}_STATS.csv
    10. Exports the running maximum LAI estimate to the specified directory in the format {geometry_name}_{start_date}_{end_date}_LAIMAX.tif
    """

    start = time.time()

    # Load the pytorch model
    model = LAI_CNN(11, 5, 1)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    # Iterate through each date
    dates = [start_date + timedelta(days=i) for i in range((end_date-start_date).days+1)]
    dates = [date.strftime("%Y-%m-%d") for date in dates]

    statistics = []
    lai_max = None
    lai_adjusted_max = None
    for d in dates:
        # The csv requires the date in day/month/year format
        d_slash = datetime.strptime(d, "%Y-%m-%d").strftime("%d/%m/%Y")

        # See if the image exists
        raster_path = op.join(s2_dir, f"{geometry_name}_{d}.tif")
        if not op.exists(raster_path):
            print(f"Skipping {d} because {raster_path} does not exist")
            
            stat = {
                "Date": d_slash,
                "n_pixels": 0,
                "interpolated": 0,
                "LAI Mean": None,
                "LAI Stddev": None,
                "LAI Mean Adjusted": None,
                "LAI Stddev Adjusted": None
            }
            statistics.append(stat)
            continue
            
        # Load the image
        src = rio.open(raster_path)
        array = src.read()

        # If the last band of the image is all zeros, skip
        # The first three bands are geometry values
        if np.all(array[-1] == 0):
            print(f"Skipping {d} because {raster_path} is all zeros")
            stat = {
                "Date": d_slash,
                "n_pixels": 0,
                "interpolated": 0,
                "LAI Mean": None,
                "LAI Stddev": None,
                "LAI Mean Adjusted": None,
                "LAI Stddev Adjusted": None
            }
            statistics.append(stat)
            continue

        # Built-in scaling
        array = array * 0.0001

        # Input
        x = torch.tensor(array, dtype=torch.float32).unsqueeze(0)
        LAI_estimate = model(x)
        LAI_estimate = LAI_estimate.squeeze(0).squeeze(0).detach().numpy()
        LAI_estimate[array[-1] == 0] = np.nan

        # Adjustment
        if adjustment == "none":
            LAI_adjusted = LAI_estimate
        if adjustment == "wheat":
            LAI_adjusted = LAI_estimate**2 * 0.0482 + LAI_estimate * 0.9161 + 0.0026
        elif adjustment == "maize":
            LAI_adjusted = LAI_estimate**2 * -0.078 + LAI_estimate * 1.4 - 0.18

        # Export as GeoTIFF
        filename = op.join(lai_dir, f"{geometry_name}_{d}_LAI_{adjustment}-adj.tif")
        profile = src.profile
        profile.update(count=3, dtype='float32', compress='lzw', nodata=np.nan, driver='GTiff')
        with rio.open(filename, 'w', **profile) as dst:
            dst.write(LAI_estimate, 1)
            dst.write(LAI_adjusted, 2)
            # Set band description to estimateLAI
            dst.set_band_description(1, "estimateLAI")
            dst.set_band_description(2, "adjustedLAI")
        print(f"Exported {filename}")
        
        # Calculate statistics
        statistics.append({
            "Date": d_slash,
            "n_pixels": np.sum(~np.isnan(LAI_estimate)),
            "interpolated": 0,
            "LAI Mean": np.nanmean(LAI_estimate),
            "LAI Stddev": np.nanstd(LAI_estimate),
            "LAI Mean Adjusted": np.nanmean(LAI_adjusted),
            "LAI Stddev Adjusted": np.nanstd(LAI_adjusted),
        })

        # Update running maximum
        if lai_max is None:
            lai_max = LAI_estimate
        else:
            lai_max = np.nanmax([lai_max, LAI_estimate], axis=0)
        
        if lai_adjusted_max is None:
            lai_adjusted_max = LAI_adjusted
        else:
            lai_adjusted_max = np.nanmax([lai_adjusted_max, LAI_adjusted], axis=0)

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    # Cubic splite interpolation

    for col in ["LAI Mean", "LAI Adjusted"]:
        real_X = []
        real_Y = []
        nan_X = []
        for i, stat in enumerate(statistics):
            if stat[col] is None:
                nan_X.append(i)
            else:
                real_X.append(i)
                real_Y.append(stat[col])
        
        cs = CubicSpline(np.array(real_X), np.array(real_Y), bc_type='not-a-knot')
        for i in nan_X:
            statistics[i][col] = max(cs(i),0)
            statistics[i]["interpolated"] = 1


    # Write statistics to CSV
    filename = op.join(lai_dir, f"{geometry_name}_{start_date}_{end_date}_STATS_{adjustment}-adj.csv")
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, fieldnames=statistics[0].keys())
        writer.writeheader()
        writer.writerows(statistics)
    print(f"Exported {filename}")
    
    # Export running maximum
    # Set 0 to nan
    filename = op.join(lai_dir, f"{geometry_name}_{start_date}_{end_date}_STATS.csv")
    with rio.open(filename, 'w', **profile) as dst:
        dst.write(lai_max, 1)
        dst.write(lai_adjusted_max, 2)
        dst.set_band_description(1, "estimateLAImax")
        dst.set_band_description(2, "adjustedLAImax")
    print(f"Exported {filename}")

    print(f"Finished in {time.time()-start:.2f} seconds")

if __name__ == "__main__":
    main()