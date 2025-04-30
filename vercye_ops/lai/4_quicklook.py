import os
import click
import csv
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.argument('max_path', type=click.Path(exists=True))
def main(csv_path, max_path):
    """Produces a quicklook of LAI analysis results
    
    This script produces a quicklook plot of the LAI curve and the maximum raster to verify the successful completion of the script.
    """

    out_filepath = csv_path.replace('STATS.csv', 'QUICKLOOK.png')
    suptitle = Path(csv_path).stem

    # open CSV
    lai_stats = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lai_stats.append(row)
    
    # open max raster
    max_ds = rio.open(max_path)
    max_array = max_ds.read()

    # Initialize figure
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.subplot2grid((2, 2), (0, 0), fig=fig)
    ax2 = plt.subplot2grid((2, 2), (0, 1), fig=fig)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)

    # Rasters
    cm_min = np.nanmin(max_array)
    cm_max = np.nanmax(max_array)
    im1 = ax1.imshow(max_array[0], cmap='viridis', vmin=cm_min, vmax=cm_max)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("Maximum LAI")
    ax1.set_axis_off()

    im2 = ax2.imshow(max_array[1], cmap='viridis', vmin=cm_min, vmax=cm_max)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Maximum LAI Adjusted")
    ax2.set_axis_off()

    obs_dates = [datetime.strptime(r['Date'], '%d/%m/%Y') for r in lai_stats if not int(r['interpolated'])]
    obs_lai_mean = np.array([float(r['LAI Mean']) for r in lai_stats if not int(r['interpolated'])])
    obs_lai_mean_adjusted = np.array([float(r['LAI Mean Adjusted']) for r in lai_stats if not int(r['interpolated'])])
    obs_lai_median = np.array([float(r['LAI Median']) for r in lai_stats if not int(r['interpolated'])])
    obs_lai_median_adjusted = np.array([float(r['LAI Median Adjusted']) for r in lai_stats if not int(r['interpolated'])])

    all_dates = [datetime.strptime(r['Date'], '%d/%m/%Y') for r in lai_stats]
    all_lai_mean = np.array([float(r['LAI Mean']) for r in lai_stats])
    all_lai_mean_adjusted = np.array([float(r['LAI Mean Adjusted']) for r in lai_stats])
    all_lai_median = np.array([float(r['LAI Median']) for r in lai_stats])
    all_lai_median_adjusted = np.array([float(r['LAI Median Adjusted']) for r in lai_stats])


    # Plot LAI mean
    ax3.scatter(obs_dates, obs_lai_mean, label="LAI MEAN", color='tab:orange', s=4, zorder=2)
    ax3.plot(all_dates, all_lai_mean, color='tab:orange', zorder=1)

    ax3.scatter(obs_dates, obs_lai_mean_adjusted, label="LAI MEAN Adjusted", color='tab:blue', s=4, zorder=2)
    ax3.plot(all_dates, all_lai_mean_adjusted, color='tab:blue', zorder=1)

    # Plot LAI median
    ax3.scatter(obs_dates, obs_lai_median, label="LAI MEDIAN", color='darkorange', s=4, zorder=2)
    ax3.plot(all_dates, all_lai_median, color='darkorange', zorder=1)

    ax3.scatter(obs_dates, obs_lai_median_adjusted, label="LAI MEDIAN Adjusted", color='royalblue', s=4, zorder=2)
    ax3.plot(all_dates, all_lai_median_adjusted, color='royalblue', zorder=1)

    ax3.legend()
    ax3.set_axisbelow(True)
    ax3.grid(linestyle='dashed')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('LAI')
    ax3.set_title('LAI Timeseries')

    fig.suptitle(suptitle)

    fig.savefig(out_filepath, dpi=300)


if __name__ == "__main__":
    main()