import os

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract

from vercye_ops.evaluation.evaluate_yield_estimates import compute_metrics, create_scatter_plot, save_scatter_plot

shapefile = "/gpfs/data1/cmongp2/vercye/experiemnts/rayons_ref/Rayons_yield_WinterWheat-26-8-2025.shp"
yield_column = "yield_t_ha"
year_column = "year"
yield_conversion_factor = 1000  # t/ha to kg/ha
out_dir = "/gpfs/data1/cmongp2/vercye/experiemnts/rayons_eval_ukr_v7"
os.makedirs(out_dir, exist_ok=True)

years_rasters = {
    "2019": "/gpfs/data1/cmongp2/sawahnr/data/ukraine/vercye_runs/Ukraine_National_V7/2019/T-0/aggregated_yield_map_ukr_v7_2019_T-0.tif",
    "2020": "/gpfs/data1/cmongp2/sawahnr/data/ukraine/vercye_runs/Ukraine_National_V7/2020/T-0/aggregated_yield_map_ukr_v7_2020_T-0.tif",
    "2021": "/gpfs/data1/cmongp2/sawahnr/data/ukraine/vercye_runs/Ukraine_National_V7/2021/T-0/aggregated_yield_map_ukr_v7_2021_T-0.tif",
    "2022": "/gpfs/data1/cmongp2/sawahnr/data/ukraine/vercye_runs/Ukraine_National_V7/2022/T-0/aggregated_yield_map_ukr_v7_2022_T-0.tif",
}

all_gdfs = {}

for year, raster in years_rasters.items():
    print(year)
    gdf = gpd.read_file(shapefile)
    gdf = gdf[gdf[year_column] == year].copy()
    gdf["reported_yield"] = gdf[yield_column] * yield_conversion_factor

    # Exactextract statistics
    with rasterio.open(raster) as src:
        print(src.nodata)
        exit()
        stats = exact_extract(
            src,
            gdf,
            ["mean", "median"],
            include_cols=["reported_yield", "NAME_1", "NAME_2", "year"],
            output="pandas",
            include_geom=True,
        )
        print(stats)

    # Compute metrics
    metrics_median = compute_metrics(preds=stats["median"], obs=stats["reported_yield"])
    metrics_mean = compute_metrics(preds=stats["mean"], obs=stats["reported_yield"])

    print(f"{year}: MEAN METRICS: ", metrics_mean)
    print(f"{year}: MEDIAN METRICS: ", metrics_median)

    # Scatterplot
    scatter_plot = create_scatter_plot(preds=stats["mean"], obs=stats["reported_yield"])
    out_plot_fpath = os.path.join(out_dir, f"{year}_plot.png")
    save_scatter_plot(scatter_plot, out_plot_fpath)

    all_gdfs[year] = stats

# Compute metrics and plot for all years together
all_df = pd.concat([gdf.assign(year=year) for year, gdf in all_gdfs.items()], ignore_index=True)
metrics_median = compute_metrics(preds=all_df["median"], obs=all_df["reported_yield"])
metrics_mean = compute_metrics(preds=all_df["mean"], obs=all_df["reported_yield"])

print("All years: MEAN METRICS: ", metrics_mean)
print("All years: MEDIAN METRICS: ", metrics_median)

scatter_plot = create_scatter_plot(preds=all_df["mean"], obs=all_df["reported_yield"], obs_years=all_df["year"])
out_plot_fpath = os.path.join(out_dir, "allyears_plot.png")
save_scatter_plot(scatter_plot, out_plot_fpath)

merged_gdf = gpd.GeoDataFrame(all_df)
merged_gdf.to_file(os.path.join(out_dir, "all_years_preds_obs_raoins.geojson"))
