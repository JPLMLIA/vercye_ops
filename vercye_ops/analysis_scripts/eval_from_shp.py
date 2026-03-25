import os

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract

from vercye_ops.evaluation.evaluate_yield_estimates import compute_metrics, create_scatter_plot, save_scatter_plot

shapefile = "/gpfs/data1/cmongp2/vercye/data/yieldstudies/ukraine_best_config_03_11_25/shapefile/UKR_adm1.shp"
yield_column = "reported_mean_yield_kg_ha"
year_column = "year"
yield_conversion_factor = 1  # t/ha to kg/ha
out_dir = "/gpfs/data1/cmongp2/vercye/data/yieldstudies/ukraine_best_config_03_11_25/exps"
os.makedirs(out_dir, exist_ok=True)

years_rasters = {
    "2022": "/gpfs/data1/cmongp2/vercye/data/yieldstudies/ukraine_best_config_03_11_25/ukraine_best_config_03_11_25/2022/T-0/aggregated_yield_map_ukraine_best_config_03_1_2022_T-0.tif",
}

refdata_2022 = "/gpfs/data1/cmongp2/vercye/data/yieldstudies/ukraine_best_config_03_11_25/reference_data/referencedata_oblast-2022.csv"

all_gdfs = {}

for year, raster in years_rasters.items():
    print(year)
    gdf = gpd.read_file(shapefile)

    # merge reported data
    reported_gdf = gpd.read_file(refdata_2022)
    gdf = gdf.merge(reported_gdf, left_on="NAME_1", right_on="region")
    gdf[yield_column] = gdf[yield_column].astype(float)

    # gdf = gdf[gdf[year_column] == year].copy()
    gdf["reported_yield"] = gdf[yield_column] * yield_conversion_factor

    # Exactextract statistics
    with rasterio.open(raster) as src:
        print(src.nodata)
        stats = exact_extract(
            src,
            gdf,
            ["mean", "median"],
            include_cols=[
                "reported_yield",
                "NAME_1",
            ],
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
