import glob
import os
import os.path as op
from datetime import datetime
from logging import StreamHandler

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize
from xhtml2pdf import pisa

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()

logger.setLevel("INFO")


def compute_global_summary(regions_summary):
    total_area_ha = regions_summary["total_area_ha"].sum()
    total_production_ton = regions_summary["total_production_ton"].sum()
    total_production_kg = regions_summary["total_production_kg"].sum()
    mean_yield_kg = total_production_kg / total_area_ha

    # APSIM (pure simulation) totals when the APSIM-mosaic columns are present
    apsim_total_production_ton = None
    apsim_total_production_kg = None
    apsim_mean_yield_kg = None
    if (
        "total_production_kg_apsim" in regions_summary.columns
        and "total_production_ton_apsim" in regions_summary.columns
    ):
        apsim_total_production_kg = regions_summary["total_production_kg_apsim"].sum()
        apsim_total_production_ton = regions_summary["total_production_ton_apsim"].sum()
        if total_area_ha > 0:
            apsim_mean_yield_kg = apsim_total_production_kg / total_area_ha

    num_total_region = len(regions_summary)

    # Add total reported production and mean reported yield if all regions have reported data
    if "reported_mean_yield_kg_ha" in regions_summary.columns:
        reported_regions_data = regions_summary[~regions_summary["reported_mean_yield_kg_ha"].isna()]

        # Sometimes we don't have the reported data for every region.
        if regions_summary["reported_mean_yield_kg_ha"].isna().any():
            logger.warning("Some regions have NaN reported yield.")
            logger.warning(
                f"Regions with nan reported yield: {regions_summary[regions_summary['reported_mean_yield_kg_ha'].isna()]['region'].values}"
            )

        num_regions_with_yield_referencedata = len(reported_regions_data)
        cropmask_areas_ha = reported_regions_data["total_area_ha"].sum()
        weighted_sum_rep_prod_kg = (
            regions_summary["reported_mean_yield_kg_ha"] * regions_summary["total_area_ha"]
        ).sum()
        mean_reported_yield_kg = weighted_sum_rep_prod_kg / cropmask_areas_ha
    else:
        mean_reported_yield_kg = None
        num_regions_with_yield_referencedata = 0

    if "reported_production_kg" in regions_summary.columns:
        # In case we do not have the reported mean yield, we can still compute the total production and mean yield
        reported_regions_data = regions_summary[~regions_summary["reported_production_kg"].isna()]

        if regions_summary["reported_production_kg"].isna().any():
            logger.warning("Some regions have NaN reported yield. Not reporting.")
            logger.warning(
                f"Regions with nan reported yield: {regions_summary[regions_summary['reported_production_kg'].isna()]['region'].values}"
            )

        reported_total_production_kg = reported_regions_data["reported_production_kg"].sum()
        reported_total_production_ton = reported_total_production_kg / 1000
        num_regions_with_production_referencedata = len(reported_regions_data)
    else:
        reported_total_production_ton = None
        num_regions_with_production_referencedata = None

    return {
        "total_area_ha": total_area_ha,
        "total_production_ton": total_production_ton,
        "mean_yield_kg": mean_yield_kg,
        "reported_total_production_ton": reported_total_production_ton,
        "mean_reported_yield_kg": mean_reported_yield_kg,
        "num_total_region": num_total_region,
        "num_regions_with_yield_referencedata": num_regions_with_yield_referencedata,
        "num_regions_with_production_referencedata": num_regions_with_production_referencedata,
        "apsim_total_production_ton": apsim_total_production_ton,
        "apsim_mean_yield_kg": apsim_mean_yield_kg,
    }


def get_regions_geometry_paths(regions_dir):
    return {
        region: op.join(regions_dir, region, f"{region}.geojson")
        for region in os.listdir(regions_dir)
        if op.isdir(op.join(regions_dir, region))
    }


def get_contrasting_text_color(rgb):
    """Returns black or white based on perceived brightness of the background color."""
    brightness = np.dot(rgb[:3], [0.299, 0.587, 0.114])  # Standard luminance formula
    return "black" if brightness > 0.5 else "white"


def _place_region_labels(ax, merged, cmap, norm, fontsize=7, max_regions=300):
    """Label each region, but only where the text actually fits inside its polygon.

    The previous behaviour was all-or-nothing (label every region, or none above a
    count threshold), which left dense clusters of small geometries unreadable.
    matplotlib has no built-in decluttering, so measure the rendered text extent
    against the polygon's own extent in display space and degrade gracefully:
    full label -> value only -> no label.

    Placement uses representative_point() rather than centroid: for concave or
    multipart geometries the centroid can fall outside the polygon entirely.
    """
    if len(merged) > max_regions:
        logger.info("Skipping region labels: %d regions exceeds the %d cap.", len(merged), max_regions)
        return 0, len(merged)

    fig = ax.figure
    fig.canvas.draw()  # a renderer is required before text extents can be measured
    renderer = fig.canvas.get_renderer()

    placed = skipped = 0
    for _, row in merged.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        point = geom.representative_point()
        minx, miny, maxx, maxy = geom.bounds
        (x0, y0), (x1, y1) = ax.transData.transform([(minx, miny), (maxx, maxy)])
        poly_w, poly_h = abs(x1 - x0), abs(y1 - y0)

        text_color = get_contrasting_text_color(cmap(norm(row["mean_yield_kg_ha"])))
        value = safe_int(row["mean_yield_kg_ha"])

        for candidate in (f"{row['region']}\n{value}", f"{value}"):
            label = ax.text(
                point.x,
                point.y,
                candidate,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=fontsize,
                weight="bold",
                color=text_color,
            )
            bbox = label.get_window_extent(renderer=renderer)
            if bbox.width <= poly_w and bbox.height <= poly_h:
                placed += 1
                break
            label.remove()
        else:
            skipped += 1

    logger.info("Region labels: %d placed, %d omitted (too small at this scale).", placed, skipped)
    return placed, skipped


def create_map(regions_summary, combined_geojson):
    # Merge geometry with summary data
    combined_geojson["region"] = combined_geojson["region"].astype(str)
    regions_summary["region"] = regions_summary["region"].astype(str)
    merged = combined_geojson.merge(regions_summary, left_on="region", right_on="region")

    if merged.empty or merged["mean_yield_kg_ha"].dropna().empty:
        logger.warning("No matching regions between geometry and summary data - returning empty map")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("No data available for this aggregation level", fontsize=16)
        ax.axis("off")
        return ax

    yield_values = merged["mean_yield_kg_ha"]
    vmin = np.nanpercentile(yield_values, 2)
    vmax = np.nanpercentile(yield_values, 98)

    # Define colormap and normalization
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot map
    fig, ax = plt.subplots(figsize=(12, 8))
    merged.plot(
        column="mean_yield_kg_ha",
        cmap=cmap,
        legend=True,
        legend_kwds={"label": "Estimated Mean Yield (kg/ha)"},
        ax=ax,
    )

    # Add region labels, dropping any that cannot fit inside their own polygon.
    _place_region_labels(ax, merged, cmap, norm)

    ax.set_title("Crop Productivity Overview - Estimated Mean Yield (kg/ha) per Region", fontsize=16)
    ax.axis("off")
    return ax


def load_admin_shapefile(shapefile_path, name_column):
    """Load an aggregation-level shapefile and use its geometries directly for the report map."""
    gdf = gpd.read_file(shapefile_path)
    gdf[name_column] = gdf[name_column].astype(str)
    gdf = gdf.drop_duplicates(subset=[name_column]).reset_index(drop=True)
    gdf["region"] = gdf[name_column]
    return gdf


def combine_geojsons(regions_geometry_paths):
    """Combine individual simulation-region GeoJSONs into a single GeoDataFrame."""
    geo_dfs = []
    crs = None

    for region, path in regions_geometry_paths.items():
        if op.exists(path):
            gdf = gpd.read_file(path)
            if crs is None:
                crs = gdf.crs
            else:
                if crs != gdf.crs:
                    raise Exception(f"CRS mismatch between regions: {crs} != {gdf.crs}")
            gdf["region"] = region
            geo_dfs.append(gdf)

    combined_gdf = gpd.GeoDataFrame(pd.concat(geo_dfs, ignore_index=True), crs=geo_dfs[0].crs)
    return combined_gdf


def convert_geotiff_to_png_with_legend(geotiff_path, output_png_path, width=3840, height=2160):
    with rasterio.open(geotiff_path) as src:
        # Downsample large rasters on read to avoid OOM - target ~4000px on longest side
        max_dim = 4000
        scale = min(max_dim / src.height, max_dim / src.width, 1.0)
        out_height = max(1, int(src.height * scale))
        out_width = max(1, int(src.width * scale))

        data = src.read(1, out_shape=(out_height, out_width))
        nodata = src.nodata

    # Replace no-data values with NaN for visualization
    data = np.where(data == nodata, np.nan, data)

    # Normalize the data for color mapping
    # Using percentiles to avoid outliers affecting the color mapping. Outliers will have the same color.
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.cm.viridis

    colored_data = colormap(norm(data))
    colored_data[np.isnan(data)] = [0.7, 0.7, 0.7, 1]
    rgb_image = (colored_data[:, :, :3] * 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300, constrained_layout=True)

    ax.imshow(rgb_image, aspect="equal")
    ax.axis("off")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label("Yield kg/ha")

    ax.set_title("Crop Productivity Pixel-Level - Estimated Yield in kg/ha", fontsize=16)
    fig.savefig(output_png_path, format="PNG", bbox_inches="tight", dpi=400)
    plt.close(fig)
    return output_png_path


LAI_PLOT_COLUMN_PREFERENCE = [
    "LAI Median Adjusted",
    "LAI Median",
    "LAI Mean Adjusted",
    "LAI Mean",
]


def create_level_lai_plot(regions_dir, section_name):
    """Render the observed LAI timeseries for one aggregation level.

    Reads agg_lai_timeseries_{level}_*.csv (written by
    aggregate_lai_timeseries_per_level) and draws one thin line per region plus a
    bold level-mean, so a level with 47 counties stays readable. Returns the PNG
    path, or None when there is no timeseries for this level (e.g. the primary
    simulation level, which has no aggregation shapefile).
    """
    pattern = op.join(regions_dir, f"agg_lai_timeseries_{section_name}_*.csv")
    prefix = f"agg_lai_timeseries_{section_name}_"
    matches = [f for f in glob.glob(pattern) if op.basename(f).startswith(prefix)]
    if not matches:
        logger.info("No aggregated LAI timeseries found for level '%s'; skipping its LAI plot.", section_name)
        return None

    df = pd.read_csv(matches[0])
    column = next((c for c in LAI_PLOT_COLUMN_PREFERENCE if c in df.columns), None)
    if column is None or df.empty:
        logger.warning("LAI timeseries for level '%s' has no usable LAI column or no rows.", section_name)
        return None

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", column]).sort_values("Date")
    if df.empty:
        logger.warning("LAI timeseries for level '%s' has no valid rows after parsing.", section_name)
        return None

    regions = list(dict.fromkeys(df["region"]))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for region in regions:
        sub = df[df["region"] == region]
        ax.plot(sub["Date"], sub[column], linewidth=0.9, alpha=0.45, label=region)

    level_mean = df.groupby("Date")[column].mean()
    ax.plot(level_mean.index, level_mean.values, linewidth=2.4, color="black", label="Level mean")

    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.set_title(f"Observed LAI timeseries - {section_name}")
    ax.grid(alpha=0.25)
    # A legend only helps while the entries are individually distinguishable.
    if len(regions) <= 8:
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(
            0.99,
            0.97,
            f"{len(regions)} regions",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="0.35",
        )
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = op.join(regions_dir, f"lai_timeseries_{section_name}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote LAI timeseries plot for level '%s' (%d regions).", section_name, len(regions))
    return out_path


def build_section_params(
    section_name,
    aggregated_yield_estimates_path,
    referencedata_path,
    evaluation_results_path,
    regions_dir,
    admin_column_name,
    admin_shapefile_path=None,
):
    logger.info(f"Building section parameters for {section_name}...")
    regions_summary = pd.read_csv(aggregated_yield_estimates_path)
    reference_yield_agg = None

    if referencedata_path is not None:
        logger.info("Loading referencedata_ data...")
        gt = pd.read_csv(referencedata_path)
        cols = ["region"]
        if "reported_production_kg" in gt.columns:
            cols.append("reported_production_kg")
        if "reported_mean_yield_kg_ha" in gt.columns:
            cols.append("reported_mean_yield_kg_ha")

        if "reported_mean_yield_kg_ha" in regions_summary.columns:
            regions_summary.drop(["reported_mean_yield_kg_ha"], axis="columns", inplace=True)

        regions_summary = regions_summary.merge(gt[cols], how="left", on="region")

        if "reported_mean_yield_kg_ha" in gt.columns:
            regions_summary["mean_err_kg_ha"] = (
                regions_summary["reported_mean_yield_kg_ha"] - regions_summary["mean_yield_kg_ha"]
            )
            print(regions_summary["mean_err_kg_ha"])

            # Compute aggregated reference yield over all regions with reference data
            regions_summary["area_weighted_prod"] = (
                regions_summary["reported_mean_yield_kg_ha"] * regions_summary["total_area_ha"]
            )
            valid_regions = regions_summary[regions_summary["area_weighted_prod"].notna()]
            total_area_ha = valid_regions["total_area_ha"].sum()
            total_production_kg = regions_summary["area_weighted_prod"].sum()
            reference_yield_agg = total_production_kg / total_area_ha if total_area_ha > 0 else None

    logger.info("Loading and combining region geometries...")
    if admin_shapefile_path is not None and admin_column_name is not None:
        # Use the aggregation-level shapefile directly - it has the correct admin boundaries
        combined_geojson = load_admin_shapefile(admin_shapefile_path, admin_column_name)
    else:
        # Primary level: combine individual simulation-region GeoJSONs
        regions_geometry_paths = get_regions_geometry_paths(regions_dir)
        combined_geojson = combine_geojsons(regions_geometry_paths)
    logger.info(f"Combined geojson shape: {combined_geojson.shape}. Num regions: {len(combined_geojson)}")

    logger.info("Creating vector yield map...")
    yield_map = create_map(regions_summary, combined_geojson)
    yield_map_fname = f"yield_map_{section_name}.png"
    yield_map_path = op.join(regions_dir, yield_map_fname)
    yield_map.figure.savefig(yield_map_path, dpi=400)

    if evaluation_results_path is not None:
        evaluation_results = pd.read_csv(evaluation_results_path)
        scatter_plot_path = evaluation_results_path.replace(".csv", ".png")
    else:
        evaluation_results = None
        scatter_plot_path = None

    # APSIM-only eval (no LAI-pixel-conversion) is written alongside as ..._no-pixel-conversion.csv/.png
    apsim_evaluation_results = None
    apsim_scatter_plot_path = None
    if evaluation_results_path is not None:
        apsim_eval_csv = evaluation_results_path.replace(".csv", "_no-pixel-conversion.csv")
        apsim_eval_png = evaluation_results_path.replace(".csv", "_no-pixel-conversion.png")
        if os.path.exists(apsim_eval_csv):
            apsim_evaluation_results = pd.read_csv(apsim_eval_csv)
        if os.path.exists(apsim_eval_png):
            apsim_scatter_plot_path = apsim_eval_png

    section_params = {
        "section_name": section_name,
        "regions_summary": regions_summary,
        "vector_yield_map_path": yield_map_path,
        "scatter_plot_path": scatter_plot_path,
        "evaluation_results": evaluation_results,
        "apsim_evaluation_results": apsim_evaluation_results,
        "apsim_scatter_plot_path": apsim_scatter_plot_path,
        "reference_yield_agg": reference_yield_agg,
        "lai_timeseries_plot_path": create_level_lai_plot(regions_dir, section_name),
    }

    return section_params


def insert_word_breaks(s, interval=4):
    return "<wbr>".join([s[i : i + interval] for i in range(0, len(s), interval)])


def save_report(report, out_fpath):
    with open(out_fpath, "w+b") as result_file:
        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            report,
            dest=result_file,
        )

        if pisa_status.err:
            print("An error occured!")


def safe_int(value, default="N/A"):
    return default if pd.isna(value) else int(value)


def fill_section_template(
    section_name,
    regions_summary,
    scatter_plot_path,
    evaluation_results,
    vector_yield_map_path,
    crop_name,
    primary_suffix,
    reference_yield_agg,
    apsim_evaluation_results=None,
    apsim_scatter_plot_path=None,
    lai_timeseries_plot_path=None,
):
    crop_name = crop_name.lower().capitalize()
    section_name = section_name if section_name != primary_suffix else "Simulation"
    section_name = section_name.lower().capitalize()
    html_content = f"""
        <hr>
        <h2 style='-pdf-keep-with-next: true;'>{section_name.lower().capitalize()}-level Evaluation</h2>
    """

    if reference_yield_agg is not None:
        html_content += f"""
            <p style='-pdf-keep-with-next: true;'>
                <strong>Aggregated Reference Yield (kg/ha):</strong> {safe_int(reference_yield_agg)} kg/ha</br>
                Aggregated from all regions with reference data in this section, so possibly incomplete for complete study area.</br>
            </p>
        """

    if evaluation_results is not None:
        html_content += f"""
            <table width="100%" border="0" cellspacing="0" cellpadding="5">
                <tr>
                    <!-- Left column: Evaluation metrics -->
                    <td width="40%" style="vertical-align: top; font-size: 0.9em; padding-top: 40px">
                        <p>
                            <strong>Note:</strong> The evaluation metrics are only computed for those regions where ground truth (reference) data is available (See table below).<br>
                            <strong>Number of Regions Evaluated:</strong> {evaluation_results['n_regions'].iloc[0]}<br>
                            <strong>Mape: </strong> {evaluation_results['mape'].iloc[0] if 'mape' in evaluation_results else '-'} <br>
                            <strong>Mean Error:</strong> {safe_int(evaluation_results['mean_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Error:</strong> {safe_int(evaluation_results['median_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Mean Absolute Error:</strong> {safe_int(evaluation_results['mean_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Absolute Error:</strong> {safe_int(evaluation_results['median_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>RMSE:</strong> {safe_int(evaluation_results['rmse_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Relative RMSE:</strong> {evaluation_results['rrmse'].iloc[0]:.2f} %<br>
                            <strong>R2 (Coefficient of Determination):</strong> {evaluation_results['r2_scikit'].iloc[0]:.3f}<br>
                            <strong>R2 (Pearson Correlation Coefficient):</strong> {evaluation_results['r2_rsq_excel'].iloc[0]:.3f}<br>
                            <strong>R2 Best Fit (Coefficient of Determination):</strong> {evaluation_results['r2_scikit_bestfit'].iloc[0]:.3f}
                        </p>
                    </td>

                    <!-- Right column: Scatter plot -->
                    <td width="60%" style="vertical-align: top; text-align: center;">
                        {f'<img src="{scatter_plot_path}" alt="Scatter Plot" style="max-width: 100%; height: auto;">' if scatter_plot_path else ''}
                    </td>
                </tr>
            </table>
        """

    html_content += "</div>" if evaluation_results is not None else ""

    if apsim_evaluation_results is not None:
        html_content += f"""
            <h3 style='-pdf-keep-with-next: true; margin-top: 18px;'>APSIM-only Evaluation</h3>
            <p style='font-size: 0.85em; color: #555;'>
              Reference vs APSIM-matched yield (no LAI-based pixel conversion). The same reference data is
              compared against the area-weighted APSIM yield per region.
            </p>
            <table width="100%" border="0" cellspacing="0" cellpadding="5">
                <tr>
                    <td width="40%" style="vertical-align: top; font-size: 0.9em; padding-top: 40px">
                        <p>
                            <strong>Number of Regions Evaluated:</strong> {apsim_evaluation_results['n_regions'].iloc[0]}<br>
                            <strong>Mape: </strong> {apsim_evaluation_results['mape'].iloc[0] if 'mape' in apsim_evaluation_results else '-'} <br>
                            <strong>Mean Error:</strong> {safe_int(apsim_evaluation_results['mean_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Error:</strong> {safe_int(apsim_evaluation_results['median_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Mean Absolute Error:</strong> {safe_int(apsim_evaluation_results['mean_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Absolute Error:</strong> {safe_int(apsim_evaluation_results['median_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>RMSE:</strong> {safe_int(apsim_evaluation_results['rmse_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Relative RMSE:</strong> {apsim_evaluation_results['rrmse'].iloc[0]:.2f} %<br>
                            <strong>R2 (Coefficient of Determination):</strong> {apsim_evaluation_results['r2_scikit'].iloc[0]:.3f}<br>
                            <strong>R2 (Pearson Correlation Coefficient):</strong> {apsim_evaluation_results['r2_rsq_excel'].iloc[0]:.3f}<br>
                            <strong>R2 Best Fit (Coefficient of Determination):</strong> {apsim_evaluation_results['r2_scikit_bestfit'].iloc[0]:.3f}
                        </p>
                    </td>
                    <td width="60%" style="vertical-align: top; text-align: center;">
                        {f'<img src="{apsim_scatter_plot_path}" alt="APSIM-only Scatter Plot" style="max-width: 100%; height: auto;">' if apsim_scatter_plot_path else ''}
                    </td>
                </tr>
            </table>
        """

    html_content += f'<img src="{vector_yield_map_path}" class="margin-img" alt="Estimated Yield Map">'
    if lai_timeseries_plot_path:
        html_content += (
            "<h3 style='-pdf-keep-with-next: true;'>Observed LAI timeseries</h3>"
            f'<img src="{lai_timeseries_plot_path}" class="margin-img" alt="Observed LAI timeseries">'
        )

    has_apsim_cols = "mean_yield_kg_ha_apsim" in regions_summary.columns

    html_content += f"""
        <table class="table table-striped table-bordered">
            <thead style='-pdf-keep-with-next: true;'>
                <tr>
                    <th>Region</th>
                    <th>Estimated Mean Yield (kg/ha)</th>
                    <th>Estimated Median Yield (kg/ha)</th>
                    {'<th>APSIM Mean Yield (kg/ha)</th>' if has_apsim_cols else ''}
                    {'<th>Reported Mean Yield (kg/ha)</th>' if 'reported_mean_yield_kg_ha' in regions_summary.columns else ''}
                    <th>Estimated Total Production (t)</th>
                    {'<th>APSIM Total Production (t)</th>' if 'total_production_ton_apsim' in regions_summary.columns else ''}
                    {'<th>Reported Total Production (t)</th>' if 'reported_production_kg' in regions_summary.columns else ''}
                    {'<th>Estimation Error (kg/ha)</th>' if 'mean_err_kg_ha' in regions_summary.columns else ''}
                    <th>{crop_name} Area (ha)</th>
                </tr>
            </thead>
            <tbody>
    """

    # Sort by errors to allow to identify problematic areas easier
    if "mean_err_kg_ha" in regions_summary.columns:
        regions_summary.sort_values(by="mean_err_kg_ha", ascending=False, inplace=True)

    for _, row in regions_summary.iterrows():
        html_content += f"""
                    <tr>
                        <td>{insert_word_breaks(row['region'])}</td>
                        <td>{safe_int(row['mean_yield_kg_ha'])}</td>
                        <td>{safe_int(row['median_yield_kg_ha'])}</td>
                        {f'<td>{safe_int(row["mean_yield_kg_ha_apsim"])}</td>' if has_apsim_cols else ''}
                        {f'<td>{safe_int(row["reported_mean_yield_kg_ha"]) if not pd.isna(row["reported_mean_yield_kg_ha"]) else "N/A"}</td>' if 'reported_mean_yield_kg_ha' in row else ''}
                        <td>{'{:,.0f}'.format(row['total_production_ton'])}</td>
                        {f'<td>{"{:,.0f}".format(row["total_production_ton_apsim"]) if not pd.isna(row["total_production_ton_apsim"]) else "N/A"}</td>' if 'total_production_ton_apsim' in row else ''}
                        {f'<td>{"{:,.0f}".format((row["reported_production_kg"] / 1000)) if not pd.isna(row["reported_production_kg"]) else "N/A"}</td>' if 'reported_production_kg' in row else ''}
                        {f'<td>{safe_int(row["mean_err_kg_ha"]) if not pd.isna(row["mean_err_kg_ha"]) else "N/A"}</td>' if 'mean_err_kg_ha' in row else ''}
                        <td>{"{:,.0f}".format(row['total_area_ha'])}</td>
                    </tr>
        """

    html_content += """
                    </tbody>
                </table>
            </div>
    """

    return html_content


def generate_final_report(sections, global_summary, metadata, met_config, aggregated_yield_map_preview_path):
    study_id = metadata["study_id"]
    description = metadata["description"]
    title = metadata["title"].capitalize()
    lai_source = metadata["lai_source"]
    original_regions_shp = metadata["original_regions_shp"]
    crop_name = metadata["crop_name"].lower().capitalize()

    start_date = metadata["start_date"]
    end_date = metadata["end_date"]
    cutoff_date = met_config["cutoff_date"]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bootstrap_css_path = os.path.join(BASE_DIR, "assets", "bootstrap.css")
    bootstrap_js_path = os.path.join(BASE_DIR, "assets", "bootstrap.bundle.min.js")
    font_path = os.path.join(BASE_DIR, "assets", "OpenSans-Regular.ttf")

    num_regions = global_summary["num_total_region"]
    num_available_regions_yield = global_summary["num_regions_with_yield_referencedata"]
    num_available_regions_production = global_summary["num_regions_with_production_referencedata"]

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Yield Report {title} - {crop_name}</title>
        <link href=\"{bootstrap_css_path}\" rel=\"stylesheet\">
        <style>
            @font-face {{
                font-family: Open Sans;
                src: url('{font_path}');
            }}
            body {{
                font-family: 'Open Sans', sans-serif;
                font-size: 14px;
                background-color: #f9f9f9;

            }}
            h1 {{
                text-align: center;

            }}
            .content-container {{
                max-width: 900px;

                background: #ffffff;

                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            table {{
                margin-top: 20px;
                font-size: 11px;
                table-layout: fixed;
                width: 100%;
                border-collapse: collapse;
            }}
            th {{
                background-color: #f2f2f2;
            }}

            th, td {{
                white-space: normal !important;   /* allow wrapping */
                overflow-wrap: anywhere;          /* modern: break anywhere if needed */
                word-wrap: break-word;            /* legacy alias/fallback */
                word-break: break-word;           /* WebKit fallback */
                -ms-word-break: break-all;        /* old IE fallback */
                word-break: break-all;            /* last-resort for old engines/wkhtmltopdf */
            }}
            .margin-img {{
                display: block;
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}

            .evaluation-image img {{
                width: 360px; /* Makes sure the image takes full available width */
                height: auto; /* Maintains aspect ratio */
                object-fit: contain; /* Ensures the image doesn't get cropped */
            }}
        </style>
    </head>
    <body>
        <div class=\"content-container\">
            <h1><strong>Yield Report {title}</strong></h1>

            <p>
            <strong>Study ID:</strong> {study_id}</br>
            <strong>Crop:</strong> {crop_name} </br>
            <strong>Date Range (YY-MM-DD):</strong> {start_date.date()} to {end_date.date()}</br>
            <strong>Met-data Cutoff Date:</strong> {cutoff_date.date()}</br>
            <strong>Source of Meteorological Data:</strong> {met_config['met_source']}. <strong>Precipiation Data:</strong> {met_config['precipitation_source']}.<br/> <strong>Precipitation Aggregation:</strong> {met_config['precipitation_agg_method']}. <strong>Fallback Precipitation:</strong> {met_config['fallback_precipitation']}</br>
            <strong>Description:</strong> {description}</br>
            <strong>LAI Source:</strong> {lai_source}</br>
            <strong>Regions Shapefile:</strong> {original_regions_shp}</br></br>
            <strong>Estimated Yield (Weighted Mean):</strong> {safe_int(global_summary['mean_yield_kg'])} kg/ha</br>
            {f"<strong>APSIM Yield (Weighted Mean):</strong> {safe_int(global_summary['apsim_mean_yield_kg'])} kg/ha</br>" if global_summary.get('apsim_mean_yield_kg') is not None else ''}
            {f"<strong>Reported Yield (Weighted Mean):</strong> {safe_int(global_summary['mean_reported_yield_kg'])} kg/ha (from {num_available_regions_yield}/{num_regions} regions)</br>" if global_summary['mean_reported_yield_kg'] is not None else ''}
            <strong>Estimated Total Production:</strong> {'{:,.0f}'.format(global_summary['total_production_ton'])} t</br>
            {f"<strong>APSIM Total Production:</strong> {'{:,.0f}'.format(global_summary['apsim_total_production_ton'])} t</br>" if global_summary.get('apsim_total_production_ton') is not None else ''}
            {f"<strong>Reference Total Production:</strong> {'{:,.0f}'.format(global_summary['reported_total_production_ton'])} t (from {num_available_regions_production}/{num_regions} regions)</br>" if global_summary['reported_total_production_ton'] is not None else ''}
            <strong>Total {crop_name} Area:</strong> {'{:,.0f}'.format(global_summary['total_area_ha'])} ha</p>

            <img src="{aggregated_yield_map_preview_path}" class="margin-img" alt="Estimated Yield per Pixel Map">

    """

    for section_name, section in sections.items():
        html_content += section

    html_content += f"""
            </div>
            <script src=\"{bootstrap_js_path}\"></script>
        </body>
        </html>
    """

    return html_content


def create_final_report(input, output, params, log, wildcards):
    """Generate an aggregated final report from multiple regions."""

    temp_log_handler = StreamHandler(log)
    temp_log_handler.setLevel("INFO")
    logger.addHandler(temp_log_handler)

    out_fpath = output["report_fpath"]

    regions_dir = params["regions_dir"]
    pixel_level_yieldmap_path = input["pixel_level_yieldmap"]
    aggregationsuffix_admincol = params["suffix_admincols"]  # dict of aggregation level suffixes and admin column names
    suffix_shapefiles = params.get("suffix_shapefiles", {})  # dict of aggregation level suffixes to shapefile paths
    primary_suffix = params["primary_suffix"]  # should be just primary - is the simulation level suffix
    year = wildcards["year"]

    metadata = {
        "study_id": params["study_id"],
        "title": params["title"],
        "description": params["description"],
        "lai_source": params["lai_source"],
        "original_regions_shp": params["original_simregions_shp"],
        "crop_name": params["crop_name"],
        "start_date": datetime.strptime(params["start_date"], "%Y-%m-%d"),
        "end_date": datetime.strptime(params["end_date"], "%Y-%m-%d"),
    }

    met_config = {
        "cutoff_date": datetime.strptime(params["cutoff_date"], "%Y-%m-%d"),
        "met_source": params["met_source"],
        "precipitation_source": params["precipitation_source"],
        "precipitation_agg_method": params["precipitation_agg_method"],
        "fallback_precipitation": params["fallback_precipitation"],
    }

    if primary_suffix not in aggregationsuffix_admincol:
        aggregationsuffix_admincol[primary_suffix] = None

    sections = {}
    global_summary = None
    for suffix, admin_column_name in aggregationsuffix_admincol.items():
        # Collect predictions
        # The aggregated yield estimates files have the additional suffix of study id year, timepoint so we use wildcards to match
        # Pin the level name by requiring the study_id to follow it immediately: a bare
        # f"agg_yield_estimates_{suffix}_*.csv" glob also matches any level whose name
        # starts with this one (e.g. "ADM1" matching the "ADM1_ThreeCounties" file), and
        # taking matching_files[0] would then build this level's section from another
        # level's numbers with nothing raised.
        aggregated_yield_estimates_patttern = os.path.join(regions_dir, f"agg_yield_estimates_{suffix}_*.csv")
        _prefix = f"agg_yield_estimates_{suffix}_{metadata['study_id']}_"
        matching_files = [
            f for f in glob.glob(aggregated_yield_estimates_patttern) if os.path.basename(f).startswith(_prefix)
        ]
        if len(matching_files) > 1:
            raise ValueError(f"Multiple aggregated yield estimates files matched level '{suffix}': {matching_files}")
        aggregated_yield_estimates_path = matching_files[0] if matching_files else None

        if aggregated_yield_estimates_path is None:
            logger.warning(
                f"Aggregated yield estimates file not found: {aggregated_yield_estimates_patttern}. Skipping."
            )
            continue

        # Collect referencedata_ and evaluation results
        gt_dir = os.path.os.path.dirname(regions_dir)
        referencedata_path = os.path.join(gt_dir, f"referencedata_{suffix}-{year}.csv")
        if not os.path.exists(referencedata_path):
            logger.warning(f"referencedata file not found: {referencedata_path}. Skipping.")
            referencedata_path = None

        evaluation_results_path = os.path.join(regions_dir, f"evaluation_{suffix}.csv")
        if not os.path.exists(evaluation_results_path):
            logger.warning(f"Evaluation results file not found: {evaluation_results_path}. Skipping.")
            evaluation_results_path = None

        logger.info(f"Processing section: {suffix}")
        section = build_section_params(
            section_name=suffix,
            aggregated_yield_estimates_path=aggregated_yield_estimates_path,
            referencedata_path=referencedata_path,
            evaluation_results_path=evaluation_results_path,
            regions_dir=regions_dir,
            admin_column_name=admin_column_name,
            admin_shapefile_path=suffix_shapefiles.get(suffix),
        )

        sections[suffix] = fill_section_template(
            **section, crop_name=metadata["crop_name"], primary_suffix=primary_suffix
        )

        if suffix == primary_suffix:
            global_summary = compute_global_summary(section["regions_summary"])

    logger.info("Creating downsampled yieldmap preview...")
    aggregated_yield_map_preview_fname = "aggregated_yield_map_preview.png"
    aggregated_yield_map_preview_path = op.join(regions_dir, aggregated_yield_map_preview_fname)
    convert_geotiff_to_png_with_legend(pixel_level_yieldmap_path, aggregated_yield_map_preview_path)

    logger.info(f"Generating final report for regions in: {regions_dir}")
    report = generate_final_report(sections, global_summary, metadata, met_config, aggregated_yield_map_preview_path)
    save_report(report, out_fpath)
    logger.info("Report generation completed.")
