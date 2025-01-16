import os.path as op
import os
import click
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def fill_report_template(yield_map_path, regions_summary, global_summary):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yield Report</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Yield Report</h1>
        <img src="{yield_map_path}" alt="Regions Map" style="width:100%; max-width:600px;">
        <h2>Summary</h2>
        <p><strong>Total Yield (t):</strong> {global_summary['total_yield_production_ton']}</p>
        <p><strong>Weighted Mean Yield (kg/ha):</strong> {global_summary['weighted_mean_yield_kg']}</p>
        <p><strong>Total Cropland Area (ha):</strong> {global_summary['total_area_ha']}</p>

        <h2>Regions Data</h2>
        <table>
            <tr>
                <th>Region</th>
                <th>Total Yield (t)</th>
                <th>Mean Yield (kg/ha)</th>
                <th>Cropland Area (ha)</th>
            </tr>
    """

    for _, row in regions_summary.iterrows():
        html_content += f"""
            <tr>
                <td>{row['region']}</td>
                <td>{row['total_yield_production_ton']}</td>
                <td>{row['mean_yield_kg_ha']}</td>
                <td>{row['total_area_ha']}</td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    return html_content


def compute_global_summary(regions_summary):
    total_area_ha = regions_summary['total_area_ha'].sum()
    total_yield_production_ton = regions_summary['total_yield_production_ton'].sum()
    total_yield_production_kg =  regions_summary['total_yield_production_kg'].sum()
    weighted_mean_yield_kg = total_yield_production_kg / total_area_ha

    return {'total_area_ha': total_area_ha, 'total_yield_production_ton': total_yield_production_ton, 'weighted_mean_yield_kg': weighted_mean_yield_kg}


def get_regions_geometry_paths(regions_dir):
    for region in os.listdir(regions_dir):
        if op.isdir(op.join(regions_dir, region)):
            op.join(regions_dir, region, f'{region}.geojson')

    return {region: op.join(regions_dir, region, f'{region}.geojson')
            for region in os.listdir(regions_dir) 
            if op.isdir(op.join(regions_dir, region))}


def create_map(regions_summary, combined_geojson):
    # Merge geometry with summary data
    merged = combined_geojson.merge(regions_summary, left_on='region', right_on='region')

    # Plot map
    fig, ax = plt.subplots(figsize=(12, 8))
    merged.plot(
        column='mean_yield_kg_ha',
        cmap='viridis',
        legend=True,
        legend_kwds={'label': "Mean Yield per Region"},
        ax=ax
    )

    # Add region labels
    for idx, row in merged.iterrows():
        centroid = row['geometry'].centroid
        ax.text(
            x=centroid.x,
            y=centroid.y,
            s=row['region'],  # Use the region name as the label
            horizontalalignment='center',
            fontsize=8,
            color='black',
        )

    ax.set_title("Yield Map with Mean Yield Per Region and Region Labels", fontsize=16)
    ax.axis('off')
    return ax


def combine_geojsons(regions_geometry_paths):
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
            gdf['region'] = region
            geo_dfs.append(gdf)
    return gpd.GeoDataFrame(pd.concat(geo_dfs, ignore_index=True), crs=geo_dfs[0].crs)


def save_yield_map(yield_map, yield_map_path):
    yield_map.figure.savefig(yield_map_path)


def generate_final_report(regions_dir):
    aggregated_data_fpath = op.join(regions_dir, 'aggregated_yield_estimates.csv')
    regions_summary = pd.read_csv(aggregated_data_fpath)
    global_summary = compute_global_summary(regions_summary)

    regions_geometry_paths = get_regions_geometry_paths(regions_dir)
    combined_geojson = combine_geojsons(regions_geometry_paths)

    yield_map = create_map(regions_summary, combined_geojson)
    yield_map_path = op.join(regions_dir, 'yield_map.png')
    save_yield_map(yield_map, yield_map_path)

    return fill_report_template(yield_map_path, regions_summary, global_summary)


def save_report(report, out_fpath):
    with open(out_fpath, 'w') as f:
        f.write(report)


@click.command()
@click.option('--regions_dir', required=True, type=click.Path(exists=True), help='Path to the directory containing region subdirectories.')
@click.option('--out_fpath', required=True, type=click.Path(), help='Path to save the aggregated final report.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(regions_dir, out_fpath, verbose):
    """Generate an aggregated final report from multiple regions."""

    if verbose:
        logger.setLevel('INFO')

    report = generate_final_report(regions_dir)
    save_report(report, out_fpath)


if __name__ == '__main__':
    cli()