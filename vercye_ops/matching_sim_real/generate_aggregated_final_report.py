import os.path as op
import os
import click
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
from xhtml2pdf import pisa

import matplotlib.pyplot as plt

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def fill_report_template(yield_map_path, regions_summary, global_summary, start_date, end_date, aggregated_yield_map_preview_path, evaluation_results, roi_name):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Yield Report {roi_name}</title>
        <link href=\"https://gist.githubusercontent.com/chachra/4075119/raw/f08b301cac2c1563b26db92a6da14477874b2e14/bootstrap.css\" rel=\"stylesheet\">
        <style>
            @font-face {{
                font-family: Open Sans;
                src: url('https://github.com/edx/edx-fonts/raw/refs/heads/master/open-sans/fonts/Regular/OpenSans-Regular.ttf');
            }}
            body {{
                font-family: 'Open Sans', sans-serif;
                background-color: #f9f9f9;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .content-container {{
                max-width: 900px;
                margin: 0 auto;
                background: #ffffff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }}
            table {{
                margin-top: 20px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                display: block;
                margin: 20px auto;
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class=\"content-container\">
            <h1><strong>Yield Report {roi_name}</strong></h1>

            <p><strong>Date Range:</strong> {start_date.date()} to {end_date.date()}</br>
            <strong>Total Yield:</strong> {global_summary['total_yield_production_ton']:.3f} t</br>
            <strong>Weighted Mean Yield:</strong> {int(global_summary['mean_yield_kg'])} kg/ha</br>
            <strong>Total Cropland Area:</strong> {global_summary['total_area_ha']:.2f} ha</p>

            <img src="{aggregated_yield_map_preview_path}" alt="Yield per Pixel Map"> 
            
            <hr>
            <h4 style='-pdf-keep-with-next: true; '>Yield Per Region</h4>

            <img src="{yield_map_path}" alt="Yield per Region Map">

            <table class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Total Yield (t)</th>
                        <th>Mean Yield (kg/ha)</th>
                        <th>Median Yield (kg/ha)</th>
                        {'<th>Reported Yield (t)</th>' if 'reported_yield_kg' in regions_summary.columns else ''}
                        {'<th>Reported Mean Yield (kg/ha)</th>' if 'reported_mean_yield_kg_ha' in regions_summary.columns else ''}
                        <th>Cropland Area (ha)</th>
                    </tr>
                </thead>
                <tbody>
    """

    for _, row in regions_summary.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row['region']}</td>
                        <td>{row['total_yield_production_ton']}</td>
                        <td>{int(row['mean_yield_kg_ha'])}</td>
                        <td>{int(row['median_yield_kg_ha'])}</td>
                        {f'<td>{(row["reported_yield_kg"] / 1000):.2f}</td>' if 'reported_yield_kg' in row else ''}
                        {f'<td>{int(row["reported_mean_yield_kg_ha"])}</td>' if 'reported_mean_yield_kg_ha' in row else ''}
                        <td>{row['total_area_ha']:.2f}</td>
                    </tr>
        """

    

    html_content += f"""
                    </tbody>
                </table>

                <hr>
                {f'''
                <h4  style='-pdf-keep-with-next: true; '>Evaluation Metrics</h4>
                <p><strong>Mean Error:</strong> {int(evaluation_results['mean_err_kg_ha'].iloc[0])} kg/ha</br>
                <strong>Median Error:</strong> {int(evaluation_results['median_err_kg_ha'].iloc[0])} kg/ha</br>
                <strong>RMSE:</strong> {int(evaluation_results['rmse_kg_ha'].iloc[0])} kg/ha</br>
                <strong>Relative RMSE:</strong> {evaluation_results['rrmse'].iloc[0]:.2f} %</br>
                <strong>R2:</strong> {evaluation_results['r2'].iloc[0]:.3f}</p>
                ''' if evaluation_results is not None else ''}

            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        """

    return html_content


def compute_global_summary(regions_summary):
    total_area_ha = regions_summary['total_area_ha'].sum()
    total_yield_production_ton = regions_summary['total_yield_production_ton'].sum()
    total_yield_production_kg =  regions_summary['total_yield_production_kg'].sum()
    mean_yield_kg = total_yield_production_kg / total_area_ha

    return {'total_area_ha': total_area_ha, 'total_yield_production_ton': total_yield_production_ton, 'mean_yield_kg': mean_yield_kg}


def get_regions_geometry_paths(regions_dir):
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
        legend_kwds={'label': "Mean Yield (kg/ha)"},
        ax=ax
    )

    # Add region labels
    for idx, row in merged.iterrows():
        centroid = row['geometry'].centroid
        ax.text(
            x=centroid.x,
            y=centroid.y,
            s=f"{row['region']} \n {int(row['mean_yield_kg_ha'])}",  # Use the region name as the label
            horizontalalignment='center',
            fontsize=7,
            color='black',
            #weight='bold'
        )

    ax.set_title("Crop Productivity Overview - Mean Yield (kg/ha) per Region", fontsize=16)
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


def convert_geotiff_to_png_with_legend(geotiff_path, output_png_path, width=3840, height=2160):
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
    
    # Replace no-data values with NaN for visualization
    data = np.where(data == src.nodata, np.nan, data)
    
    # Normalize the data for color mapping
    norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    colormap = plt.cm.viridis
    
    colored_data = colormap(norm(data))
    rgb_image = (colored_data[:, :, :3] * 255).astype(np.uint8)
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=900, constrained_layout=True)
    
    ax.imshow(rgb_image, aspect='auto')
    ax.axis('off')

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=colormap),
        ax=ax,
        orientation='horizontal',
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label('Yield kg/ha')

    ax.set_title("Crop Productivity Pixel-Level - Yield in kg/ha", fontsize=16)
    fig.savefig(output_png_path, format="PNG", bbox_inches='tight', dpi=600)
    plt.close(fig)
    return output_png_path


def generate_final_report(regions_dir, start_date, end_date, aggregated_yield_map_path, aggregated_yield_estimates_path, evaluation_results_path, gt_yield_path, roi_name):
    aggregated_data_fpath = op.join(regions_dir, aggregated_yield_estimates_path)
    regions_summary = pd.read_csv(aggregated_data_fpath)

    if gt_yield_path:
        gt = pd.read_csv(gt_yield_path)
        regions_summary = regions_summary.merge(
            gt[['reported_yield_kg', 'reported_mean_yield_kg_ha', 'region']],
            how='left',
            on='region'
        )

    global_summary = compute_global_summary(regions_summary)

    logger.info('Loading and combining region geometries...')
    regions_geometry_paths = get_regions_geometry_paths(regions_dir)
    combined_geojson = combine_geojsons(regions_geometry_paths)

    logger.info('Creating vector yield map...')
    yield_map = create_map(regions_summary, combined_geojson)
    yield_map_fname = 'yield_map.png'
    yield_map_path = op.join(regions_dir, yield_map_fname)
    yield_map.figure.savefig(yield_map_path, dpi=600)

    logger.info('Creating downsampled yieldmap preview...')
    aggregated_yield_map_preview_fname = 'aggregated_yield_map_preview.png'
    aggregated_yield_map_preview_path = op.join(regions_dir, aggregated_yield_map_preview_fname)
    convert_geotiff_to_png_with_legend(aggregated_yield_map_path, aggregated_yield_map_preview_path)

    if evaluation_results_path:
        evaluation_results = pd.read_csv(evaluation_results_path)
    else:
        evaluation_results = None

    return fill_report_template(yield_map_path, 
                                regions_summary,
                                global_summary,
                                start_date,
                                end_date,
                                aggregated_yield_map_preview_path,
                                evaluation_results,
                                roi_name)


def save_report(report, out_fpath):
    with open(out_fpath, "w+b") as result_file:
        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            report,
            dest=result_file,
        )

        if pisa_status.err:
            print("An error occured!")


@click.command()
@click.option('--regions_dir', required=True, type=click.Path(exists=True), help='Path to the directory containing region subdirectories.')
@click.option('--out_fpath', required=True, type=click.Path(), help='Path to save the aggregated final report (has to be .pdf).')
@click.option('--start_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="Start date of considered timespan in YYYY-MM-DD format.")
@click.option('--end_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=True, help="End date of considered timespan in YYYY-MM-DD format.")
@click.option('--aggregated_yield_map_path', required=True, type=click.Path(), help='Path to the combined yield map of all regions.')
@click.option('--aggregated_yield_estimates_path', required=True, type=click.Path(), help='Path to the combined yield estimates (.csv) of all regions.')
@click.option('--evaluation_results_path', required=False, type=click.Path(), help='Path to the evaluation results csv.', default=None)
@click.option('--val_fpath', required=False, type=click.Path(), help='Filepath to the csv containing the validation data per region.')
@click.option('--roi_name', required=True, type=click.STRING, help='Name of the primary region of interest.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(regions_dir, out_fpath, start_date, end_date, aggregated_yield_map_path, aggregated_yield_estimates_path, evaluation_results_path, val_fpath, roi_name, verbose):
    """Generate an aggregated final report from multiple regions."""

    if verbose:
        logger.setLevel('INFO')

    logger.info(f'Generating final report for regions in: {regions_dir}')
    report = generate_final_report(regions_dir, start_date, end_date, aggregated_yield_map_path, aggregated_yield_estimates_path, evaluation_results_path, val_fpath, roi_name)
    logger.info(f'Saving report to: {out_fpath}')
    save_report(report, out_fpath)


if __name__ == '__main__':
    cli()