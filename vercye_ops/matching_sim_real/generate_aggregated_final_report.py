import os
import os.path as op

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


def fill_report_template(yield_map_path, regions_summary, global_summary,
                         start_date, end_date, cutoff_date, aggregated_yield_map_preview_path,
                         evaluation_results, roi_name, crop_name, met_config, scatter_plot_path=None):
    crop_name = crop_name.lower().capitalize()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bootstrap_css_path = os.path.join(BASE_DIR, 'assets', 'bootstrap.css')
    bootstrap_js_path = os.path.join(BASE_DIR, 'assets', 'bootstrap.bundle.min.js')
    font_path = os.path.join(BASE_DIR, 'assets', 'OpenSans-Regular.ttf')
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Yield Report {roi_name} - {crop_name}</title>
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
            }}
            th {{
                background-color: #f2f2f2;
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
            <h1><strong>Yield Report {roi_name} - {crop_name}</strong></h1>

            <p><strong>Date Range (YY-MM-DD):</strong> {start_date.date()} to {end_date.date()}</br>
            <strong>Cutoff Date:</strong> {cutoff_date.date()}</br>
            <strong>Source of Meteorological Data:</strong> {met_config['met_source']}. <strong>Precipiation Data:</strong> {met_config['precipitation_source']}. <strong>Precipitation Aggregation:</strong> {met_config['precipitation_agg_method']}. <strong>Fallback Precipitation:</strong> {met_config['fallback_precipitation']}</br>
            <strong>Estimated Yield (Weighted Mean):</strong> {int(global_summary['mean_yield_kg'])} kg/ha</br>
            {f"<strong>Reported Yield (Weighted Mean):</strong> {int(global_summary['mean_reported_yield_kg'])} kg/ha</br>" if global_summary['mean_reported_yield_kg'] is not None else ''}
            <strong>Estimated Total Production:</strong> {'{:,.3f}'.format(global_summary['total_yield_production_ton'])} t</br>
            {f"<strong>Reference Total Production:</strong> {'{:,.3f}'.format(global_summary['reported_total_production_ton'])} t</br>" if global_summary['reported_total_production_ton'] is not None else ''}
            <strong>Total {crop_name} Area:</strong> {'{:,.2f}'.format(global_summary['total_area_ha'])} ha</p>


            {f'''
            <hr>
            <div class="evaluation-container">
                <div class="evaluation-text">
                    <h4>Evaluation Metrics</h4>
                    <p>Note: The evaluation metrics are only computed for those regions where ground truth (reference) data is available (See table below).<br>
                    <strong>Number of Regions Evaluated:</strong> {evaluation_results['n_regions'].iloc[0]}</br>
                    <strong>Mean Error:</strong> {int(evaluation_results['mean_err_kg_ha'].iloc[0])} kg/ha</br>
                    <strong>Median Error:</strong> {int(evaluation_results['median_err_kg_ha'].iloc[0])} kg/ha</br>
                    <strong>Mean Absolute Error:</strong> {int(evaluation_results['mean_abs_err_kg_ha'].iloc[0])} kg/ha</br>
                    <strong>Median Absolute Error:</strong> {int(evaluation_results['median_abs_err_kg_ha'].iloc[0])} kg/ha</br>
                    <strong>RMSE:</strong> {int(evaluation_results['rmse_kg_ha'].iloc[0])} kg/ha</br>
                    <strong>Relative RMSE:</strong> {evaluation_results['rrmse'].iloc[0]:.2f} %</br>
                    <strong>R2 (Coefficient of Determination):</strong> {evaluation_results['r2_scikit'].iloc[0]:.3f}</br>
                    <strong>R2 (Pearson Correlation Coefficient):</strong> {evaluation_results['r2_rsq_excel'].iloc[0]:.3f}</br>
                    <strong>R2 Best Fit (Coefficient of Determination):</strong> {evaluation_results['r2_scikit_bestfit'].iloc[0]:.3f}</p>
                </div>
            ''' if evaluation_results is not None else ''}

            {f'''
                <div class="evaluation-image">
                    <img src="{scatter_plot_path}" alt="Scatter Plot">
                </div>
            ''' if scatter_plot_path else ''}

            {'</div>' if evaluation_results is not None else ''}

            <img src="{aggregated_yield_map_preview_path}" class="margin-img" alt="Estimated Yield per Pixel Map"> 
            
            <hr>
            <h4 style='-pdf-keep-with-next: true; '>Yield Per Region</h4>

            <img src="{yield_map_path}" class="margin-img" alt="Estimated Yield per Region Map">

            <table class="table table-striped table-bordered">
                <thead>
                    <tr>
                        <th>Region</th>
                        <th>Estimated Mean Yield (kg/ha)</th>
                        <th>Estimated Median Yield (kg/ha)</th>
                        {'<th>Reported Mean Yield (kg/ha)</th>' if 'reported_mean_yield_kg_ha' in regions_summary.columns else ''}
                        <th>Estimated Total Production (t)</th>
                        {'<th>Reported Total Production (t)</th>' if 'reported_yield_kg' in regions_summary.columns else ''}
                        {'<th>Estimation Error (kg/ha)</th>' if evaluation_results is not None else ''}
                        <th>{crop_name} Area (ha)</th>
                    </tr>
                </thead>
                <tbody>
    """

    for _, row in regions_summary.iterrows():
        html_content += f"""
                    <tr>
                        <td>{row['region']}</td>
                        <td>{int(row['mean_yield_kg_ha'])}</td>
                        <td>{int(row['median_yield_kg_ha'])}</td>
                        {f'<td>{int(row["reported_mean_yield_kg_ha"]) if not pd.isna(row["reported_mean_yield_kg_ha"]) else "N/A" }</td>' if 'reported_mean_yield_kg_ha' in row else ''}
                        <td>{'{:,}'.format(row['total_yield_production_ton'])}</td>
                        {f'<td>{"{:,.2f}".format((row["reported_yield_kg"] / 1000)) if not pd.isna(row["reported_yield_kg"]) else "N/A"}</td>' if 'reported_yield_kg' in row else ''}
                        {f'<td>{int(row["mean_err_kg_ha"]) if not pd.isna(row["mean_err_kg_ha"]) else "N/A"}</td>' if 'mean_err_kg_ha' in row else ''}
                        <td>{"{:,.2f}".format(row['total_area_ha'])}</td>
                    </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>

            <script src=\"{bootstrap_js_path}\"></script>
        </body>
        </html>
        """

    return html_content


def compute_global_summary(regions_summary):
    total_area_ha = regions_summary['total_area_ha'].sum()
    total_yield_production_ton = regions_summary['total_yield_production_ton'].sum()
    total_yield_production_kg =  regions_summary['total_yield_production_kg'].sum()
    mean_yield_kg = total_yield_production_kg / total_area_ha

    # need to get those entries where reported_mean_yield_kg_ha is not NaN
    if 'reported_yield_kg' in regions_summary.columns:
        reported_regions_data = regions_summary[~regions_summary['reported_yield_kg'].isna()]
        reported_areas_ha = reported_regions_data['total_area_ha'].sum()
        reported_total_production_kg = reported_regions_data['reported_yield_kg'].sum()
        reported_total_production_ton = reported_total_production_kg / 1000
        mean_reported_yield_kg = reported_total_production_kg / reported_areas_ha
    elif 'reported_mean_yield_kg_ha' in regions_summary.columns:
        reported_regions_data = regions_summary[~regions_summary['reported_mean_yield_kg_ha'].isna()]
        reported_areas_ha = reported_regions_data['total_area_ha'].sum()
        reported_total_production_kg = (regions_summary['reported_mean_yield_kg_ha'] * regions_summary['total_area_ha']).sum()
        reported_total_production_ton = reported_total_production_kg / 1000
        mean_reported_yield_kg = reported_total_production_kg / reported_areas_ha
    else:
        reported_total_production_ton = None
        mean_reported_yield_kg = None

    return {'total_area_ha': total_area_ha,
            'total_yield_production_ton': total_yield_production_ton,
            'mean_yield_kg': mean_yield_kg,
            'reported_total_production_ton': reported_total_production_ton,
            'mean_reported_yield_kg': mean_reported_yield_kg}


def get_regions_geometry_paths(regions_dir):
    return {region: op.join(regions_dir, region, f'{region}.geojson')
            for region in os.listdir(regions_dir) 
            if op.isdir(op.join(regions_dir, region))}


def get_contrasting_text_color(rgb):
    """Returns black or white based on perceived brightness of the background color."""
    brightness = np.dot(rgb[:3], [0.299, 0.587, 0.114])  # Standard luminance formula
    return 'black' if brightness > 0.5 else 'white'

def create_map(regions_summary, combined_geojson):
    # Merge geometry with summary data
    combined_geojson['region'] = combined_geojson['region'].astype(str)
    regions_summary['region'] = regions_summary['region'].astype(str)
    merged = combined_geojson.merge(regions_summary, left_on='region', right_on='region')

    # Define colormap and normalization
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=merged['mean_yield_kg_ha'].min(), vmax=merged['mean_yield_kg_ha'].max())

    # Plot map
    fig, ax = plt.subplots(figsize=(12, 8))
    merged.plot(
        column='mean_yield_kg_ha',
        cmap=cmap,
        legend=True,
        legend_kwds={'label': "Estimated Mean Yield (kg/ha)"},
        ax=ax
    )

    # Add region labels with dynamic contrast adjustment
    for idx, row in merged.iterrows():
        centroid = row['geometry'].centroid
        color_rgb = cmap(norm(row['mean_yield_kg_ha']))
        text_color = get_contrasting_text_color(color_rgb)
        
        ax.text(
            x=centroid.x,
            y=centroid.y,
            s=f"{row['region']} \n {int(row['mean_yield_kg_ha'])}",  
            horizontalalignment='center',
            fontsize=7,
            weight='bold',                                                                                                            
            color=text_color,
        )

    ax.set_title("Crop Productivity Overview - Estimated Mean Yield (kg/ha) per Region", fontsize=16)
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
    colored_data[np.isnan(data)] = [1, 1, 1, 1]
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

    ax.set_title("Crop Productivity Pixel-Level - Estimated Yield in kg/ha", fontsize=16)
    fig.savefig(output_png_path, format="PNG", bbox_inches='tight', dpi=600)
    plt.close(fig)
    return output_png_path


def build_section_params(section_name, aggregated_yield_estimates_path, groundtruth_path, evaluation_results_path, regions_dir, admin_column_name):
    regions_summary = pd.read_csv(aggregated_yield_estimates_path)

    if groundtruth_path:
        gt = pd.read_csv(groundtruth_path)
        cols = ['region']
        if 'reported_yield_kg' in gt.columns:
            cols.append('reported_yield_kg')
        if 'reported_mean_yield_kg_ha' in gt.columns:
            cols.append('reported_mean_yield_kg_ha')

        regions_summary = regions_summary.merge(
            gt[cols],
            how='left',
            on='region'
        )

        if 'reported_mean_yield_kg_ha' in gt.columns:
            regions_summary['mean_err_kg_ha'] = regions_summary['reported_mean_yield_kg_ha'] - regions_summary['mean_yield_kg_ha']

    logger.info('Loading and combining region geometries...')
    regions_geometry_paths = get_regions_geometry_paths(regions_dir)
    combined_geojson = combine_geojsons(regions_geometry_paths, admin_column_name)

    logger.info('Creating vector yield map...')
    yield_map = create_map(regions_summary, combined_geojson)
    yield_map_fname = f'yield_map_{section_name}.png'
    yield_map_path = op.join(regions_dir, yield_map_fname)
    yield_map.figure.savefig(yield_map_path, dpi=600)

    if evaluation_results_path:
        evaluation_results = pd.read_csv(evaluation_results_path)
        scatter_plot_path = evaluation_results_path.replace('.csv', '.png')
    else:
        evaluation_results = None
        scatter_plot_path = None

    section_params = {
        'regions_summary': regions_summary,
        'vector_yield_map_path': yield_map_path,
        'scatter_plot_path': scatter_plot_path,
        'evaluation_results': evaluation_results,
    }

    return section_params


def save_report(report, out_fpath):
    with open(out_fpath, "w+b") as result_file:
        # convert HTML to PDF
        pisa_status = pisa.CreatePDF(
            report,
            dest=result_file,
        )

        if pisa_status.err:
            print("An error occured!")


def create_final_report(input, output, params, log, wildcards):
    """Generate an aggregated final report from multiple regions."""

    if params['verbose']:
        logger.setLevel('INFO')

    out_fpath = output['out_fpath']

    regions_dir = input['regions_dir']
    pixel_level_yieldmap_path = input['aggregated_yield_map_path']
    results_basedir = params['results_basedir']
    aggregation_suffixes = params['aggregation_suffixes']
    primary_suffix = params['primary_suffix'] # should be just primary

    metadata = {
        'roi_name': params['roi_name'],
        'crop_name': params['crop_name'],
        'start_date': params['start_date'],
        'end_date': params['end_date'],
    }

    met_config = {
        'cutoff_date': params['cutoff_date'],
        'met_source': params['met_source'],
        'precipitation_source': params['precipitation_source'],
        'precipitation_agg_method':  params['precipitation_agg_method'],
        'fallback_precipitation': params['fallback_precipitation']
    }

    sections = {}
    for suffix in aggregation_suffixes:
        # Collect predictions
        aggregated_yield_estimates_path = os.path.join(results_basedir, f'agg_yield_estimates_{suffix}.csv')

        if not os.path.exists(aggregated_yield_estimates_path):
           logger.warning(f"Aggregated yield estimates file not found: {aggregated_yield_estimates_path}. Skipping.")
           continue

        # Collect groundtruth and evaluation results
        groundtruth_path = os.path.join(results_basedir, f'groundtruth_{suffix}.csv')
        if not os.path.exists(groundtruth_path):
            logger.warning(f"Groundtruth file not found: {groundtruth_path}. Skipping.")
            continue

        evaluation_results_path = os.path.join(results_basedir, f'evaluation_{suffix}.csv')
        if not os.path.exists(evaluation_results_path):
            logger.warning(f"Evaluation results file not found: {evaluation_results_path}. Skipping.")
            continue
        

        section = build_section_params(
            section_name=suffix,
            aggregated_yield_estimates_path=aggregated_yield_estimates_path,
            groundtruth_path=groundtruth_path,
            evaluation_results_path=evaluation_results_path,
            regions_dir=regions_dir,
            admin_column_name=admin_column_name,
        )

        sections[suffix] = fill_section_template(section)

        if suffix == primary_suffix:
            global_summary = compute_global_summary(section['regions_summary'])

    logger.info('Creating downsampled yieldmap preview...')
    aggregated_yield_map_preview_fname = 'aggregated_yield_map_preview.png'
    aggregated_yield_map_preview_path = op.join(regions_dir, aggregated_yield_map_preview_fname)
    convert_geotiff_to_png_with_legend(pixel_level_yieldmap_path, aggregated_yield_map_preview_path)
    
    logger.info(f'Generating final report for regions in: {regions_dir}')
    report = generate_final_report(sections, metadata, met_config, aggregated_yield_map_preview_path)
    logger.info(f'Saving report to: {out_fpath}')
    save_report(report, out_fpath)
    logger.info('Report generation completed.')