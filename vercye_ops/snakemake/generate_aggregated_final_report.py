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
from datetime import datetime
import glob
from logging import StreamHandler


from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def compute_global_summary(regions_summary):
    total_area_ha = regions_summary['total_area_ha'].sum()
    total_yield_production_ton = regions_summary['total_yield_production_ton'].sum()
    total_yield_production_kg =  regions_summary['total_yield_production_kg'].sum()
    mean_yield_kg = total_yield_production_kg / total_area_ha

    # Add total reported production and mean reported yield if all regions have reported data
    if 'reported_mean_yield_kg_ha' in regions_summary.columns:
        reported_regions_data = regions_summary[~regions_summary['reported_mean_yield_kg_ha'].isna()]

        if regions_summary['reported_mean_yield_kg_ha'].isna().any():
            logger.warning('Some regions have NaN reported yield. Not reporting global aggregated reference.')
            logger.warning(f"Regions with nan reported yield: {regions_summary[regions_summary['reported_mean_yield_kg_ha'].isna()]['region'].values}")

            reported_total_production_ton = None
            mean_reported_yield_kg = None
        else:
            cropmask_areas_ha = reported_regions_data['total_area_ha'].sum()
            reported_total_production_kg = (regions_summary['reported_mean_yield_kg_ha'] * regions_summary['total_area_ha']).sum()
            reported_total_production_ton = reported_total_production_kg / 1000
            mean_reported_yield_kg = reported_total_production_kg / cropmask_areas_ha
    elif 'reported_production_kg' in regions_summary.columns:
        reported_regions_data = regions_summary[~regions_summary['reported_production_kg'].isna()]

        if regions_summary['reported_production_kg'].isna().any():
            logger.warning('Some regions have NaN reported yield. Not reporting.')
            logger.warning(f"Regions with nan reported yield: {regions_summary[regions_summary['reported_production_kg'].isna()]['region'].values}")

            reported_total_production_ton = None
            mean_reported_yield_kg = None
        else:
            reported_total_production_kg = reported_regions_data['reported_production_kg'].sum()
            reported_total_production_ton = reported_total_production_kg / 1000
            cropmap_areas_ha = reported_regions_data['total_area_ha'].sum()
            mean_reported_yield_kg = reported_total_production_kg / cropmap_areas_ha
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

    # Add region labels with dynamic contrast adjustment if not too many regions
    if len(merged) < 70:
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


def combine_geojsons(regions_geometry_paths, admin_column_name):
    geo_dfs = []
    crs = None

    # Case 1: This is the simulation level geojsons - no aggregation admin column needed
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

    combined_gdf = gpd.GeoDataFrame(pd.concat(geo_dfs, ignore_index=True), crs=geo_dfs[0].crs)

    # case 2: This is the aggregated geojsons - we need to merge them based on the admin column
    if admin_column_name is not None:
        combined_gdf = combined_gdf.dissolve(by=admin_column_name).reset_index()
        combined_gdf['region'] = combined_gdf[admin_column_name]

    return combined_gdf


def convert_geotiff_to_png_with_legend(geotiff_path, output_png_path, width=3840, height=2160):
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
    
    # Replace no-data values with NaN for visualization
    data = np.where(data == src.nodata, np.nan, data)
    
    # Normalize the data for color mapping
    # Using percentiles to avoid outliers affecting the color mapping. Outliers will have the same color.
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    norm = Normalize(vmin=vmin, vmax=vmax)
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
    fig.savefig(output_png_path, format="PNG", bbox_inches='tight', dpi=450)
    plt.close(fig)
    return output_png_path


def build_section_params(section_name, aggregated_yield_estimates_path, groundtruth_path, evaluation_results_path, regions_dir, admin_column_name):
    logger.info(f'Building section parameters for {section_name}...')
    regions_summary = pd.read_csv(aggregated_yield_estimates_path)

    if groundtruth_path is not None:
        logger.info('Loading groundtruth data...')
        gt = pd.read_csv(groundtruth_path)
        cols = ['region']
        if 'reported_production_kg' in gt.columns:
            cols.append('reported_production_kg')
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
    logger.info(f'Combined geojson shape: {combined_geojson.shape}. Num regions: {len(regions_geometry_paths)}')

    logger.info('Creating vector yield map...')
    yield_map = create_map(regions_summary, combined_geojson)
    yield_map_fname = f'yield_map_{section_name}.png'
    yield_map_path = op.join(regions_dir, yield_map_fname)
    yield_map.figure.savefig(yield_map_path, dpi=600)

    if evaluation_results_path is not None:
        evaluation_results = pd.read_csv(evaluation_results_path)
        scatter_plot_path = evaluation_results_path.replace('.csv', '.png')
    else:
        evaluation_results = None
        scatter_plot_path = None

    section_params = {
        'section_name': section_name,
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


def fill_section_template(section_name, regions_summary, scatter_plot_path, evaluation_results, vector_yield_map_path, crop_name, primary_suffix):
    crop_name = crop_name.lower().capitalize()
    section_name = section_name if section_name != primary_suffix else 'Simulation'
    section_name = section_name.lower().capitalize()
    html_content = f"""
        <hr>
        <h2 style='-pdf-keep-with-next: true;'>{section_name.lower().capitalize()}-level Evaluation</h2>
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
                            <strong>Mean Error:</strong> {int(evaluation_results['mean_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Error:</strong> {int(evaluation_results['median_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Mean Absolute Error:</strong> {int(evaluation_results['mean_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>Median Absolute Error:</strong> {int(evaluation_results['median_abs_err_kg_ha'].iloc[0])} kg/ha<br>
                            <strong>RMSE:</strong> {int(evaluation_results['rmse_kg_ha'].iloc[0])} kg/ha<br>
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

    html_content +=  "</div>" if evaluation_results is not None else ""

    html_content +=  f'<img src="{vector_yield_map_path}" class="margin-img" alt="Estimated Yield Map">'

    html_content += f"""
        <table class="table table-striped table-bordered">
            <thead>
                <tr>
                    <th>Region</th>
                    <th>Estimated Mean Yield (kg/ha)</th>
                    <th>Estimated Median Yield (kg/ha)</th>
                    {'<th>Reported Mean Yield (kg/ha)</th>' if 'reported_mean_yield_kg_ha' in regions_summary.columns else ''}
                    <th>Estimated Total Production (t)</th>
                    {'<th>Reported Total Production (t)</th>' if 'reported_production_kg' in regions_summary.columns else ''}
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
                        {f'<td>{"{:,.2f}".format((row["reported_production_kg"] / 1000)) if not pd.isna(row["reported_production_kg"]) else "N/A"}</td>' if 'reported_production_kg' in row else ''}
                        {f'<td>{int(row["mean_err_kg_ha"]) if not pd.isna(row["mean_err_kg_ha"]) else "N/A"}</td>' if 'mean_err_kg_ha' in row else ''}
                        <td>{"{:,.2f}".format(row['total_area_ha'])}</td>
                    </tr>
        """

    html_content += f"""
                    </tbody>
                </table>
            </div>
    """

    return html_content

def generate_final_report(sections, global_summary, metadata, met_config, aggregated_yield_map_preview_path):
    study_id = metadata['study_id']
    description = metadata['description']
    title = metadata['title'].capitalize()
    original_lai_shp = metadata['original_lai_shp']
    original_regions_shp = metadata['original_regions_shp']
    crop_name = metadata['crop_name'].lower().capitalize()

    start_date = metadata['start_date']
    end_date = metadata['end_date']
    cutoff_date = met_config['cutoff_date']

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
            <h1><strong>Yield Report {title}</strong></h1>

            <p>
            <strong>Study ID:</strong> {study_id}</br>
            <strong>Crop:</strong> {crop_name} </br>
            <strong>Date Range (YY-MM-DD):</strong> {start_date.date()} to {end_date.date()}</br>
            <strong>Met-data Cutoff Date:</strong> {cutoff_date.date()}</br>
            <strong>Source of Meteorological Data:</strong> {met_config['met_source']}. <strong>Precipiation Data:</strong> {met_config['precipitation_source']}.<br/> <strong>Precipitation Aggregation:</strong> {met_config['precipitation_agg_method']}. <strong>Fallback Precipitation:</strong> {met_config['fallback_precipitation']}</br>
            <strong>Description:</strong> {description}</br>
            <strong>LAI Shapefile:</strong> {original_lai_shp}</br>
            <strong>Regions Shapefile:</strong> {original_regions_shp}</br></br>
            <strong>Estimated Yield (Weighted Mean):</strong> {int(global_summary['mean_yield_kg'])} kg/ha</br>
            {f"<strong>Reported Yield (Weighted Mean):</strong> {int(global_summary['mean_reported_yield_kg'])} kg/ha</br>" if global_summary['mean_reported_yield_kg'] is not None else ''}
            <strong>Estimated Total Production:</strong> {'{:,.3f}'.format(global_summary['total_yield_production_ton'])} t</br>
            {f"<strong>Reference Total Production:</strong> {'{:,.3f}'.format(global_summary['reported_total_production_ton'])} t</br>" if global_summary['reported_total_production_ton'] is not None else ''}
            <strong>Total {crop_name} Area:</strong> {'{:,.2f}'.format(global_summary['total_area_ha'])} ha</p>

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
    temp_log_handler.setLevel('INFO')
    logger.addHandler(temp_log_handler)

    out_fpath = output['report_fpath']

    regions_dir = params['regions_dir']
    pixel_level_yieldmap_path = input['pixel_level_yieldmap']
    aggregationsuffix_admincol = params['suffix_admincols'] # dict of aggregation level suffixes and admin column names
    primary_suffix = params['primary_suffix'] # should be just primary - is the simulation level suffix
    year = wildcards['year']

    metadata = {
        'study_id': params['study_id'],
        'title': params['title'],
        'description': params['description'],
        'original_lai_shp': params['original_lai_shp'],
        'original_regions_shp': params['original_simregions_shp'],
        'crop_name': params['crop_name'],
        'start_date': datetime.strptime(params['start_date'], "%Y-%m-%d"),
        'end_date': datetime.strptime(params['end_date'], "%Y-%m-%d"),
    }

    met_config = {
        'cutoff_date': datetime.strptime(params['cutoff_date'], "%Y-%m-%d"),
        'met_source': params['met_source'],
        'precipitation_source': params['precipitation_source'],
        'precipitation_agg_method':  params['precipitation_agg_method'],
        'fallback_precipitation': params['fallback_precipitation']
    }

    if primary_suffix not in aggregationsuffix_admincol:
        aggregationsuffix_admincol[primary_suffix] = None

    sections = {}
    global_summary = None
    for suffix, admin_column_name in aggregationsuffix_admincol.items():
        # Collect predictions
        # The aggregated yield estimates files have the additional suffix of sudy id year, timepoint so we use wildcards to match
        aggregated_yield_estimates_patttern = os.path.join(regions_dir, f'agg_yield_estimates_{suffix}_*.csv')
        matching_files = glob.glob(aggregated_yield_estimates_patttern)
        aggregated_yield_estimates_path = matching_files[0] if matching_files else None

        if aggregated_yield_estimates_path is None:
           logger.warning(f"Aggregated yield estimates file not found: {aggregated_yield_estimates_patttern}. Skipping.")
           continue

        # Collect groundtruth and evaluation results
        gt_dir = os.path.os.path.dirname(regions_dir)
        groundtruth_path = os.path.join(gt_dir, f'groundtruth_{suffix}-{year}.csv')
        if not os.path.exists(groundtruth_path):
            logger.warning(f"Groundtruth file not found: {groundtruth_path}. Skipping.")
            groundtruth_path = None

        evaluation_results_path = os.path.join(regions_dir, f'evaluation_{suffix}.csv')
        if not os.path.exists(evaluation_results_path):
            logger.warning(f"Evaluation results file not found: {evaluation_results_path}. Skipping.")
            evaluation_results_path = None
        
        section = build_section_params(
            section_name=suffix,
            aggregated_yield_estimates_path=aggregated_yield_estimates_path,
            groundtruth_path=groundtruth_path,
            evaluation_results_path=evaluation_results_path,
            regions_dir=regions_dir,
            admin_column_name=admin_column_name,
        )

        sections[suffix] = fill_section_template(**section, crop_name=metadata['crop_name'], primary_suffix=primary_suffix)

        if suffix == primary_suffix:
            global_summary = compute_global_summary(section['regions_summary'])

    logger.info('Creating downsampled yieldmap preview...')
    aggregated_yield_map_preview_fname = 'aggregated_yield_map_preview.png'
    aggregated_yield_map_preview_path = op.join(regions_dir, aggregated_yield_map_preview_fname)
    convert_geotiff_to_png_with_legend(pixel_level_yieldmap_path, aggregated_yield_map_preview_path)
    
    logger.info(f'Generating final report for regions in: {regions_dir}')
    report = generate_final_report(sections, global_summary, metadata, met_config, aggregated_yield_map_preview_path)
    logger.info(f'Saving report to: {out_fpath}')
    save_report(report, out_fpath)
    logger.info('Report generation completed.')