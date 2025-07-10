import os
import re
import shutil
import tempfile
from pathlib import Path

import click
import geopandas as gpd
from dotenv import dotenv_values
from ruamel.yaml import YAML

from vercye_ops.utils.convert_shapefile_to_geojson import \
    convert_shapefile_to_geojson

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def rel_path(*path_parts):
    return os.path.join(BASE_DIR, *path_parts)


def load_yaml_ruamel(filepath):
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    with open(filepath, "r") as f:
        return yaml_loader.load(f), yaml_loader


def prepare_study(config_path):
    config, _ = load_yaml_ruamel(config_path)

    shapefile_path = config["regions_shp_name"]
    admin_col = config["regions_shp_col"]
    filter_col = config.get("regions_shp_filter_col")
    filter_vals = config.get("regions_shp_filter_values", [])
    apsim_template_paths_filter_col = config.get("APSIM_TEMPLATE_PATHS_FILTER_COL_NAME")
    apsim_template_paths = config["APSIM_TEMPLATE_PATHS"]

    output_dir = str(Path(config_path).parent / Path(config_path).parent.name)
    if os.path.exists(output_dir):
        raise ValueError(f'A basedirectory already exists under {output_dir}.')
    os.makedirs(output_dir)

    snakefile_template_config_path = rel_path('examples/run_config_template.yaml')
    snakefile_config, ruamel_yaml = load_yaml_ruamel(snakefile_template_config_path)

    # Update snakemakefile with values from the template
    snakefile_config['years'] = config['years']
    snakefile_config['timepoints'] = config['timepoints']
    snakefile_config['apsim_params']['time_bounds'] = config['timepoints_config']
    snakefile_config['matching_params']['target_crs']  = config['target_crs']

    projection_crs = config['target_crs'].strip('\'"')
    geojsons_folder = tempfile.TemporaryDirectory()

    # Create individual geojsons of each shapefile entry
    convert_shapefile_to_geojson(
        shp_fpath=shapefile_path,
        admin_name_col=admin_col,
        projection_crs=projection_crs,
        output_head_dir=geojsons_folder.name
    )

    # Filter regions based on provided column + values
    # and map to APSIM files
    keep_regions = []
    regions_apsimfile = {}

    for f in sorted(os.listdir(geojsons_folder.name)):
        region_name = f
        geojson_folder_path = os.path.join(geojsons_folder.name, f)
        if not os.path.isdir(geojson_folder_path):
            continue

        gdf = gpd.read_file(os.path.join(geojson_folder_path, f + ".geojson"))

        if len(gdf) > 1:
            raise ValueError(f"Shapefile {shapefile_path} has more than one polygon.")

        if filter_col is not None and gdf[filter_col][0] not in filter_vals:
            continue

        if apsim_template_paths_filter_col:
            if apsim_template_paths_filter_col not in gdf.columns:
                raise ValueError(f"Column {apsim_template_paths_filter_col} not found in the shapefile.")
            key = gdf[apsim_template_paths_filter_col][0]
            regions_apsimfile[region_name] = apsim_template_paths[key]
        else:
            regions_apsimfile[region_name] = apsim_template_paths['all']

        keep_regions.append(region_name)


    # Copy all regions into each year/timepoint directory
    years = snakefile_config['years']
    timepoints = snakefile_config['timepoints']

    for region in keep_regions:
        region_file = os.path.join(geojsons_folder.name, region, f"{region}.geojson")
        for year in years:
            for tp in timepoints:
                folder = os.path.join(output_dir, str(year), str(tp), region)
                os.makedirs(folder, exist_ok=True)
                shutil.copy(region_file, folder)

    # Copy APSIM file into each year/timepoint/region directory and adapt the 
    # simulation start and end date for that year
    for year in years:
        for tp in timepoints:
            for region in keep_regions:
                start = snakefile_config['apsim_params']['time_bounds'][year][tp]['sim_start_date']
                end = snakefile_config['apsim_params']['time_bounds'][year][tp]['sim_end_date']
                template_path = regions_apsimfile[region]

                with open(template_path, "r", encoding="utf-8") as f:
                    data = f.read()

                data = re.sub(r'"Start":\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"',
                              f'"Start": "{start}T00:00:00"', data)
                data = re.sub(r'"End":\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"',
                              f'"End": "{end}T00:00:00"', data)

                out_path = os.path.join(output_dir, str(year), str(tp), region, f"{region}_template.apsimx")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(data)

    # Update config with the region metadata
    snakefile_config['regions'] = keep_regions
    snakefile_config['regions_shp_col'] = admin_col
    snakefile_config['regions_shp_filter_col'] = filter_col
    snakefile_config['regions_shp_filter_values'] = filter_vals
    snakefile_config['APSIM_TEMPLATE_PATHS'] = apsim_template_paths
    snakefile_config['regions_shp_name'] = shapefile_path
    snakefile_config['sim_study_head_dir'] = str(output_dir)

    # Fill in LAI section in run_config based on lai creation config 
    lai_config_path = str(Path(config_path).parent / 'lai_config.yaml')
    if os.path.exists(lai_config_path):
        print('imagery found')
        lai_config, _ = load_yaml_ruamel(lai_config_path)
        snakefile_config['lai_source'] = str(Path(lai_config['geojson_path']).name)
        snakefile_config['lai_params']['lai_dir'] =  os.path.join(lai_config['out_dir'], 'merged-lai')
        snakefile_config['lai_params']['lai_region'] = lai_config['region_out_prefix']
        snakefile_config['lai_params']['lai_resolution'] = lai_config['resolution']

    # Transfer LAI start-enddate configuration to run config
    lai_dict = {}
    for year in config['timepoints_config']:
        lai_dict[year] = {}
        for timepoint in config['timepoints_config'][year]:
            lai_dict[year][timepoint] = [
                config['timepoints_config'][year][timepoint]['lai_start_date'],
                config['timepoints_config'][year][timepoint]['lai_end_date']]

    snakefile_config['lai_params']['time_bounds'] = lai_dict

    # If env file is set, load cache dir presets
    env_file_path =Path(BASE_DIR).parent / '.env'
    if (env_file_path).exists():
        env_vars = dotenv_values(env_file_path)
        if 'CHIRPS_DIR' in env_vars:
            snakefile_config['apsim_params']['chirps_dir'] = env_vars['CHIRPS_DIR']

        if 'NP_CACHE_DIR' in env_vars:
            snakefile_config['apsim_params']['nasapower_cache_dir'] = env_vars['NP_CACHE_DIR']

        if 'ERA5_CACHE_DIR' in env_vars:
            snakefile_config['apsim_params']['era5_cache_dir'] = env_vars['ERA5_CACHE_DIR']

    updated_config_path = os.path.join(output_dir, 'config.yaml')
    with open(updated_config_path, "w") as f:
        ruamel_yaml.dump(snakefile_config, f)
        click.echo(f"Run configuration created under: {updated_config_path}")
        click.echo(f"Edit the configuarion with your individual options.")

    geojsons_folder.cleanup()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    """Generate region folders and APSIM files from a shapefile and YAML config."""
    prepare_study(config_path)


if __name__ == "__main__":
    main()
