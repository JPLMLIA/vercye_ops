import os
import shutil
import re
import tempfile
import yaml
import click
from pathlib import Path

import geopandas as gpd
from ruamel.yaml import YAML
from vercye_ops.apsim.convert_shapefile_to_geojson import convert_shapefile_to_geojson


def load_yaml(filepath):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def load_yaml_ruamel(filepath):
    yaml_loader = YAML()
    yaml_loader.preserve_quotes = True
    with open(filepath, "r") as f:
        return yaml_loader.load(f), yaml_loader
    
def prepare_study(config_path):
    config = load_yaml(config_path)

    shapefile_path = config["regions_shp_name"]
    admin_col = config["regions_shp_col"]
    filter_col = config.get("regions_shp_filter_col")
    filter_vals = config.get("regions_shp_filter_values", [])
    apsim_template_paths_filter_col = config.get("APSIM_TEMPLATE_PATHS_FILTER_COL_NAME")
    apsim_template_paths = config["APSIM_TEMPLATE_PATHS"]

    output_dir = Path(config_path).parent

    snakefile_template_config_path = os.path.join(output_dir, "run_config_template.yaml")
    snakefile_config, ruamel_yaml = load_yaml_ruamel(snakefile_template_config_path)

    projection_crs = snakefile_config['matching_params']['target_crs'].strip('\'"')
    geojsons_folder = tempfile.TemporaryDirectory()

    convert_shapefile_to_geojson(
        shp_fpath=shapefile_path,
        admin_name_col=admin_col,
        projection_crs=projection_crs,
        output_head_dir=geojsons_folder.name
    )

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

    years = snakefile_config['years']
    timepoints = snakefile_config['timepoints']

    for region in keep_regions:
        region_file = os.path.join(geojsons_folder.name, region, f"{region}.geojson")
        for year in years:
            for tp in timepoints:
                folder = os.path.join(output_dir, str(year), str(tp), region)
                os.makedirs(folder, exist_ok=True)
                shutil.copy(region_file, folder)

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

    updated_config_path = snakefile_template_config_path.replace("_template", "")
    with open(updated_config_path, "w") as f:
        ruamel_yaml.dump(snakefile_config, f)
        click.echo(f"Config updated with regions: {keep_regions}")
        click.echo(f"Written to: {updated_config_path}")

    geojsons_folder.cleanup()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    """Generate region folders and APSIM files from a shapefile and YAML config."""
    prepare_study(config_path)


if __name__ == "__main__":
    main()
