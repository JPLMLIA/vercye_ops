import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import click
import geopandas as gpd
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as DQ

from vercye_ops.utils.convert_shapefile_to_geojson import convert_shapefile_to_geojson
from vercye_ops.utils.env_utils import get_env_vars, is_env_set, load_yaml_ruamel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def rel_path(*path_parts):
    """Using relative path to files since code might be run from a different module."""
    return os.path.join(BASE_DIR, *path_parts)


def prepare_study(config: Dict[str, any], output_dir: str, lai_config_path: Optional[str] = None):
    """Prepares the initial study directory structure, APSIM files, and a run_config_template.

    Simplifies the setup of new yield studies:Creates the expected layout of a studies following the year->timepoint->regions pattern.
    Each region is extracted as an individual geojson from the provided shapefile. The APSIM files are copied to each region and
    adjusted to the correct simulation start and end date. Reference data is named and placed correctly. Additional available
    information is already placed into the final run_config template.

    Args:
        config (Dict): Config containing all the required information. An example can be found in examples/setup_config_template.yaml
        output_dir (str): Where to create the new basedirectory structure.
        lai_config_path (str): Path to an LAI config if available. Is used to fill in the lai directory and data name and resolution.

    Returns:
        path to the new run config template that is partially filled with available information.
    """
    shapefile_path = config["regions_shp_name"]
    admin_col = config["regions_shp_col"]
    filter_col = config.get("regions_shp_filter_col")
    filter_vals = config.get("regions_shp_filter_values", [])
    apsim_template_paths_filter_col = config.get("APSIM_TEMPLATE_PATHS_FILTER_COL_NAME")
    apsim_template_paths = config["APSIM_TEMPLATE_PATHS"]

    if os.path.exists(output_dir):
        raise ValueError(f"A basedirectory already exists under {output_dir}.")
    os.makedirs(output_dir)

    snakefile_template_config_path = rel_path("examples/run_config_template.yaml")
    snakefile_config, ruamel_yaml = load_yaml_ruamel(snakefile_template_config_path)

    # Update snakemakefile with values from the template
    snakefile_config["years"] = config["years"]
    snakefile_config["timepoints"] = config["timepoints"]
    snakefile_config["apsim_params"]["time_bounds"] = config["timepoints_config"]

    target_crs = config["target_crs"].strip("'\"")
    if "proj" in target_crs:
        # force: "<single-quoted value>"
        snakefile_config["matching_params"]["target_crs"] = DQ(f"'{target_crs}'")
    else:
        # still force double quotes even if not proj-based
        snakefile_config["matching_params"]["target_crs"] = DQ(target_crs)

    projection_crs = target_crs
    geojsons_folder = tempfile.TemporaryDirectory()

    # Create individual geojsons of each shapefile entry. Extracts centroid in target crs.
    convert_shapefile_to_geojson(
        shp_fpath=shapefile_path,
        admin_name_col=admin_col,
        projection_crs=projection_crs,
        output_head_dir=geojsons_folder.name,
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
            raise ValueError(f"GeoJSON {shapefile_path} has more than one polygon.")

        if filter_col is not None and gdf[filter_col][0] not in filter_vals:
            continue

        if apsim_template_paths_filter_col:
            if apsim_template_paths_filter_col not in gdf.columns:
                raise ValueError(f"Column {apsim_template_paths_filter_col} not found in the shapefile.")
            key = gdf[apsim_template_paths_filter_col][0]
            regions_apsimfile[region_name] = apsim_template_paths[key]
        else:
            regions_apsimfile[region_name] = apsim_template_paths["all"]

        keep_regions.append(region_name)

    # Copy all regions into each year/timepoint directory
    years = snakefile_config["years"]
    timepoints = snakefile_config["timepoints"]

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
                start = snakefile_config["apsim_params"]["time_bounds"][year][tp]["sim_start_date"]
                end = snakefile_config["apsim_params"]["time_bounds"][year][tp]["sim_end_date"]
                template_path = regions_apsimfile[region]

                with open(template_path, "r", encoding="utf-8") as f:
                    data = f.read()

                data = re.sub(
                    r'"Start":\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"',
                    f'"Start": "{start}T00:00:00"',
                    data,
                )
                data = re.sub(
                    r'"End":\s*"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"',
                    f'"End": "{end}T00:00:00"',
                    data,
                )

                out_path = os.path.join(output_dir, str(year), str(tp), region, f"{region}_template.apsimx")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(data)

    # Copy the reference data
    known_agg_lvls = []
    for year, path_list in config["REFERENCE_DATA_PATHS"].items():
        for entry in path_list:
            for reference_data_name, original_path in entry.items():
                if not Path(original_path).suffix == ".csv":
                    raise ValueError(f"Reference data {original_path} must be .csv")

                new_name = f"referencedata_{reference_data_name}-{year}.csv"
                new_path = os.path.join(str(output_dir), str(year), new_name)
                shutil.copy(original_path, new_path)
                known_agg_lvls.append(reference_data_name)

    config.yaml_set_comment_before_after_key("REFERENCE_DATA_PATHS", before=None)

    # Update config with the region metadata
    snakefile_config["regions"] = keep_regions
    snakefile_config["regions_shp_col"] = admin_col
    snakefile_config["regions_shp_filter_col"] = filter_col
    snakefile_config["regions_shp_filter_values"] = filter_vals
    snakefile_config["regions_shp_name"] = shapefile_path
    snakefile_config["sim_study_head_dir"] = str(output_dir)

    # Fill in LAI section in run_config based on lai creation config if it exists
    if os.path.exists(lai_config_path):
        lai_config, _ = load_yaml_ruamel(lai_config_path)

        # Check that the file was actually filled in as in not just the created template
        if "XXXX" not in lai_config["out_dir"]:
            snakefile_config["lai_source"] = str(Path(lai_config["geojson_path"]).name)
            snakefile_config["lai_params"]["lai_dir"] = os.path.join(lai_config["out_dir"], "merged-lai")
            snakefile_config["lai_params"]["lai_region"] = lai_config["region_out_prefix"]
            snakefile_config["lai_params"]["lai_resolution"] = lai_config["resolution"]

    # Transfer LAI start-enddate configuration to run config
    lai_dict = {}
    for year in config["timepoints_config"]:
        lai_dict[year] = {}
        for timepoint in config["timepoints_config"][year]:
            start = config["timepoints_config"][year][timepoint]["lai_start_date"]
            end = config["timepoints_config"][year][timepoint]["lai_end_date"]

            # Create a flow-style list: ['start', 'end']
            seq = CommentedSeq([start, end])
            seq.fa.set_flow_style()

            lai_dict[year][timepoint] = seq

    snakefile_config["lai_params"]["time_bounds"] = lai_dict

    # If env file is set, load cache dir presets
    if is_env_set():
        env_vars = get_env_vars()
        if "CHIRPS_DIR" in env_vars:
            snakefile_config["apsim_params"]["chirps_dir"] = env_vars["CHIRPS_DIR"]

        if "NP_CACHE_DIR" in env_vars:
            snakefile_config["apsim_params"]["nasapower_cache_dir"] = env_vars["NP_CACHE_DIR"]

        if "ERA5_CACHE_DIR" in env_vars:
            snakefile_config["apsim_params"]["era5_cache_dir"] = env_vars["ERA5_CACHE_DIR"]

        if "EE_PROJECT_NAME" in env_vars:
            snakefile_config["apsim_params"]["ee_project"] = env_vars["EE_PROJECT_NAME"]

        if "APSIM_PATH" in env_vars:
            snakefile_config["apsim_execution"]["local"]["executable_fpath"] = env_vars["APSIM_PATH"]

        if "MATCHING_SCRIPT_PATH" in env_vars:
            snakefile_config["scripts"]["match_sim_real"] = env_vars["MATCHING_SCRIPT_PATH"]

    # Update cropmask keys based on required years
    cropmasks_data = {}
    for year in config["years"]:
        cropmasks_data[year] = "XXXX"
    snakefile_config["lai_params"]["crop_mask"] = cropmasks_data

    # Update ref data keys based on provided years and agg levels
    agg_lvls_data = {}
    for agg_lvl in known_agg_lvls:
        if agg_lvl != "primary":  # dont add primary, as this is default
            agg_lvls_data[agg_lvl] = "XXXX"
    snakefile_config["eval_params"]["aggregation_levels"] = agg_lvls_data
    snakefile_config["study_id"] = Path(output_dir).name

    # Write updated and prepared run config
    updated_config_path = os.path.join(output_dir, "config.yaml")
    with open(updated_config_path, "w") as f:
        ruamel_yaml.dump(snakefile_config, f)
        click.echo(f"Run configuration created under: {updated_config_path}")
        click.echo("Edit the configuarion with your individual options.")

    geojsons_folder.cleanup()
    return updated_config_path


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def main(config_path):
    """Utility to simplify the expected setup of a yield study.

    Generates region folders containing individual geojsons and APSIM files for each year and timepoint
    from a shapefile and config.

    Args:
        config_path (str): Path to the .yaml config.

    Returns:
        None
    """

    config, _ = load_yaml_ruamel(config_path)
    output_dir = str(Path(config_path).parent / Path(config_path).parent.name)
    lai_config_path = str(Path(config_path).parent / "lai_config.yaml")

    prepare_study(config, output_dir, lai_config_path)


if __name__ == "__main__":
    main()
