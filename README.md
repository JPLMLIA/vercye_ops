# vercye_ops
Code to support operations of the Versatile Crop Yield Estimate (VeRCYe) pipeline.

# Features
* Tools to greatly reduce manual effort required for executing the VERYCE crop yield estimate pipeline.
* All workflow steps are wrapped in a well-documented CLI interface to permit step by step execution.
* The core CLI steps are also wrapped in a Snakemake-based data processing pipeline to batch execute yield estimates in an easy to run and reproducible manner.

# Quick Start
## Requirements
Requirements are listed in [requirements.txt](requirements.txt). You can install them via conda/mamba or pip as below. Navigate to this package's root directory and then run:

`conda install --yes --file requirements.txt`
or
`pip install -r requirements.txt`

Note: As of June 2024, if you use conda, you may need to manually install `Snakemake` and a specific dependency via pip: `pip install snakemake pulp=2.7.0`

## Package
Navigate to this package's root directory and then run:
`pip install -e .`

## Manually build APSIM on RHEL
Only necessary if you can't directly download/use the binary.

See the instructions at [vercye_ops/apsim/README.md](vercye_ops/apsim/README.md).

## Install jq
To manipulate json files via the command line, install https://jqlang.github.io/jq/ or if on HPC resources, you might be able to activate it with `module load jq`

# Run Instructions

## Convert Shapefile to GeoJSONs (one-time)
If you have a shapefile, extract each of the geospatial polygons into an individual geojson. This also sets up a directory tree with each geospatial polygon as its own folder (prepping for parallel execution of all regions).

```
python ~/Builds/vercye_ops/vercye_ops/apsim/convert_shapefile_to_geojson.py \
--shp_fpath /path/to/UKR_adm1.shp \
--admin_level oblast
--output_head_dir ~/Data/vercye_ops/Ukraine_Admin_Units/simstudy_20240607 \
--verbose
```

## Add .apsimx template files for each region (one-time)
To fill a template .apsimx file, make sure there is one present in each region's directory that you're trying to analyze. If you want to copy the same .apsimx file to many region subdirectories, you can do so like:
```
cd path/to/apsim_simulation_20240607  # Path where regions exist (e.g., from above shapefile conversion)
source_file="/path/to/my/template.apsimx"; for dir in *; do cp "$source_file" "${dir}/${dir}_template.apsimx"; done
```

## Update the Snakemake Config
To run a new simulation, save your desired configuration in the `.yaml` file in the `vercye_ops/snakemake` directory (`config.yaml` by default). Here, you'll want to update the head directory containing the simulation for each region, the region names themselves (likely generated with the shapefile conversion script), and the start/end dates for both the simulation and weather data.

Other config parameters should rarely (if ever) need updates.

## Full Snakemake Pipeline
1. Update the Snakemake configuration at [/vercye_ops/snakemake/config.yaml](/vercye_ops/snakemake/config.yaml)
1. Run `snakemake -c 4` to execute the pipeline with 4 cores. Some other useful command line flags are `--configfile <filename>` to use an alternative config or `-F` to force rerunning the pipeline.

  Other helpful command line args are documented on the [snakemake CLI page](https://snakemake.readthedocs.io/en/stable/executing/cli.html).

## Outputs and Rerunning
The simulation outputs will be saved at the head directory path you specified in your Snakemake config file.

To run the pipeline over the same region(s), either use Snakemake's `-F` flag or delete the log files at `vercye_ops/snakemake/logs_*`.

# Tests
Tests are handled by pytest. Navigate to the root directory and execute `pytest` to run them.

# License
See the [LICENCE](LICENSE)

# Copyright Notice
"Copyright 2024, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.