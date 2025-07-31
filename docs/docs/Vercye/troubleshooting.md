This section contains a few tips on what to do if you are encountering errors during pipeline execution.

- `Missing input files for rule xyz`: Check the error output under `affected files`. This outlines the files that snakemake expects to be present, however they are do not exist. You can manually check the directory if they exist. Typically this points to an error in the configuration, as for example when a `region.geojson` is supposed to be missing, this points to the basedirectory being incorrectly setup / the wrong path being provided to the base directory somewhere in the config.

- `Error in rule LAI_analysis`: An error related to not enough points or something similar typicallt indicates that in all of your LAI data there are not sufficient dates that meet the required minimum pixels without clouds for the specific region. 

However, this rarely should be the case when running with LAI data of multiple months (a typical season). Typically, this rather indicates that the `LAI parameters` were incorrectly set in the config. Check that the `lai_region`, `lai_resolution`, `lai_dir` and `file_ext` are correctly set.

- `Error in rule match_sim_real:  KeyError: None`: Typically indicates that the APSIM simulation was externally interrupted or unexpectedly failed. In such a case you will have to find the `--db_path` option in the `shell` section in the tracelog and manually delete the `.db` file.