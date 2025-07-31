The VeRCYe pipeline generates outputs at multiple levels of abstraction, including:

- **Pixel-wise yield predictions**
- **Predictions for each region of interest (ROI)**
- **Aggregated insights across multiple ROIs**
- **Aggregated insights across multiple ROIs & Years**

This document details all output artifacts and their computation methods.

## 1. Per Region (-and timepoint) Outputs

### Reports

- **`yield_report.html`**: This summarize the yield study on this region and timepoint. The displayed values are the following:
    - Number of simulation traces in mean data: The number of APSIM simulations that best match remotely sensed LAI data, after filtering out simulations in `match_sim_rs_lai.py`. We referr to these as 'matched'
    - Date range: APSIM simulation date range.
    - Mean yield rate: See `converted_map_yield_estimate.csv` under `mean_yield_kg_ha`.
    - Production: See `converted_map_yield_estimate.csv` under `total_yield_production_ton`.
    - LAI filtered on step x: APSIM Simulated `Wheat.Leaf.LAI` (or `Maize.Leaf.LAI`) for a specific simulation and simulation date, with the simulation being filtered out at step x, as defined in `match_sim_rs_lai.py`.
    - Yield filtered on step x: APSIM Simulated yield for a specific simulation and simulation date, with the simulation being filtered out at step x, as defined in `match_sim_rs_lai.py`. Yield in kg/ha.
    - RS mean LAI: The remotely sensed mean LAI at each date. The mean is taken over the spatial axes, so the mean of the pixel values of all pixel locations that are within the regions geometry.
    - Mean LAI: The mean simulated LAI eat each date. Hereby the mean for each date, is computed as the mean of all matched (not-filtered out) AP-SIM simulated `Wheat.Leaf.LAI` (or `Maize.Leaf.LAI`) values at that date. Yield in
    - Mean Yield: Analogues to `Mean LAI` for the APSIM simulated yield. All simulated yield values are in kg/ha.
- **`yield_report.png`**: A non-interactive snapshot with lower resolution of the `.html` yield report.
- **`weather_report.html`**: Allows to interactively inspect the meteorological data. All values except the rain graph are computed for the centroid of the region. For the rain graph, if the precipitation data source in the experiment is NASA_Power, this is also for the centroid. If the precipitation data source is CHIRPS, this might either be the value at the centroid or the mean of all CHIRPS values within the region, depending on what was specified in the configuration. 


### Maps

- **`yield_map.tif`**: Raster output representing the yield in kg/ha per pixel. The pixel values are derived my multiplying the remotely sensed LAI with a conversion factor. The conversion factor is computed from the matched simulations of the region.

### Detailed Outputs

- **`converted_map_yield_estimate.csv`**: 
    - mean_yield_kg_ha: Mean yield based on pixel level yield predictions from `yield_map.tif`. Mean over all pixels estimates in kg/ha.
    - median_yield_kg_ha: Median yield based on pixel level yield predictions from `yield_map.tif`. Medial over all pixel estimates in kg/ha. 
    - total_area_ha: Total cropland in region in ha. Computed from number of cropland pixels in region multiplied with the pixel size.
    - total_yield_production_kg: Total yield from the region in kg. Sum over yield per pixel.
    - total_yield_production_ton: `total_yield_production_kg` / 1000

- **`sim_matches.csv`**: Contains aggregated statistics per APSIM simulation:
    - SimulationID: The internal ID of the APSIM simulation.
    - Max_Yield: The maximum yield that the APSIM simulation reaches during the specified simulation date range.
    - Max_LAI: The maximum LAI that the APSIM simulation reaches during the specified simulation date range.
    - StepFilteredOut: The step number at which the APSIM simulation was removed from matching candidates, as described in `match_sim_rs_lai.py`. If this field is empty/nan, the simultion was matched and not filtered out.
    - Further details not publicly available at the moment.

- **matched_sims.csv**: Complete simulations data of the matched simulations from APSIM.

- **`conversion_factor.csv`**: Contains numerous values related to the APSIM simulations (and remotely sensed LAI). `Max_Yield` and `Max_LAI` as defined in `sim_matches.csv`.
    - apsim_mean_yield_estimate_kg_ha: The mean of `Max_Yield` over the remaining APSIM simulations fter filtering out simulations as described in `match_sim_rs_lai.py`.
    - apsim_max_matched_lai: The maximum `Max_Sim_LAI` value over all matched (filtered) APSIM simulations. 
    - apsim_max_matched_lai_date: The correspinding date on which `apsim_max_matched_lai` is reached.
    - apsim_max_all_lai: The maximum `Max_Sim_LAI` value over all (not-filtered) APSIM simulations. 
    - apsim_max_all_lai_date: The correspinding date on which `apsim_max_all_lai` is reached.
    - apsim_matched_std_yield_estimate_kg_ha: The standard deviation in kg/ha of `Max_Yield` over all matched (filtered) APSIM simulations.
    - apsim_all_std_yield_estimate_kg_ha: The standard deviation in kg/ha of `Max_Yield` over all (not-filtered) APSIM simulations.
    - apsim_matched_maxlai_std: The standard LAI deviation of `Max_Sim_LAI` over the matched (filtered) APSIM simulations.
    - apsim_all_maxlai_std: The standard LAI deviation of `Max_Sim_LAI` over all (not-filtered) APSIM simulations.
    - max_rs_lai: For each date the remotely sensed mean LAI is computed (negative values clipped to 0). Hereby the mean is taken spatially over the region. `max_rs_lai` then defines the maximum LAI value of these. 
    - max_rs_lai_date: The correspinding date of the `max_rs_lai` value.
    - conversion_factor: The factor used to convert the remotely sensed LAI raster to yield estimates that are in kg/ha per pixel.


- **`LAI_STATS.csv`**: Insights on the estimated LAI from the remotely sensed data.
    Before computation all negative values are clipped to 0.
    - Date
    - n_pixels: Number of non-nan pixels in the estimated LAI raster for this date.
    - interpolated: 1 if this LAI value is interpolated based on surrounding values since no remote sensed image was available on this date. 0 it is estimated with the ML model from the remotely sensed image.
    - LAI mean: Mean estimated LAI value over all spatial locations at this date. 
    - LAI stddev: Standard deviation of LAI values over all spatial locations at this date.
    - LAI mean adjusted: Mean estimated adjsuted LAI value over all spatial locations at this date. Adjustment is used to adjust for different crops e.g maize or wheat as specified in `3_analysis_LAI.py`.
    - LAI stddev adjusted: Analogous to `LAI stddev` for the adjusted LAI.

- **`met.csv`**: Meteorological data fetched for each date in the daterange. If CHIRPS was used for precipitation data, it is **NOT** included here. Documentation of other columns will be added soon. The Column names follow the NASAPower output format. If ERA5 data is fetched it is converted to the same output format as NASAPower has.
- **`weather.met`**: Met file generated from `met.csv*` that follows the format that APSIM expects. Includes the CHIRPS data in the rain column if CHIPRS is used.

- **`cropmask_constrained.tif`**: Binary cropmask (raster) constrained to the region.
- **`LAI_MAX.tif`**: Raster of the remotely sensed LAI given as the maximum value per pixel troughout the date range.
- **`VALID`** This file simply indicates whether the region meets the minimum cropland pixel requirements and is only used by `Snakemake`.


## 2. Aggregated Outputs
The aggregated outputs are produced for each year-timepoint combination. They combine the artifacts from the indival regions into single files that are easier to work with. The filenames will contain suffixed defined as `studyID_year_timepoint`, to allow easier sharing of these results.

- **`final_report_suffix.pdf`**: This document gives a final, simple to understand overview of all regional results and preview images of the maps. The description of values not listed below is either documented under the regional output section or the validation outputs section.

    - date range: Simulation start and end date
    - Met-data Cutoff Date: Date specified in the config file for last data to provide met data to APSIM predictions. Not necessarily the true last date, as data might be unavailable from an earlier timepoint already.
    - total yield: The sum of the estimated yield of all regions.
    - Estimated Yield (Weighted Mean): Sum of predicted production of all simulation level ROIs / sum of total cropland from all simulation level ROIs.
    - Estimated Total Production: Sum of the prediction production from all the simulation level ROIs.
    - Total cropland area: The sum of all the ropland area of all ROIs in ha.
    - Crop productivity pixel level: Visualizes the crop productivity in kg/ha per pixel. Derived from the remotely sensed LAI data coupled with the APSIM simulation trough the conversion factor.

- **`agg_yield_estimates_{agglevel}_suffix.csv`**: The detailed predictions with debug data from all simulation ROIs aggregated by the specified column from the config aggregation level. If `primary`, it is the simulation level data.

- **`agg_laicurves_suffix.html`**: Interactive visualization of the LAI data from all regions in the yield study. Enables quicker debugging.

### Aggregated Maps
- **`yield_map_{aggregation-level}.png`**: Preview image of all ROI boundaries with the corresponding mean and median yield in kg/ha. The aggregation level defines the name specified in the config by which the simulation level data was aggregated. If `primary`, it is the simulation level.
- **`aggregated_yield_map_preview.png`**: Downsampled image preview of the all regional pixel-level yieldmaps merged into one.

- **`aggregated_yield_map_suffix.tif`**: Spatially aggregated yield maps from all regions (See regional `yield_map.tif`).
- **`aggregated_LAI_max_suffix.tif`**: Spatially aggregated LAI_MAX maps from all regions (See regional `LAI_MAX.tif`).
- **`aggregated_cropmask_suffix.tif`**: Spatially aggregated cropmasks from all regions (See regional `cropmask_constrained.tif`). 
- **`aggregated_region_boundaries_suffix.geojson`**: All regional geojsons merged into a single one for easier importing into GIS.

### Aggregated Meteorological Insights

- **`aggregated_met_stats_suffix.pdf`**: Averaged meteorological data per region visualized as a vector map. Includes statistics for the specific year and averaged over the last 20 years. Results are computed as per-year averages (grouped by location lat/lon) from the met files.
- **`aggregated_met_stats_suffix.tif`**: Same met stats as in the pdf, but as a tif with the same resolution as LAI data. Facilitates analysis for field-scale studies.

### Multiyear Aggregated Outputs
- **`multiyear_summary_{study_id}.html`**: Contains aggregated LAI curves for every year to enable spotting differenced in LAI throughout time. Contains metrics computed over all predictions from multiple years.

- **`all_predictions_{agg_lvl}_{timepoint}.csv`**: A csv containing all yield + reference data from multiple years per timepoint and aggregation levels. Very helpfull for further analysis of the predictions and additional validation. Also containing LAI & simulation details for the `primary` aggregation level.


## 3. Validation Outputs
If reported data was provided, common metrics are written to the `final_report.html` and `evaluation-{agglevel}.csv`.
Details regarding the computation are implemented in `vercye_ops/matching_sim_real/evaluate_yield_estimates.py`.

### Error Metrics
- **Mean Error (`mean_err_kg_ha`)**: The mean error computed as the mean over the yield errors (reported_mean_yield - predicted_mean_yield) of all study regions. Hereby the predicted and reported yield are the mean yield of the region in kg/ha.
- **Median Error (`median_err_kg_ha`)**: The median error computed as the median over the yield errors (reported_mean_yield - predicted_mean_yield) of all study regions. Hereby the predicted and reported yield are the mean yield of the region in kg/ha.
- **Absolute Mean Error (`mean_err_kg_ha`)**: Absolute mean error, see above.
- **Absolute Median Error (`median_err_kg_ha`)**: Absolute median errorm see above.
- **Root Mean Square Error (`rmse`)**: The root mean square error of the predicted and reported mean yields of all study regions in kg/ha.
- **Relative RMSE (`rrmse`)**: In percent. Computed as `rmse / mean(reported_mean_yield) * 100`
- **R² Score Scikit (`r2_scikit`)**: The R2 score of the predicted and reported mean yield per region. Computed as the coefficient of determination with [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
- **R² Score Excel (`r2_rsq_excel`)**: The R2 score of the predicted and reported mean yield per region. Computed as the quare of the Pearson product moment correlation coefficient , as implemented in the excel `RSQ` function [See here](https://support.microsoft.com/en-us/office/rsq-function-d7161715-250d-4a01-b80d-a8364f2be08f).