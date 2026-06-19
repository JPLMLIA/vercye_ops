# Experimental copy of match_sim_real_17-06-25.py for the Ukraine yield-improvement
# work (branch: ukraine-yield-improvement). Logic is identical to the operational
# script at baseline; only the fragile relative import was replaced with the proper
# absolute package import so the file is robust regardless of run cwd.
import click
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_config

from vercye_ops.utils.init_logger import get_logger
from vercye_ops.matching_sim_real.utils import load_simulation_data

logger = get_logger()


# Function to calculate required metrics for each simulation
def calculate_metrics(merged_df, sim_id, crop_name, drought_threshold, rs_lai_agg_col):
    """
    Calculate the metrics required to assess the match between simulated and remotely sensed LAI data.

    Parameters
    ----------
    merged_df : pandas.DataFrame
        DataFrame containing merged simulation and RS LAI data aligned by 'Date'.
        
    sim_id : int
        The SimulationID for which metrics are calculated.

    Returns
    -------
    tuple
        A tuple containing the SimulationID, LAI gap, timing gap, Green LAI RMSE, and Senescence LAI RMSE and Max simulated LAI.
    """

    if not crop_name:
        raise ValueError("No Crop adjustment was specified. Therefore it is unclear how to lookup LAI values. Currently supporting either wheat or maize.")

    crop_name = crop_name.capitalize()

    # Filter the merged DataFrame for the current SimulationID
    merged_df_subset = merged_df.loc[merged_df['SimulationID'] == sim_id, :]

    # Extract the aligned LAI values from the merged DataFrame
    sim_lai = merged_df_subset[f'{crop_name}.Leaf.LAI']  # Simulated LAI values
    rs_lai_aligned = merged_df_subset[rs_lai_agg_col]      # Corresponding RS LAI values

    # check if duplicate dates exist
    if rs_lai_aligned.index.duplicated().any():
        raise ValueError("Duplicate dates found in the RS LAI data. Please ensure that the dates are unique.")

    # Robustness guard: some APSIM sowing-date scenarios produce simulations whose date
    # range ends before the RS LAI observation window begins (e.g. an early-terminated
    # sim ending in autumn, with no overlap into spring). For such a sim the aligned RS
    # series is entirely NaN, and pandas >=2.0 raises "Encountered all NA values" on
    # idxmax (older pandas returned NaN). Such a sim carries no signal for matching
    # against the RS LAI, so return NaN metrics; it is then excluded by the timing-gap
    # filter (NaN Timing_Gap fails the <= threshold test) and never enters the matched set.
    if sim_lai.isna().all() or rs_lai_aligned.isna().all():
        max_yield = merged_df_subset['Yield'].max()
        return sim_id, np.nan, np.nan, np.nan, np.nan, np.nan, max_yield, np.nan, np.nan

    # Calculate the maximum LAI values for both simulated and RS data and get gap
    max_sim_lai = sim_lai.max()
    max_rs_lai = rs_lai_aligned.max()
    lai_gap = abs(max_sim_lai - max_rs_lai)

    # Calculate the peak sim/real LAI and gap between the sim/real LAI values in terms of days
    max_sim_idx = sim_lai.idxmax()
    max_rs_idx = rs_lai_aligned.idxmax()
    timing_gap = np.abs((max_sim_idx - max_rs_idx).days)

    max_yield = merged_df_subset['Yield'].max()
    max_sim_lai_date = merged_df_subset[f'{crop_name}.Leaf.LAI'].idxmax()

    # TODO: Validate the following logic and threshold
    if max_rs_lai < drought_threshold:
        return sim_id, lai_gap, timing_gap, np.nan, np.nan, np.nan, max_yield, max_sim_lai, max_sim_lai

    # Calculate the RMSE during the green LAI stage (before the peak LAI)
    # This is when the crop leaves are photosynthetically active
    earliest_valid_rs_loc = rs_lai_aligned.index.get_loc(rs_lai_aligned[rs_lai_aligned > drought_threshold].first_valid_index())
    max_rs_loc = rs_lai_aligned.index.get_loc(max_rs_idx)
    latest_valid_rs_loc = rs_lai_aligned.index.get_loc(rs_lai_aligned[rs_lai_aligned > drought_threshold].last_valid_index())

    # Note: Change < to <= since it none of the simulations were passing this check
    # if not earliest_valid_rs_loc < max_rs_loc < latest_valid_rs_loc:
    if not earliest_valid_rs_loc <= max_rs_loc <= latest_valid_rs_loc:
        raise ValueError("Invalid index locations for LAI RMSE calculation.")

    # TODO FIX - THIS IS NOTT GREEN LAI need to take only until peak
    green_lai_rmse = np.sqrt(np.nanmean(np.square(sim_lai.iloc[earliest_valid_rs_loc:latest_valid_rs_loc+1] - rs_lai_aligned.iloc[earliest_valid_rs_loc:latest_valid_rs_loc+1])))

    # Calculate the RMSE during the senescence stage (after the peak LAI)
    # This is when the crop leaves are not photosynthetically active
    senescence_lai_avg = np.nanmean(sim_lai.iloc[max_rs_loc:latest_valid_rs_loc+1])
    senescence_lai_rmse = np.sqrt(np.nanmean(np.square(sim_lai.iloc[max_rs_loc:latest_valid_rs_loc+1] - rs_lai_aligned.iloc[max_rs_loc:latest_valid_rs_loc+1])))

    # Return the calculated metrics along with the SimulationID
    return sim_id, lai_gap, timing_gap, green_lai_rmse, senescence_lai_rmse, senescence_lai_avg, max_yield, max_sim_lai, max_sim_lai_date

def apply_matching(results_df, max_rs_lai, drought_threshold, lai_gap_quantile, senescence_lai_quantile, green_lai_rmse_quantile):
    # Step 1: Severe drought case - take mean of 3 simulationss with lowest yield
    if max_rs_lai < drought_threshold:
        logger.info("Severe drought case detected. Taking mean of 3 simulations with lowest yield.")
        worst_sim_ids = results_df['Max_Yield'].nsmallest(3).index
        results_df.loc[~results_df.index.isin(worst_sim_ids), 'StepFilteredOut'] = 1
        return results_df

    #####################
    # Step 2: Filter by LAI gap (lowest X% based on provided quantile)
    lai_gap_threshold = results_df['LAI_Gap'].quantile(lai_gap_quantile)
    results_df.loc[results_df['LAI_Gap'] > lai_gap_threshold, 'StepFilteredOut'] = 2
    
    #####################
    # Step 3: Filter by timing gap (+/-5 days, adjust if necessary)
    timing_gap = 5
    good_inds = (results_df['Timing_Gap'] <= timing_gap) & results_df['StepFilteredOut'].isna()
    while good_inds.any() == False:
        timing_gap += 5
        good_inds = (results_df['Timing_Gap'] <= timing_gap) & results_df['StepFilteredOut'].isna()

    results_df.loc[~good_inds & results_df['StepFilteredOut'].isna(), 'StepFilteredOut'] = 3
    logger.info(f'Final timing gap: {timing_gap} days')

    #####################
    # Step 4: Filter by Senescence LAI (highest X% based on provided quantile)
    # Using the highest average, even though these might not be the true best fit to the LAI, but the RS LAI underestimates senescence LAI
    senescence_lai_threshold = results_df.loc[results_df['StepFilteredOut'].isna(), 'Senescence_LAI_AVG'].quantile(senescence_lai_quantile)
    
    results_df.loc[(results_df['Senescence_LAI_AVG'] < senescence_lai_threshold) & results_df['StepFilteredOut'].isna(),
                   'StepFilteredOut'] = 4

    #####################
    # Step 5: Final selection based on Green LAI RMSE (lowest X% based on provided quantile)
    if results_df['StepFilteredOut'].isna().sum() >= 10:  # Per paper, only apply this filter if >= 10 simulations remain
        green_lai_rmse_threshold = results_df.loc[results_df['StepFilteredOut'].isna(), 'Green_LAI_RMSE'].quantile(green_lai_rmse_quantile)
        results_df.loc[(results_df['Green_LAI_RMSE'] > green_lai_rmse_threshold) & results_df['StepFilteredOut'].isna(),
                       'StepFilteredOut'] = 5
    else:
        logger.info(f"Skipping final Green LAI RMSE filter step, as only {results_df['StepFilteredOut'].isna().sum()} simulations remain.")

    return results_df


def match_simulations(rs_lai_csv, db_path, sim_matches_output_fpath, conversion_factor_output_fpath, crop_name, use_adjusted, lai_agg_type, lai_gap_quantile, senescence_lai_quantile, green_lai_rmse_quantile, drought_threshold, n_jobs, verbose):
    """
    Find the best matching simulations to remotely sensed LAI.
    
    RS_LAI_CSV: Path to the CSV file containing the RS LAI time series.
    DB_PATH: Path to the SQLite database containing the simulation data.
    """
    if verbose:
        logger.setLevel('INFO')
        
    # Load RS LAI data
    rs_df = pd.read_csv(rs_lai_csv, index_col='Date', parse_dates=['Date'], dayfirst=True)


    # Identify which LAI column to use based on the aggregation type and adjustment flag
    rs_lai_agg_str = 'Mean' if lai_agg_type.lower() == 'mean' else 'Median'
    rs_lai_agg_col = f'LAI {rs_lai_agg_str} Adjusted' if use_adjusted else f'LAI {rs_lai_agg_str}'
    logger.info(f'Using {rs_lai_agg_col} column for RS data!')

    # Load simulation data from SQLite database
    sim_df = load_simulation_data(db_path, crop_name)
    sim_ids = sim_df['SimulationID'].unique()
    
    ###################################
    # Step 0: Align the dates between simulated LAI data and remotely sensed LAI data
    # Merge on date index
    merged_df = rs_df.merge(sim_df, how='right', left_index=True, right_index=True)
    
    # Process simulations in parallel
    logger.info(f'Processing {len(sim_df)} simulation rows...')
    with parallel_config(n_jobs=n_jobs):
        results = Parallel()(delayed(calculate_metrics)(merged_df, sim_id, crop_name, drought_threshold, rs_lai_agg_col) for sim_id in sim_ids)
    
    # Convert results to a DataFrame. These columns must match the output order of calculate_metrics
    results_df = pd.DataFrame(results, columns=['SimulationID', 'LAI_Gap', 'Timing_Gap', 'Green_LAI_RMSE', 'Senescence_LAI_RMSE', 'Senescence_LAI_AVG', 'Max_Yield', 'Max_Sim_LAI', 'Max_Sim_LAI_Date'])
    results_df = results_df.set_index('SimulationID').sort_index()
    results_df['StepFilteredOut'] = pd.NA

    max_rs_lai = rs_df[rs_lai_agg_col].max()
    results_df = apply_matching(results_df, max_rs_lai, drought_threshold, lai_gap_quantile, senescence_lai_quantile, green_lai_rmse_quantile)
    
    #####################
    # Export yield info to disk
    good_sim_ids = results_df.loc[results_df['StepFilteredOut'].isna()]

    results_df.to_csv(sim_matches_output_fpath)

    logger.info(f"Results saved to: {sim_matches_output_fpath}")
    logger.info(f"Best matching Simulation IDs:\n{good_sim_ids.index.to_list()}")

    ###################################
    # Generate Yield Conversion Factor in prep for yield map generation
    apsim_mean_yield_estimate = results_df.loc[results_df['StepFilteredOut'].isna(), 'Max_Yield'].mean()
    if apsim_mean_yield_estimate == 0:
        logger.error("APSIM mean yield estimate is 0, which is likely incorrect and could cause downstream errors.")
        
    max_rs_lai_row = rs_df.loc[rs_df[rs_lai_agg_col].idxmax()]
    max_rs_lai = max_rs_lai_row[rs_lai_agg_col]
    max_rs_lai_date = max_rs_lai_row.name

    conversion_factor = 0 if max_rs_lai == 0 else apsim_mean_yield_estimate / max_rs_lai

    # Compute maximum LAI with date for the final selected simulations and for all simulations
    max_matched_sim_lai_row = results_df.loc[results_df.loc[results_df['StepFilteredOut'].isna(), 'Max_Sim_LAI'].idxmax()]
    max_matched_sim_lai = max_matched_sim_lai_row['Max_Sim_LAI']
    max_matched_sim_lai_date = max_matched_sim_lai_row['Max_Sim_LAI_Date']
    max_total_sim_lai_row = results_df.loc[results_df['Max_Sim_LAI'].idxmax()]
    max_total_sim_lai = max_total_sim_lai_row['Max_Sim_LAI']
    max_total_sim_lai_date = max_total_sim_lai_row['Max_Sim_LAI_Date']

    # Compute yield and lai std dev for the final selected simulations and for all simulations
    apsim_matched_std_yield = results_df.loc[results_df['StepFilteredOut'].isna(), 'Max_Yield'].std()
    apsim_total_std_yield = results_df['Max_Yield'].std()
    apsim_matched_maxlai_std = results_df.loc[results_df['StepFilteredOut'].isna(), 'Max_Sim_LAI'].std()
    apsim_total_maxlai_std = results_df['Max_Sim_LAI'].std()
    
    conversion_df = pd.DataFrame([{'apsim_mean_yield_estimate_kg_ha': apsim_mean_yield_estimate, 
                                   'max_rs_lai': max_rs_lai, 
                                   'max_rs_lai_date': max_rs_lai_date,
                                   'conversion_factor': conversion_factor,
                                   'apsim_max_matched_lai': max_matched_sim_lai,
                                   'apsim_max_matched_lai_date': max_matched_sim_lai_date,
                                   'apsim_max_all_lai': max_total_sim_lai,
                                   'apsim_max_all_lai_date': max_total_sim_lai_date,\
                                   'apsim_matched_std_yield_estimate_kg_ha': apsim_matched_std_yield,
                                   'apsim_all_std_yield_estimate_kg_ha': apsim_total_std_yield,
                                   'apsim_matched_maxlai_std': apsim_matched_maxlai_std,
                                   'apsim_all_maxlai_std': apsim_total_maxlai_std}])
    conversion_df.to_csv(conversion_factor_output_fpath, index=False)

    logger.info(f"Mean yield estimate: {apsim_mean_yield_estimate:0.2f}, Max RS LAI: {max_rs_lai:0.2f}")
    logger.info(f"Conversion factor: {conversion_factor:0.2f}")
    logger.info(f"Conversion file saved to: {conversion_factor_output_fpath}")


@click.command()
@click.option('--rs_lai_csv', required=True, type=click.Path(exists=True), help='Path to remotely sensed LAI CSV file')
@click.option('--db_path', required=True, type=click.Path(exists=True), help='Path to APSIM database')
@click.option('--sim_matches_output_fpath', required=True, type=click.Path(), help='Path to save matching simulations to')
@click.option('--conversion_factor_output_fpath', required=True, type=click.Path(), help='Path to save conversion factor to')
@click.option('--crop_name', required=True, type=click.Choice(['wheat', 'maize']), help='Crop name to use for LAI lookup in APSIM')
@click.option('--use_adjusted', is_flag=True, help='Whether or not to used the adjusted LAI values')
@click.option('--lai_agg_type', required=True, type=click.Choice(['mean', 'median']), help='Type of how the LAI was aggregated over a ROI. "mean" or "median" supported.')
@click.option('--lai_gap_quantile', default=0.2, type=float, help='Percentile threshold for LAI gap filtering (Step 2)')
@click.option('--senescence_lai_quantile', default=0.8, type=float, help='Quantile threshold for Senescence LAI filtering (Step 4)')
@click.option('--green_lai_rmse_quantile', default=0.2, type=float, help='Quantile threshold for Green LAI RMSE filtering (Step 5)')
@click.option('--drought_threshold', default=0.9, type=float, help='Threshold for severe drought conditions')
@click.option('--n_jobs', default=10, help='Number of parallel jobs to run')
@click.option('--verbose', is_flag=True, help='Enable verbose mode to print out debug info.')
def cli(rs_lai_csv, db_path, sim_matches_output_fpath, conversion_factor_output_fpath, crop_name, use_adjusted, lai_agg_type, lai_gap_quantile, senescence_lai_quantile, green_lai_rmse_quantile, drought_threshold, n_jobs, verbose):
    match_simulations(rs_lai_csv, db_path, sim_matches_output_fpath, conversion_factor_output_fpath, crop_name, use_adjusted, lai_agg_type, lai_gap_quantile, senescence_lai_quantile, green_lai_rmse_quantile, drought_threshold, n_jobs, verbose)


if __name__ == '__main__':
    cli()