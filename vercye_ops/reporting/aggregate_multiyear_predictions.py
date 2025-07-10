import click
import os
from glob import glob

import pandas as pd

def get_available_timepoints(base_dir):
    timepoints = []
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for timepoint in os.listdir(year_path):
            tp_path = os.path.join(year_path, timepoint)
            if not os.path.isdir(tp_path):
                continue

            timepoints.append(timepoint)
    
    return timepoints

def get_avaiable_agg_levels(base_dir):
    agg_levels = []
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for timepoint in os.listdir(year_path):
            tp_path = os.path.join(year_path, timepoint)
            if not os.path.isdir(tp_path):
                continue

            preds_pattern = os.path.join(base_dir, year, timepoint, f'agg_yield_estimates_*_*.csv')
            agg_preds_files = glob(preds_pattern)
            agg_levels = [os.path.basename(f).split('_')[3] for f in agg_preds_files]

            agg_levels.extend(agg_levels)

    return list(set(agg_levels))

def collect_files(base_dir, agg_lvl_name, timepoint):
    pred_paths = {}
    gt_paths = {}

    # Collect all aggregated predictions at this agg lvl & timepoint
    for year in os.listdir(base_dir):
        preds_pattern = os.path.join(base_dir, year, timepoint, f'agg_yield_estimates_{agg_lvl_name}_*.csv')
        agg_preds_files = glob(preds_pattern)
        if len(agg_preds_files) > 1:
            raise Exception(f'More than one aggregated yield estimates file detected for {year}, {timepoint}, {agg_lvl_name}.')

        if len(agg_preds_files) == 1:
            pred_paths[year] = (agg_preds_files[0])


        gt_path = os.path.join(base_dir, year, f'referencedata__{agg_lvl_name}-{year}.csv')
        if os.path.exists(gt_path):
            gt_paths[year] = (gt_path)

    return pred_paths, gt_paths

def merge_preds_gt_yearly(preds_paths, gt_paths):
    yearly_dfs = []
    for year, year_pred_path in preds_paths.items():
        df_pred = pd.read_csv(year_pred_path)
        df_pred['year'] = year

        if year in gt_paths:
            df_gt = pd.read_csv(gt_paths[year])
            df_merged = pd.merge(df_pred, df_gt, on='region') # Inner join
            yearly_dfs.append(df_merged)
        else:
            yearly_dfs.append(df_pred)

    return yearly_dfs

        

def aggregate_years(base_dir, agg_lvl_name, timepoint):
    preds_paths, gt_paths = collect_files(base_dir, agg_lvl_name, timepoint)

    yearly_merged_dfs = merge_preds_gt_yearly(preds_paths, gt_paths)

    all_years_df = pd.concat(yearly_merged_dfs)
    all_years_df.sort_values(['year', 'region'], inplace=True)

    return all_years_df


@click.command()
@click.option('--base-dir', required=True, type=click.Path(exists=True), help='Basedirectory with subdirectories being the years, then timepoints.')
@click.option('--output-suffix', type=str, help='A unique suffix for the output.')
def main(base_dir: str, output_suffix: str):
    agg_lvls = get_avaiable_agg_levels(base_dir)
    timepoints = get_available_timepoints(base_dir)

    for agg_lvl_name in agg_lvls:
        for timepoint in timepoints:
            agg_df = aggregate_years(base_dir, agg_lvl_name, timepoint)
            out_file = os.path.join(base_dir, f'all_predictions_{output_suffix}_{agg_lvl_name}_{timepoint}.csv')
            agg_df.to_csv(out_file)


if __name__ == "__main__":
    main()