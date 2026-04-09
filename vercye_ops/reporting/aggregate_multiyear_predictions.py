import os
from glob import glob

import click
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

    return list(set(timepoints))


def _extract_agg_level_name(filename, year, timepoint):
    """Extract aggregation level name from a filename like
    agg_yield_estimates_{level_name}_{study_id}_{year}_{timepoint}.csv

    Since both level_name and study_id can contain underscores, we strip the
    known prefix and the known suffix (_{year}_{timepoint}.csv) and then take
    everything up to the last underscore-separated token as the level name
    (the last token before _{year}_{timepoint} is the study_id).
    """
    # e.g. "agg_yield_estimates_Kenya_Three_Counties_Counties_YearlyTotals_kenya-prelim_2021_T-0.csv"
    base = os.path.basename(filename)
    prefix = "agg_yield_estimates_"
    suffix = f"_{year}_{timepoint}.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    # middle = "Kenya_Three_Counties_Counties_YearlyTotals_kenya-prelim"
    middle = base[len(prefix):-len(suffix)]
    # The last underscore-separated segment is the study_id (sanitized, so no underscores in it)
    # e.g. "kenya-prelim" — split off the last segment
    parts = middle.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]  # level name
    return middle


def get_avaiable_agg_levels(base_dir):
    all_agg_levels = []
    for year in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for timepoint in os.listdir(year_path):
            tp_path = os.path.join(year_path, timepoint)
            if not os.path.isdir(tp_path):
                continue

            preds_pattern = os.path.join(base_dir, year, timepoint, "agg_yield_estimates_*_*.csv")
            agg_preds_files = glob(preds_pattern)
            agg_levels = [
                _extract_agg_level_name(f, year, timepoint)
                for f in agg_preds_files
            ]
            all_agg_levels.extend([lvl for lvl in agg_levels if lvl is not None])

    return list(set(all_agg_levels))


def collect_files(base_dir, agg_lvl_name, timepoint):
    pred_paths = {}
    gt_paths = {}

    # Collect all aggregated predictions at this agg lvl & timepoint
    for year in os.listdir(base_dir):
        preds_pattern = os.path.join(base_dir, year, timepoint, f"agg_yield_estimates_{agg_lvl_name}_*_{year}_{timepoint}.csv")
        agg_preds_files = glob(preds_pattern)
        if len(agg_preds_files) > 1:
            raise Exception(
                f"More than one aggregated yield estimates file detected for {year}, {timepoint}, {agg_lvl_name}."
            )

        if len(agg_preds_files) == 1:
            pred_paths[year] = agg_preds_files[0]

        gt_path = os.path.join(base_dir, year, f"referencedata_{agg_lvl_name}-{year}.csv")
        if os.path.exists(gt_path):
            gt_paths[year] = gt_path

    return pred_paths, gt_paths


def merge_preds_gt_yearly(preds_paths, gt_paths):
    yearly_dfs = []
    for year, year_pred_path in preds_paths.items():
        df_pred = pd.read_csv(year_pred_path)

        # Rename any existing 'year' column to avoid conflicts
        if "year" in df_pred.columns:
            df_pred["year_original"] = df_pred["year"]
            df_pred.drop(columns=["year"], inplace=True)

        df_pred["year"] = year

        if year in gt_paths:
            df_gt = pd.read_csv(gt_paths[year])
            if "year" in df_gt.columns:
                df_gt["year_ref_original"] = df_gt["year"]
                df_gt.drop(columns=["year"], inplace=True)
            df_merged = pd.merge(df_pred, df_gt, on="region")  # Inner join
            yearly_dfs.append(df_merged)
        else:
            yearly_dfs.append(df_pred)

    return yearly_dfs


def aggregate_years(base_dir, agg_lvl_name, timepoint):
    preds_paths, gt_paths = collect_files(base_dir, agg_lvl_name, timepoint)
    if not preds_paths:
        return pd.DataFrame({})

    yearly_merged_dfs = merge_preds_gt_yearly(preds_paths, gt_paths)

    all_years_df = pd.concat(yearly_merged_dfs)
    all_years_df.sort_values(["year", "region"], inplace=True)

    return all_years_df


@click.command()
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(exists=True),
    help="Basedirectory with subdirectories being the years, then timepoints.",
)
@click.option("--output-suffix", type=str, help="A unique suffix for the output.")
def main(base_dir: str, output_suffix: str):
    agg_lvls = get_avaiable_agg_levels(base_dir)
    timepoints = get_available_timepoints(base_dir)

    for agg_lvl_name in agg_lvls:
        for timepoint in timepoints:
            agg_df = aggregate_years(base_dir, agg_lvl_name, timepoint)
            out_file = os.path.join(base_dir, f"all_predictions_{output_suffix}_{agg_lvl_name}_{timepoint}.csv")
            agg_df.to_csv(out_file)


if __name__ == "__main__":
    main()
