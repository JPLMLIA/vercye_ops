"""match_sim_real_v2 — principled sim/real LAI matcher (Ukraine yield-improvement).

Replaces the original quantile-filter + senescence-up-bias + drought-branch selection
with a single softmax-weighted whole-curve similarity:

  1. Align each APSIM simulation's daily LAI curve to the observed (smoothed, crop-adjusted)
     remotely-sensed LAI curve on shared dates.
  2. Score each sim by a magnitude-aware whole-curve similarity to the observed curve
     (default: negative RMSE over the aligned season).
  3. Convert similarities to weights via softmax(score / T) and take the
     PROBABILITY-WEIGHTED MEAN of per-sim max yields as the regional yield estimate.

Motivation (validated empirically on Ukraine 2020-2024, Mykolaiv+Poltava):
  - APSIM's intrinsic LAI->yield relationship is essentially unbiased vs real field yields
    (real yield≈2165+586·peakLAI vs APSIM≈1970+607·peakLAI).
  - The original method's Step-4 (pick HIGHEST senescence-LAI quantile) selects sims ABOVE
    that line -> systematic over-prediction, catastrophic in low-LAI/cloudy years.
  - The `drought_threshold` "3 lowest-yield sims" branch is crude and overshoots.
  - A softmax-weighted whole-curve match lands on the unbiased LAI->yield line, tracks
    inter-annual variation smoothly, and needs no per-year hacks. Peak-LAI alone is a weak
    yield predictor (within-ensemble r~0.4-0.7); the whole curve / AUC carry more signal.

Output contract matches the original script (sim_matches.csv + conversion_factor.csv with the
same key columns) so the downstream pipeline (mean-anchor conversion, reporting) is unchanged.
"""
import click
import numpy as np
import pandas as pd
from scipy.special import softmax

from vercye_ops.utils.init_logger import get_logger
from vercye_ops.matching_sim_real.utils import load_simulation_data

logger = get_logger()


def whole_curve_scores(merged, sim_ids, lai_col, rs_col, metric):
    """Return per-sim similarity score and (max_yield, max_sim_lai, max_sim_lai_date)."""
    scores = {}
    info = {}
    for sid in sim_ids:
        sub = merged.loc[merged['SimulationID'] == sid]
        sim = sub[lai_col]
        obs = sub[rs_col]
        max_yield = sub['Yield'].max()
        max_sim_lai = sim.max()
        max_sim_lai_date = sim.idxmax() if sim.notna().any() else pd.NaT
        info[sid] = (max_yield, max_sim_lai, max_sim_lai_date)
        # align on dates where BOTH sim and obs are present
        m = sim.notna() & obs.notna()
        if m.sum() < 5:
            scores[sid] = np.nan
            continue
        s = sim[m].to_numpy(dtype=float)
        o = obs[m].to_numpy(dtype=float)
        if metric == 'neg_rmse':
            scores[sid] = -np.sqrt(np.mean((s - o) ** 2))
        elif metric == 'neg_auc_diff':
            scores[sid] = -abs(s.sum() - o.sum())
        elif metric == 'cosine':
            scores[sid] = float(s.dot(o) / (np.linalg.norm(s) * np.linalg.norm(o) + 1e-9))
        else:
            raise ValueError(f'Unknown metric {metric}')
    return scores, info


def match_simulations(rs_lai_csv, db_path, sim_matches_output_fpath, conversion_factor_output_fpath,
                      crop_name, use_adjusted, lai_agg_type, metric, temperature, verbose, **_ignored):
    if verbose:
        logger.setLevel('INFO')

    crop_name_c = crop_name.capitalize()
    lai_col = f'{crop_name_c}.Leaf.LAI'

    rs_df = pd.read_csv(rs_lai_csv, index_col='Date', parse_dates=['Date'], dayfirst=True)
    rs_agg = 'Mean' if lai_agg_type.lower() == 'mean' else 'Median'
    rs_col = f'LAI {rs_agg} Adjusted' if use_adjusted else f'LAI {rs_agg}'
    logger.info(f'Using {rs_col} for RS data; metric={metric}, T={temperature}')

    sim_df = load_simulation_data(db_path, crop_name)
    sim_ids = sim_df['SimulationID'].unique()

    merged = rs_df.merge(sim_df, how='right', left_index=True, right_index=True)

    scores, info = whole_curve_scores(merged, sim_ids, lai_col, rs_col, metric)

    res = pd.DataFrame({
        'SimulationID': list(sim_ids),
        'Similarity': [scores[s] for s in sim_ids],
        'Max_Yield': [info[s][0] for s in sim_ids],
        'Max_Sim_LAI': [info[s][1] for s in sim_ids],
        'Max_Sim_LAI_Date': [info[s][2] for s in sim_ids],
    }).set_index('SimulationID').sort_index()

    valid = res['Similarity'].notna()
    if valid.sum() == 0:
        logger.error('No simulations had sufficient overlap with the RS LAI window; yield set to 0.')
        weighted_yield = 0.0
        res['Weight'] = 0.0
    else:
        sc = res.loc[valid, 'Similarity'].to_numpy(dtype=float)
        w = softmax(sc / temperature)
        res['Weight'] = 0.0
        res.loc[valid, 'Weight'] = w
        weighted_yield = float(np.sum(w * res.loc[valid, 'Max_Yield'].to_numpy(dtype=float)))

    # "matched" set = sims carrying the top 95% of probability mass (for reporting / save_matched_sims)
    res = res.sort_values('Weight', ascending=False)
    res['StepFilteredOut'] = pd.NA
    cum = res['Weight'].cumsum()
    keep = cum <= 0.95
    keep.iloc[0] = True  # always keep the top sim
    res.loc[~keep, 'StepFilteredOut'] = 1
    res = res.sort_index()
    res.to_csv(sim_matches_output_fpath)
    logger.info(f'Saved sim matches to {sim_matches_output_fpath}; weighted yield={weighted_yield:.1f}')

    # Conversion factor file (mean-anchor conversion only needs apsim_mean_yield_estimate_kg_ha;
    # other columns kept for reporting/back-compat).
    max_rs_row = rs_df.loc[rs_df[rs_col].idxmax()] if rs_df[rs_col].notna().any() else None
    max_rs_lai = float(max_rs_row[rs_col]) if max_rs_row is not None else 0.0
    max_rs_lai_date = max_rs_row.name if max_rs_row is not None else pd.NaT
    conversion_factor = 0.0 if max_rs_lai == 0 else weighted_yield / max_rs_lai

    matched = res[res['StepFilteredOut'].isna()]
    conv = pd.DataFrame([{
        'apsim_mean_yield_estimate_kg_ha': weighted_yield,
        'max_rs_lai': max_rs_lai,
        'max_rs_lai_date': max_rs_lai_date,
        'conversion_factor': conversion_factor,
        'apsim_max_matched_lai': matched['Max_Sim_LAI'].max() if len(matched) else np.nan,
        'apsim_max_matched_lai_date': pd.NaT,
        'apsim_max_all_lai': res['Max_Sim_LAI'].max(),
        'apsim_max_all_lai_date': pd.NaT,
        'apsim_matched_std_yield_estimate_kg_ha': matched['Max_Yield'].std() if len(matched) else np.nan,
        'apsim_all_std_yield_estimate_kg_ha': res['Max_Yield'].std(),
        'apsim_matched_maxlai_std': matched['Max_Sim_LAI'].std() if len(matched) else np.nan,
        'apsim_all_maxlai_std': res['Max_Sim_LAI'].std(),
        'matching_method': f'softmax_{metric}_T{temperature}',
        'n_effective_sims': float(1.0 / np.sum(res['Weight'] ** 2)) if res['Weight'].sum() > 0 else 0.0,
    }])
    conv.to_csv(conversion_factor_output_fpath, index=False)
    logger.info(f'Conversion factor file saved to {conversion_factor_output_fpath}')


@click.command()
@click.option('--rs_lai_csv', required=True, type=click.Path(exists=True))
@click.option('--db_path', required=True, type=click.Path(exists=True))
@click.option('--sim_matches_output_fpath', required=True, type=click.Path())
@click.option('--conversion_factor_output_fpath', required=True, type=click.Path())
@click.option('--crop_name', required=True, type=click.Choice(['wheat', 'maize']))
@click.option('--use_adjusted', is_flag=True)
@click.option('--lai_agg_type', required=True, type=click.Choice(['mean', 'median']))
@click.option('--metric', default='neg_rmse', type=click.Choice(['neg_rmse', 'neg_auc_diff', 'cosine']))
@click.option('--temperature', default=0.03, type=float, help='softmax temperature (lower=sharper); 0.03 keeps effective sample ~80-90 sims (validated on Ukraine reliable targets)')
@click.option('--n_jobs', default=10)  # accepted for CLI compatibility; unused
@click.option('--drought_threshold', default=0.0, type=float)  # accepted, unused
@click.option('--senescence_lai_quantile', default=0.0, type=float)  # accepted, unused
@click.option('--lai_gap_quantile', default=0.0, type=float)  # accepted, unused
@click.option('--green_lai_rmse_quantile', default=0.0, type=float)  # accepted, unused
@click.option('--verbose', is_flag=True)
def cli(**kwargs):
    match_simulations(**kwargs)


if __name__ == '__main__':
    cli()
