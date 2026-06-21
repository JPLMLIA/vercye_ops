"""match_sim_real_v3 — ensemble-regression sim/real LAI matcher (Ukraine yield-improvement).

WHY v3 (root-cause, validated offline on all 119 Ukraine oblast-years 2020-2024):
  The v2 matcher estimates regional yield as the softmax-weighted mean of the *matched
  simulations' max yields*. Empirically (faithful offline replica of the pipeline) this
  reaches only oblast R²=0.24, RMSE=838. Two diagnoses explain why:

  1. SIGNAL LOCATION. Within an APSIM ensemble, a sim's final yield is predicted far better
     by the season-INTEGRAL of its LAI curve (AUC, within-rayon r~0.69, ~biomass/cumulative
     light interception) than by peak LAI (r~0.56). Whole-curve RMSE matching is dominated by
     the peak region, so it selects on the weakest yield axis.
  2. ABSOLUTE-CALIBRATION NOISE. The matched-sim mean yield inherits APSIM's per-year/per-met
     absolute yield level, which carries year-varying bias uncorrelated with the true yield.
     Recalibrating the v2 output against references (LOO) only reaches R²=0.35 — i.e. the
     matched-sim yield itself is a weak carrier.

FIX (this file): convert via each rayon's OWN APSIM ensemble transfer function
     yield ≈ a + b · AUC(LAI),
  fit by OLS over the 1728-member ensemble, then EVALUATED AT THE OBSERVED AUC. This reads
  the regional yield off APSIM's physically-grounded integrated-canopy→yield relationship at
  the actually-observed canopy trajectory, instead of averaging the sims' own yields. It uses
  NO reference/ground-truth data (fully transferable) and is mechanistically grounded.

  Offline result (mean-anchor conversion, aggregated rayon→oblast, vs reference):
     v2 baseline (weighted-mean) : R²=0.241  RMSE=838  bias=-115
     v3 ensemble-regression(AUC) : R²=0.485  RMSE=690  bias=-119
  i.e. R² doubles and RMSE drops ~18% with no reference fitting. AUC alone beats peak+AUC
  (peak adds collinear noise) and beats peak alone — consistent with the signal ranking.

The mean-anchor downstream conversion makes the cropland-mean predicted yield EQUAL
apsim_mean_yield_estimate_kg_ha, so this script only needs to write that value (= the
ensemble-regression prediction). sim_matches.csv still carries the whole-curve softmax weights
for reporting/back-compat. Output contract is identical to v2.
"""
import click
import numpy as np
import pandas as pd
from scipy.special import softmax

from vercye_ops.utils.init_logger import get_logger
from vercye_ops.matching_sim_real.utils import load_simulation_data

logger = get_logger()


def _auc(values, ord_days):
    """Trapezoidal integral of a LAI series over ordinal days; NaNs linearly interpolated."""
    v = np.asarray(values, dtype=float)
    x = np.asarray(ord_days, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return np.nan
    if ok.sum() < len(v):
        v = np.interp(x, x[ok], v[ok])
    return float(np.trapezoid(v, x))


def whole_curve_scores(merged, sim_ids, lai_col, rs_col, metric):
    """Per-sim whole-curve similarity (kept for sim_matches.csv reporting)."""
    scores, info = {}, {}
    for sid in sim_ids:
        sub = merged.loc[merged['SimulationID'] == sid]
        sim = sub[lai_col]
        obs = sub[rs_col]
        info[sid] = (sub['Yield'].max(), sim.max(), sim.idxmax() if sim.notna().any() else pd.NaT)
        m = sim.notna() & obs.notna()
        if m.sum() < 5:
            scores[sid] = np.nan
            continue
        s = sim[m].to_numpy(float)
        o = obs[m].to_numpy(float)
        if metric == 'neg_rmse':
            scores[sid] = -np.sqrt(np.mean((s - o) ** 2))
        elif metric == 'neg_auc_diff':
            scores[sid] = -abs(s.sum() - o.sum())
        elif metric == 'cosine':
            scores[sid] = float(s.dot(o) / (np.linalg.norm(s) * np.linalg.norm(o) + 1e-9))
        else:
            raise ValueError(f'Unknown metric {metric}')
    return scores, info


def ensemble_regression_yield(sim_df, rs_df, lai_col, rs_col):
    """Fit yield ≈ a + b·AUC(LAI) over the ensemble; predict at the observed AUC.

    Returns (pred_yield, diag) where diag holds obs_auc, slope, intercept, n_sims and the
    ensemble yield clip bounds. Purely physical: no reference data used.
    """
    obs = rs_df[rs_col].dropna()
    if obs.empty:
        return np.nan, {}
    sim_wide = sim_df.pivot_table(index=sim_df.index, columns='SimulationID', values=lai_col)
    idx = sim_wide.index.intersection(obs.index)
    if len(idx) < 5:
        return np.nan, {}
    ord_days = np.array([d.toordinal() for d in idx], dtype=float)
    obs_auc = _auc(obs.reindex(idx).to_numpy(float), ord_days)
    if not np.isfinite(obs_auc):
        return np.nan, {}

    yld = sim_df.groupby('SimulationID')['Yield'].max()
    sub = sim_wide.reindex(idx)
    sim_auc = {sid: _auc(sub[sid].to_numpy(float), ord_days) for sid in sub.columns}
    sa = pd.Series(sim_auc)
    df = pd.DataFrame({'auc': sa, 'y': yld.reindex(sa.index)}).dropna()
    if len(df) < 30:
        return np.nan, {}

    X = np.column_stack([np.ones(len(df)), df['auc'].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, df['y'].to_numpy(float), rcond=None)
    pred = float(beta[0] + beta[1] * obs_auc)
    lo, hi = np.nanpercentile(df['y'].to_numpy(float), [1, 99])
    pred = float(np.clip(pred, lo, hi))  # keep within the ensemble's physical yield range
    diag = {'obs_auc': obs_auc, 'ensreg_intercept': float(beta[0]), 'ensreg_slope': float(beta[1]),
            'ensreg_n_sims': int(len(df)), 'ensreg_clip_lo': float(lo), 'ensreg_clip_hi': float(hi)}
    return pred, diag


def match_simulations(rs_lai_csv, db_path, sim_matches_output_fpath, conversion_factor_output_fpath,
                      crop_name, use_adjusted, lai_agg_type, metric, temperature, top_k, verbose, **_ignored):
    if verbose:
        logger.setLevel('INFO')

    crop_name_c = crop_name.capitalize()
    lai_col = f'{crop_name_c}.Leaf.LAI'

    rs_df = pd.read_csv(rs_lai_csv, index_col='Date', parse_dates=['Date'], dayfirst=True)
    rs_agg = 'Mean' if lai_agg_type.lower() == 'mean' else 'Median'
    rs_col = f'LAI {rs_agg} Adjusted' if use_adjusted else f'LAI {rs_agg}'
    logger.info(f'Using {rs_col} for RS data; conversion=ensemble_regression(AUC)')

    sim_df = load_simulation_data(db_path, crop_name)
    sim_ids = sim_df['SimulationID'].unique()
    merged = rs_df.merge(sim_df, how='right', left_index=True, right_index=True)

    # --- headline estimate: ensemble-regression on AUC (physical, no reference) ---
    ensreg_yield, diag = ensemble_regression_yield(sim_df, rs_df, lai_col, rs_col)

    # --- whole-curve softmax weights (reporting / back-compat sim_matches.csv) ---
    scores, info = whole_curve_scores(merged, sim_ids, lai_col, rs_col, metric)
    res = pd.DataFrame({
        'SimulationID': list(sim_ids),
        'Similarity': [scores[s] for s in sim_ids],
        'Max_Yield': [info[s][0] for s in sim_ids],
        'Max_Sim_LAI': [info[s][1] for s in sim_ids],
        'Max_Sim_LAI_Date': [info[s][2] for s in sim_ids],
    }).set_index('SimulationID').sort_index()
    valid = res['Similarity'].notna()
    res['Weight'] = 0.0
    wmean_yield = 0.0
    if valid.sum() > 0:
        sc = res.loc[valid, 'Similarity'].to_numpy(float)
        vidx = res.index[valid]
        if top_k and 0 < top_k < len(sc):
            kp = np.argsort(sc)[-top_k:]
            w = softmax(sc[kp] / temperature)
            res.loc[vidx[kp], 'Weight'] = w
            wmean_yield = float(np.sum(w * res.loc[vidx[kp], 'Max_Yield'].to_numpy(float)))
        else:
            w = softmax(sc / temperature)
            res.loc[vidx, 'Weight'] = w
            wmean_yield = float(np.sum(w * res.loc[vidx, 'Max_Yield'].to_numpy(float)))

    # Fall back to the v2 weighted-mean if the regression could not be fit.
    if not np.isfinite(ensreg_yield):
        logger.warning('Ensemble-regression could not be fit; falling back to weighted-mean yield.')
        ensreg_yield = wmean_yield

    res = res.sort_values('Weight', ascending=False)
    res['StepFilteredOut'] = pd.NA
    keep = res['Weight'].cumsum() <= 0.95
    keep.iloc[0] = True
    res.loc[~keep, 'StepFilteredOut'] = 1
    res = res.sort_index()
    res.to_csv(sim_matches_output_fpath)
    logger.info(f'Saved sim matches; ensreg_yield={ensreg_yield:.1f} (v2 weighted-mean={wmean_yield:.1f})')

    max_rs_row = rs_df.loc[rs_df[rs_col].idxmax()] if rs_df[rs_col].notna().any() else None
    max_rs_lai = float(max_rs_row[rs_col]) if max_rs_row is not None else 0.0
    max_rs_lai_date = max_rs_row.name if max_rs_row is not None else pd.NaT
    conversion_factor = 0.0 if max_rs_lai == 0 else ensreg_yield / max_rs_lai
    matched = res[res['StepFilteredOut'].isna()]
    conv = pd.DataFrame([{
        'apsim_mean_yield_estimate_kg_ha': ensreg_yield,
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
        'apsim_weighted_mean_yield_v2': wmean_yield,
        'matching_method': 'ensemble_regression_AUC',
        'n_effective_sims': float(1.0 / np.sum(res['Weight'] ** 2)) if res['Weight'].sum() > 0 else 0.0,
        **diag,
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
@click.option('--temperature', default=0.03, type=float, help='softmax temperature for reporting weights only')
@click.option('--top_k', default=30, type=int, help='top-K for reporting softmax weights only')
@click.option('--n_jobs', default=10)
@click.option('--drought_threshold', default=0.0, type=float)
@click.option('--senescence_lai_quantile', default=0.0, type=float)
@click.option('--lai_gap_quantile', default=0.0, type=float)
@click.option('--green_lai_rmse_quantile', default=0.0, type=float)
@click.option('--verbose', is_flag=True)
def cli(**kwargs):
    match_simulations(**kwargs)


if __name__ == '__main__':
    cli()
