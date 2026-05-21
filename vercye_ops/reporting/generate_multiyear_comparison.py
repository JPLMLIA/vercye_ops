import os
import re
import shutil
import tempfile
import zipfile
from glob import glob

import click
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from vercye_ops.evaluation.evaluate_yield_estimates import compute_metrics, create_scatter_plot, get_preds_obs, load_csv
from vercye_ops.reporting.generate_lai_plot import load_lai_files, parse_lai_file

color_palette = px.colors.qualitative.Plotly
mean_palette = px.colors.qualitative.Set1


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>{title}</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <!-- Bootstrap JS deps for collapse -->
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" crossorigin="anonymous"></script>
    <style>
        body {{ background-color: #f8f9fa; font-family: 'Arial', sans-serif; padding: 20px; }}
        h1, h2, h3 {{ color: #343a40; }}
        .card {{ box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-container {{ margin-bottom: 20px; }}
        .card-body {{ overflow: hidden; }}
        .metrics-table .table {{ margin-bottom: 0; }}
        .metrics-table table td, .metrics-table table th {{ vertical-align: middle; word-break: break-word; }}
        .table-wrap {{ overflow-x: auto; }}
        .img-fit {{ width: 100%; height: auto; display:block; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="my-4 text-center">{title}</h1>
        {content}
    </div>
</body>
</html>
"""


def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def get_available_years(input_dir):
    years = []
    for entry in os.listdir(input_dir):
        path = os.path.join(input_dir, entry)
        if os.path.isdir(path) and entry.isdigit():
            years.append(entry)
    return sorted(years)


def get_available_timepoints(reference_year_dir):
    return sorted(d for d in os.listdir(reference_year_dir) if os.path.isdir(os.path.join(reference_year_dir, d)))


def _assign_plot_dates(df, season_year):
    """Anchor every date relative to its season's Jan 1 and project onto year 2000.

    The ``season_year`` is the agronomic season label, which on disk corresponds
    to the per-year subdirectory the LAI file came from. A season window
    entirely inside one calendar year (e.g. Feb-Aug 2022 under ``2022/``) maps
    cleanly into year 2000. A cross-calendar season (e.g. Sept 2023 - Mar 2024
    under ``2023/``) maps continuously from Sept 2000 to Mar 2001, so the
    PlotDate axis always increases monotonically across the season.
    """
    base = pd.Timestamp(year=2000, month=1, day=1)
    season_anchor = pd.Timestamp(year=int(season_year), month=1, day=1)
    df = df.copy()
    df["PlotDate"] = base + (df["Date"] - season_anchor)
    return df


def _make_month_ticks(plot_dates):
    """Monthly tick values + labels spanning the supplied PlotDate series."""
    s = pd.Series(plot_dates).dropna()
    if s.empty:
        return [], []
    start = s.min().to_period("M").to_timestamp()
    end = s.max().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
    month_starts = pd.date_range(start=start, end=end, freq="MS")
    return month_starts, [d.strftime("%b") for d in month_starts]


def plot_lai_means_figure(input_dir, timepoint, years, lai_agg_type, adjusted):
    combined = []
    season_years = []
    for year in years:
        season_year = int(year)
        for fp in load_lai_files(os.path.join(input_dir, year, timepoint)):
            df, region, _ = parse_lai_file(fp, lai_agg_type, adjusted)
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            df = _assign_plot_dates(df, season_year)
            df["SeasonYear"] = season_year
            df["Region"] = region
            combined.append(df)
            season_years.append(season_year)

    if not combined:
        return None

    full_df = pd.concat(combined, ignore_index=True)

    col = "LAI " + ("Mean" if lai_agg_type == "mean" else "Median")
    if adjusted:
        col += " Adjusted"

    fig = go.Figure()
    added_years = []
    for season_year in sorted(set(season_years)):
        df_y = full_df[full_df["SeasonYear"] == season_year]
        if df_y.empty:
            continue
        m = df_y.groupby("PlotDate")[col].mean().reset_index()
        m = m.sort_values("PlotDate")
        fig.add_trace(
            go.Scatter(
                x=m["PlotDate"],
                y=m[col],
                mode="lines",
                name=str(season_year),
                legendgroup=f"y{season_year}",
                line=dict(width=4, color=mean_palette[len(added_years) % len(mean_palette)]),
                opacity=1,
                visible=True,
                showlegend=True,
                hovertemplate="Date: %{x|%d/%m}<br>Mean LAI: %{y:.2f}<extra></extra>",
            )
        )
        added_years.append(season_year)

    tick_vals, tick_text = _make_month_ticks(full_df["PlotDate"])

    fig.update_layout(
        title=dict(text=f"{col} by Day-of-Year - Aggregated by Year", x=0.5),
        xaxis=dict(title="Month", tickmode="array", tickvals=tick_vals, ticktext=tick_text, type="date"),
        yaxis=dict(title=f'LAI {"Adjusted" if adjusted else "Non-Adjusted"}'),
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
        font=dict(family="Arial", size=12),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def generate_lai_year_images(input_dir, timepoint, years, lai_agg_type, adjusted, assets_dir):
    cards = []
    produced = []
    for year in sorted(map(int, years)):
        combined = []
        for fp in load_lai_files(os.path.join(input_dir, str(year), timepoint)):
            df, region, _ = parse_lai_file(fp, lai_agg_type, adjusted)
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            df["Region"] = region
            combined.append(df)
        if not combined:
            continue
        dfy = pd.concat(combined, ignore_index=True)
        dfy = _assign_plot_dates(dfy, year)

        col = "LAI " + ("Mean" if lai_agg_type == "mean" else "Median")
        if adjusted:
            col += " Adjusted"

        fig = go.Figure()
        for region, grp in dfy.groupby("Region"):
            grp = grp.sort_values("PlotDate")
            fig.add_trace(
                go.Scatter(
                    x=grp["PlotDate"],
                    y=grp[col],
                    mode="lines",
                    name=str(region),
                    line=dict(width=1),
                    opacity=0.5,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        tick_vals, tick_text = _make_month_ticks(dfy["PlotDate"])

        fig.update_layout(
            title=dict(text=f"{col} - Regions in {year}", x=0.5),
            xaxis=dict(title="Month", tickmode="array", tickvals=tick_vals, ticktext=tick_text, type="date"),
            yaxis=dict(title=f'LAI {"Adjusted" if adjusted else "Non-Adjusted"}'),
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40),
            height=400,
            font=dict(family="Arial", size=12),
            showlegend=False,
        )
        tp_tag = sanitize(timepoint)
        img_name = f"lai_{tp_tag}_{year}.png"
        img_path = os.path.join(assets_dir, img_name)
        pio.write_image(fig, img_path, format="png", scale=2)
        produced.append(("assets/" + img_name, year))
        cards.append(
            f"""
            <div class="col-lg-6 mb-4">
              <div class="card h-100">
                <div class="card-header"><h5 class="mb-0">LAI regions - {year}</h5></div>
                <div class="card-body">
                  <img src="assets/{img_name}" alt="LAI regions {year}" class="img-fit rounded">
                </div>
              </div>
            </div>
            """
        )
    if not cards:
        return "", []
    return f"<div class='row'>{''.join(cards)}</div>", produced


def _apsim_pred_column(df):
    """Return the column name to use for APSIM-only predictions, or None if neither is available."""
    if "mean_yield_kg_ha_apsim" in df.columns:
        return "mean_yield_kg_ha_apsim"
    if "apsim_mean_yield_estimate_kg_ha" in df.columns:
        return "apsim_mean_yield_estimate_kg_ha"
    return None


def load_obs_preds(input_dir, timepoint, years, agg_levels):
    """Load predictions + reference data per level, for both LAI-converted and APSIM-only views.

    Returns ``{level: {"std": {...}, "apsim": {...} | None}}`` where each inner dict has the
    same shape as before (``only_preds`` + ``obs_preds`` tuples).
    """
    results = {}
    for lvl in agg_levels:
        std_preds, std_preds_years = [], []
        std_preds_for_obs, std_obs, std_obs_years = [], [], []

        apsim_preds, apsim_preds_years = [], []
        apsim_preds_for_obs, apsim_obs, apsim_obs_years = [], [], []
        apsim_available = False

        for year in years:
            base = os.path.join(input_dir, year, timepoint)
            est = glob(os.path.join(base, f"agg_yield_estimates_{lvl}_*.csv"))
            if not est:
                continue
            if len(est) > 1:
                raise ValueError(f"Multiple yield estimate files found for {year} at level {lvl}: {est}")

            preds_df = load_csv(est[0])
            std_preds.extend(preds_df["mean_yield_kg_ha"])
            std_preds_years.extend([year] * len(preds_df))

            apsim_col = _apsim_pred_column(preds_df)
            if apsim_col:
                apsim_available = True
                apsim_preds.extend(preds_df[apsim_col])
                apsim_preds_years.extend([year] * len(preds_df))

            val = glob(os.path.join(input_dir, year, f"referencedata_{lvl}-*.csv"))
            if val:
                data_std = get_preds_obs(est[0], val[0], pixel_converted=True)
                std_obs.extend(data_std["obs"])
                std_preds_for_obs.extend(data_std["preds"])
                std_obs_years.extend([year] * len(data_std["obs"]))
                try:
                    data_apsim = get_preds_obs(est[0], val[0], pixel_converted=False)
                    apsim_obs.extend(data_apsim["obs"])
                    apsim_preds_for_obs.extend(data_apsim["preds"])
                    apsim_obs_years.extend([year] * len(data_apsim["obs"]))
                    apsim_available = True
                except ValueError:
                    # No APSIM-only column on this level - leave apsim_obs_preds empty for this year
                    pass

        results[lvl] = {
            "std": {
                "only_preds": (std_preds, std_preds_years),
                "obs_preds": (std_obs, std_preds_for_obs, std_obs_years),
            },
            "apsim": (
                {
                    "only_preds": (apsim_preds, apsim_preds_years),
                    "obs_preds": (apsim_obs, apsim_preds_for_obs, apsim_obs_years),
                }
                if apsim_available
                else None
            ),
        }
    return results


def create_predictions_plot(preds, years):
    df = pd.DataFrame({"Predictions": preds, "Year": years})
    fig = go.Figure()
    for yr, grp in df.groupby("Year"):
        fig.add_trace(
            go.Violin(x=[yr] * len(grp), y=grp["Predictions"], name=yr, box_visible=True, meanline_visible=True)
        )
    fig.update_layout(
        title=dict(text="Yield Predictions Distribution by Year from all simulation regions.", x=0.5),
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="Yield (kg/ha)",
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        font=dict(family="Arial", size=12),
    )
    return fig


def _extract_agg_level_name(filename, year, timepoint):
    """Extract aggregation level name from a filename like
    agg_yield_estimates_{level_name}_{study_id}_{year}_{timepoint}.csv
    """
    base = os.path.basename(filename)
    prefix = "agg_yield_estimates_"
    suffix = f"_{year}_{timepoint}.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    middle = base[len(prefix) : -len(suffix)]
    parts = middle.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return middle


def identify_agg_levels(input_dir, years):
    lvls = set(["primary"])
    for y in years:
        for tp_dir in glob(os.path.join(input_dir, y, "*")):
            if not os.path.isdir(tp_dir):
                continue
            timepoint = os.path.basename(tp_dir)
            files = glob(os.path.join(tp_dir, "agg_yield_estimates_*.csv"))
            for f in files:
                lvl = _extract_agg_level_name(f, y, timepoint)
                if lvl:
                    lvls.add(lvl)
    return sorted(lvls)


def load_yearly_eval_data(input_dir, timepoint, agg_levels, years):
    results = {}
    for lvl in agg_levels:
        results[lvl] = {}
        for year in years:
            base = os.path.join(input_dir, year, timepoint)

            # default
            eval_plot_std = glob(os.path.join(base, f"evaluation_{lvl}.png"))
            eval_metrics_std = glob(os.path.join(base, f"evaluation_{lvl}.csv"))

            # no-pixel-conversion
            eval_plot_npc = glob(os.path.join(base, f"evaluation_{lvl}_no-pixel-conversion.png"))
            eval_metrics_npc = glob(os.path.join(base, f"evaluation_{lvl}_no-pixel-conversion.csv"))

            if not (eval_plot_std and eval_metrics_std) and not (eval_plot_npc and eval_metrics_npc):
                continue  # nothing to show for this year/level

            if (
                len(eval_plot_std) > 1
                or len(eval_metrics_std) > 1
                or len(eval_plot_npc) > 1
                or len(eval_metrics_npc) > 1
            ):
                raise ValueError(f"Multiple eval files found for {year} at level {lvl}.")

            results[lvl][year] = {
                "std": {
                    "plot": eval_plot_std[0] if eval_plot_std else None,
                    "metrics": eval_metrics_std[0] if eval_metrics_std else None,
                },
                "npc": {
                    "plot": eval_plot_npc[0] if eval_plot_npc else None,
                    "metrics": eval_metrics_npc[0] if eval_metrics_npc else None,
                },
            }
    return results


def _metrics_csv_to_table_html(csv_path: str, title: str = "Metrics") -> str:
    df = load_csv(csv_path)
    rows = []
    if df.shape[1] == 2 and df.shape[0] >= 1:
        col_a, col_b = df.columns.tolist()
        for _, r in df.iterrows():
            k = str(r[col_a])
            v = r[col_b]
            rows.append((k, v))
    elif df.shape[0] >= 1:
        rec = df.iloc[0].to_dict()
        for k, v in rec.items():
            if k is None or str(k).strip() == "":
                continue
            rows.append((str(k), v))
    else:
        return "<p><em>No metrics found.</em></p>"
    rows_html = "".join(f"<tr><th scope='row'>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"""
    <div class='metrics-table'>
      <strong>{title}</strong>
      <div class="table-wrap">
        <table class='table table-sm table-bordered mt-2'>
          <thead class='thead-light'>
            <tr><th>Metric</th><th>Value</th></tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """


def render_yearly_eval_html(yearly_eval_data: dict, lvl: str, timepoint: str) -> tuple[str, list, dict, bool]:
    data_for_lvl = yearly_eval_data.get(lvl, {})
    if not data_for_lvl:
        return "<p><em>No yearly evaluation available.</em></p>", [], {}, False

    cards, used_imgs = [], []
    name_map = {}
    tp_tag = sanitize(timepoint)
    lvl_tag = sanitize(lvl)
    group_id = f"{tp_tag}_{lvl_tag}"

    has_alt_any = False

    for year in sorted(data_for_lvl.keys(), key=lambda y: int(y) if str(y).isdigit() else y):
        rec = data_for_lvl[year]
        std_plot = rec["std"]["plot"]
        std_csv = rec["std"]["metrics"]
        npc_plot = rec["npc"]["plot"]
        npc_csv = rec["npc"]["metrics"]

        has_alt = bool(npc_plot and npc_csv)
        has_alt_any = has_alt_any or has_alt

        # Build metrics tables
        table_std = (
            _metrics_csv_to_table_html(std_csv, title=f"Metrics ({lvl}, {year})")
            if std_csv
            else "<p><em>No metrics (std).</em></p>"
        )
        table_npc = (
            _metrics_csv_to_table_html(npc_csv, title=f"Metrics (no-pixel, {lvl}, {year})")
            if npc_csv
            else "<p><em>No metrics (no-pixel).</em></p>"
        )

        # Track/copy images and suggest names
        yr_tag = sanitize(year)

        plot_std_src = ""
        if std_plot:
            used_imgs.append(std_plot)
            ext_std = os.path.splitext(std_plot)[1] or ".png"
            suggested_std = f"evaluation_{lvl_tag}_{tp_tag}_{yr_tag}{ext_std.lower()}"
            name_map[std_plot] = suggested_std
            plot_std_src = std_plot

        plot_npc_src = ""
        if npc_plot:
            used_imgs.append(npc_plot)
            ext_npc = os.path.splitext(npc_plot)[1] or ".png"
            suggested_npc = f"evaluation_{lvl_tag}_{tp_tag}_{yr_tag}_no-pixel-conversion{ext_npc.lower()}"
            name_map[npc_plot] = suggested_npc
            plot_npc_src = npc_plot

        # Two stacked versions - share the yield-source class scheme with the rest of the section
        img_std_html = (
            f'<img src="{plot_std_src}" alt="Evaluation {lvl} {year}" class="img-fit rounded ys-{group_id}-std" style="display:;">'
            if plot_std_src
            else ""
        )
        img_npc_html = (
            f'<img src="{plot_npc_src}" alt="Evaluation {lvl} {year} (APSIM-only)" class="img-fit rounded ys-{group_id}-apsim" style="display:none;">'
            if plot_npc_src
            else ""
        )

        tbl_std_html = f"<div class='ys-{group_id}-std' style='display:;'>{table_std}</div>"
        tbl_npc_html = f"<div class='ys-{group_id}-apsim' style='display:none;'>{table_npc}</div>" if has_alt else ""

        card = f"""
        <div class="col-lg-6 mb-4">
          <div class="card h-100">
            <div class="card-header d-flex justify-content-between align-items-center">
              <h5 class="mb-0">Year {year} - {lvl}</h5>
              <span class="badge badge-primary">Per-year evaluation</span>
            </div>
            <div class="card-body">
              <div class="row">
                {img_std_html}
                {img_npc_html}
                {tbl_std_html}
                {tbl_npc_html}
              </div>
            </div>
          </div>
        </div>
        """
        cards.append(card)

    # No local toggle here - the section-wide yield-source toggle (rendered in the
    # main loop) drives `.ys-{group_id}-std` / `.ys-{group_id}-apsim` visibility for
    # both yearly cards and the multiyear stats/eval block.
    return (
        f"""
    <div class="d-flex align-items-center mb-2">
      <strong>Yearly evaluation ({lvl})</strong>
    </div>
    <div class="row">{''.join(cards)}</div>
    """,
        used_imgs,
        name_map,
        has_alt_any,
    )


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--lai-agg-type", type=click.Choice(["mean", "median"]), default="mean")
@click.option("--adjusted", is_flag=True)
@click.option(
    "--title",
    type=str,
    default="Multiyear Interactive Summary",
    help="Title for the HTML report. Enclose in quotes if it contains spaces.",
)
@click.option(
    "--output-file",
    type=click.Path(),
    required=True,
    help="Path for the resulting ZIP. If not ending with .zip, it will be appended.",
)
def main(input_dir, lai_agg_type, adjusted, title, output_file):
    years = get_available_years(input_dir)
    reference = os.path.join(input_dir, years[0])
    timepoints = get_available_timepoints(reference)
    agg_levels = identify_agg_levels(input_dir, years)

    temp_root = tempfile.mkdtemp(prefix="report_bundle_")
    assets_dir = os.path.join(temp_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    referenced_images = []
    desired_name_map = {}
    content = []

    for tp in timepoints:
        lai_mean_fig = plot_lai_means_figure(input_dir, tp, years, lai_agg_type, adjusted)
        if lai_mean_fig is not None:
            lai_html = pio.to_html(lai_mean_fig, include_plotlyjs="cdn", full_html=False)
            lai_years_html, lai_imgs = generate_lai_year_images(
                input_dir, tp, years, lai_agg_type, adjusted, assets_dir
            )

            # Collapse controls for yearly LAI images
            lai_collapse_id = f"collapse_lai_{sanitize(tp)}"
            lai_button = ""
            lai_collapse_block = ""
            if lai_years_html.strip():
                lai_button = f"""
                  <button class="btn btn-outline-primary mb-3" type="button"
                          data-toggle="collapse" data-target="#{lai_collapse_id}"
                          aria-expanded="false" aria-controls="{lai_collapse_id}">
                    Show yearly LAI region traces
                  </button>
                """
                lai_collapse_block = f"""
                  <div class="collapse" id="{lai_collapse_id}">
                    <div class="card card-body">
                      {lai_years_html}
                    </div>
                  </div>
                """

            content.append(
                f"""
                <div class='card mb-4'>
                  <div class='card-header'><h2>{tp} - LAI</h2></div>
                  <div class='card-body plot-container'>
                    {lai_html}
                    <p><em>The interactive plot shows yearly means. Use the button below to toggle per-year region traces.</em></p>
                    {lai_button}
                    {lai_collapse_block}
                  </div>
                </div>
            """
            )

        obs_preds = load_obs_preds(input_dir, tp, years, agg_levels)
        yearly_eval_data = load_yearly_eval_data(input_dir, tp, agg_levels, years)

        for lvl, data in obs_preds.items():
            std_data = data["std"]
            apsim_data = data["apsim"]

            std_preds, std_preds_years = std_data["only_preds"]
            if len(std_preds) == 0:
                continue

            tp_tag = sanitize(tp)
            lvl_tag = sanitize(lvl)
            # Shared yield-source group id across yearly cards, violin, and multiyear scatter
            group_id = f"{tp_tag}_{lvl_tag}"
            has_apsim_multi = apsim_data is not None and len(apsim_data["only_preds"][0]) > 0

            # Yearly prediction distribution (violin) - render both yield sources when available
            std_pred_fig = create_predictions_plot(std_preds, std_preds_years)
            std_pred_html = pio.to_html(std_pred_fig, include_plotlyjs="cdn", full_html=False)
            apsim_pred_html = ""
            if has_apsim_multi:
                apsim_preds_, apsim_preds_years_ = apsim_data["only_preds"]
                apsim_pred_fig = create_predictions_plot(apsim_preds_, apsim_preds_years_)
                apsim_pred_html_inner = pio.to_html(apsim_pred_fig, include_plotlyjs="cdn", full_html=False)
                apsim_pred_html = f"<div class='ys-{group_id}-apsim' style='display:none;'>{apsim_pred_html_inner}</div>"
            std_pred_html = f"<div class='ys-{group_id}-std'>{std_pred_html}</div>"

            multiyear_metrics_html_std = "<p><em>No ground-truth available for multiyear metrics.</em></p>"
            multiyear_metrics_html_apsim = ""

            yearly_html_for_lvl, imgs_used, name_map_lvl, _has_alt = render_yearly_eval_html(yearly_eval_data, lvl, tp)
            referenced_images.extend(imgs_used)
            desired_name_map.update(name_map_lvl)

            # Wrap yearly evaluation in a collapsed section if it exists
            eval_section = ""
            if "No yearly evaluation available" not in yearly_html_for_lvl:
                eval_collapse_id = f"collapse_eval_{tp_tag}_{lvl_tag}"
                eval_section = f"""
                <button class="btn btn-outline-secondary mb-3" type="button"
                        data-toggle="collapse" data-target="#{eval_collapse_id}"
                        aria-expanded="false" aria-controls="{eval_collapse_id}">
                    Show per-year evaluation (plots &amp; metrics)
                </button>
                <div class="collapse" id="{eval_collapse_id}">
                    <div class="card card-body">
                    {yearly_html_for_lvl}
                    </div>
                </div>
                """
            else:
                eval_section = yearly_html_for_lvl

            obs_std, preds_std_for_obs, yrs_std = std_data["obs_preds"]
            if obs_std:
                scatter = create_scatter_plot(preds_std_for_obs, obs_std, yrs_std)
                scatter_html = pio.to_html(scatter, include_plotlyjs="cdn", full_html=False)
                metrics = compute_metrics(np.array(preds_std_for_obs), np.array(obs_std))
                metrics_rows = "".join(f"<tr><th scope='row'>{k}</th><td>{v:.3f}</td></tr>" for k, v in metrics.items())
                multiyear_metrics_html_std = f"""
                <div class='metrics-table mb-3'>
                    <strong>Metrics ({lvl} - LAI-converted):</strong>
                    <div class="table-wrap">
                      <table class='table table-sm table-bordered mt-2'>
                        <thead class='thead-light'><tr><th>Metric</th><th>Value</th></tr></thead>
                        <tbody>{metrics_rows}</tbody>
                      </table>
                    </div>
                </div>
                {scatter_html}
                """

            if apsim_data is not None:
                obs_a, preds_a, yrs_a = apsim_data["obs_preds"]
                if obs_a:
                    scatter_a = create_scatter_plot(preds_a, obs_a, yrs_a)
                    scatter_a_html = pio.to_html(scatter_a, include_plotlyjs="cdn", full_html=False)
                    metrics_a = compute_metrics(np.array(preds_a), np.array(obs_a))
                    metrics_a_rows = "".join(
                        f"<tr><th scope='row'>{k}</th><td>{v:.3f}</td></tr>" for k, v in metrics_a.items()
                    )
                    multiyear_metrics_html_apsim = f"""
                    <div class='metrics-table mb-3'>
                        <strong>Metrics ({lvl} - APSIM-only):</strong>
                        <div class="table-wrap">
                          <table class='table table-sm table-bordered mt-2'>
                            <thead class='thead-light'><tr><th>Metric</th><th>Value</th></tr></thead>
                            <tbody>{metrics_a_rows}</tbody>
                          </table>
                        </div>
                    </div>
                    {scatter_a_html}
                    """

            multiyear_metrics_html_std = f"<div class='ys-{group_id}-std'>{multiyear_metrics_html_std}</div>"
            if multiyear_metrics_html_apsim:
                multiyear_metrics_html_apsim = (
                    f"<div class='ys-{group_id}-apsim' style='display:none;'>{multiyear_metrics_html_apsim}</div>"
                )

            # Single yield-source toggle for this whole section. Flips visibility of every
            # `.ys-{group_id}-std` / `.ys-{group_id}-apsim` element: violin, multiyear
            # metrics+scatter, and the yearly per-year cards/metrics.
            has_yearly_apsim = _has_alt
            show_apsim_toggle = has_apsim_multi or has_yearly_apsim or bool(multiyear_metrics_html_apsim)
            yield_toggle_btn = ""
            if show_apsim_toggle:
                yield_toggle_btn = f"""
                <div class='alert alert-light border d-inline-block py-2 px-3 mb-3' style='font-size:0.95em;'>
                  <strong>Yield source:</strong>
                  <span class='badge badge-info ys-{group_id}-label-std' style='display:;'>LAI-converted (pixel-level)</span>
                  <span class='badge badge-warning ys-{group_id}-label-apsim' style='display:none;'>APSIM-only (no LAI pixel conversion)</span>
                  <button class='btn btn-sm btn-outline-dark ml-3' type='button'
                  onclick="
                      ['std','apsim'].forEach(function(k){chr(123)}
                          document.querySelectorAll('.ys-{group_id}-'+k).forEach(function(e){chr(123)}e.style.display = (e.style.display==='none'?'':'none');{chr(125)});
                          document.querySelectorAll('.ys-{group_id}-label-'+k).forEach(function(e){chr(123)}e.style.display = (e.style.display==='none'?'':'none');{chr(125)});
                      {chr(125)});
                  ">
                  Switch
                  </button>
                </div>
                """

            content.append(
                f"""
                <div class='card mb-4'>
                  <div class='card-header'><h3>{tp} - Predictions {lvl}</h3></div>
                  <div class='card-body plot-container'>
                    {yield_toggle_btn}
                    {std_pred_html}
                    {apsim_pred_html}
                    {eval_section}
                    {multiyear_metrics_html_std}
                    {multiyear_metrics_html_apsim}
                  </div>
                </div>
            """
            )

    path_map = {}
    seen, used_names = set(), set()
    for p in referenced_images:
        if not p or not os.path.exists(p) or p in seen:
            continue
        seen.add(p)
        suggested = desired_name_map.get(p)
        if suggested:
            suggested = sanitize(os.path.splitext(suggested)[0]) + (os.path.splitext(suggested)[1] or ".png")
            new_name = suggested
        else:
            new_name = os.path.basename(p)
        stem, ext = os.path.splitext(new_name)
        i = 1
        while new_name in used_names:
            i += 1
            new_name = f"{stem}_{i}{ext}"
        used_names.add(new_name)
        dst = os.path.join(assets_dir, new_name)
        shutil.copy2(p, dst)
        path_map[p] = os.path.join("assets", new_name)

    html_content = "".join(content)
    for original, rel in path_map.items():
        html_content = html_content.replace(f'src="{original}"', f'src="{rel}"')

    html_filename = "report.html"
    html_path = os.path.join(temp_root, html_filename)
    full_html = HTML_TEMPLATE.format(content=html_content, title=title)
    with open(html_path, "w") as f:
        f.write(full_html)

    zip_out = output_file if output_file.lower().endswith(".zip") else f"{output_file}.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(html_path, arcname=html_filename)
        for root, _, files in os.walk(assets_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, temp_root)
                zf.write(full, arc)

    shutil.rmtree(temp_root, ignore_errors=True)
    print(f"Created bundle: {zip_out}")


if __name__ == "__main__":
    main()
