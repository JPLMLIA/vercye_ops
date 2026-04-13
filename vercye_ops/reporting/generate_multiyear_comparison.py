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


def _detect_cross_year(df):
    """Detect if dates span a cross-year season (e.g. Sept→Mar)."""
    months = df["Date"].dt.month
    return (months >= 7).any() and (months <= 6).any()


def _safe_replace_year(d, target_year):
    """Replace year, handling leap-day dates by shifting to Feb 28."""
    try:
        return d.replace(year=target_year)
    except ValueError:
        # Feb 29 in a leap year Feb 28 in the target non-leap year
        return d.replace(year=target_year, day=28)


def _assign_plot_dates(df, cross_year):
    """Map dates to a continuous reference period for plotting.

    For cross-year seasons (e.g. Sept→Mar), late-year months stay in 2000
    and early-year months shift to 2001 so the curve is continuous.
    """
    if cross_year:
        df["PlotDate"] = df["Date"].apply(
            lambda d: _safe_replace_year(d, 2001) if d.month <= 6 else _safe_replace_year(d, 2000)
        )
    else:
        df["PlotDate"] = df["Date"].apply(lambda d: _safe_replace_year(d, 2000))
    return df


def _make_month_ticks(cross_year):
    """Return tick values and labels spanning the actual plot date range."""
    if cross_year:
        month_starts = pd.date_range(start="2000-07-01", end="2001-06-30", freq="MS")
    else:
        month_starts = pd.date_range(start="2000-01-01", end="2000-12-31", freq="MS")
    return month_starts, month_starts.strftime("%b")


def plot_lai_means_figure(input_dir, timepoint, years, lai_agg_type, adjusted):
    combined = []
    true_years = []
    for year in years:
        for fp in load_lai_files(os.path.join(input_dir, year, timepoint)):
            df, region, _ = parse_lai_file(fp, lai_agg_type, adjusted)
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
            df["Year"] = df["Date"].dt.year
            df["Region"] = region
            true_years.extend(df["Date"].dt.year.unique())
            combined.append(df)

    if not combined:
        return None

    full_df = pd.concat(combined, ignore_index=True)
    cross_year = _detect_cross_year(full_df)
    full_df = _assign_plot_dates(full_df, cross_year)

    col = "LAI " + ("Mean" if lai_agg_type == "mean" else "Median")
    if adjusted:
        col += " Adjusted"

    fig = go.Figure()
    added_years = []
    for year in sorted(set(map(int, true_years))):
        df_y = full_df[full_df["Year"] == year]
        if df_y.empty:
            continue
        m = df_y.groupby("PlotDate")[col].mean().reset_index()
        m = m.sort_values("PlotDate")
        fig.add_trace(
            go.Scatter(
                x=m["PlotDate"],
                y=m[col],
                mode="lines",
                name=str(year),
                legendgroup=f"y{year}",
                line=dict(width=4, color=mean_palette[len(added_years) % len(mean_palette)]),
                opacity=1,
                visible=True,
                showlegend=True,
                hovertemplate="Date: %{x|%d/%m}<br>Mean LAI: %{y:.2f}<extra></extra>",
            )
        )
        added_years.append(year)

    tick_vals, tick_text = _make_month_ticks(cross_year)

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
        cross_year = _detect_cross_year(dfy)
        dfy = _assign_plot_dates(dfy, cross_year)

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

        tick_vals, tick_text = _make_month_ticks(cross_year)

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


def load_obs_preds(input_dir, timepoint, years, agg_levels):
    results = {}
    for lvl in agg_levels:
        all_preds, all_preds_years, preds_for_obs, all_obs, all_obs_years = [], [], [], [], []
        for year in years:
            base = os.path.join(input_dir, year, timepoint)
            est = glob(os.path.join(base, f"agg_yield_estimates_{lvl}_*.csv"))
            if not est:
                continue
            if len(est) > 1:
                raise ValueError(f"Multiple yield estimate files found for {year} at level {lvl}: {est}")
            preds_df = load_csv(est[0])
            all_preds.extend(preds_df["mean_yield_kg_ha"])
            all_preds_years.extend([year] * len(preds_df))
            val = glob(os.path.join(input_dir, year, f"referencedata_{lvl}-*.csv"))
            if val:
                data = get_preds_obs(est[0], val[0])
                all_obs.extend(data["obs"])
                preds_for_obs.extend(data["preds"])
                all_obs_years.extend([year] * len(data["obs"]))
        results[lvl] = {
            "only_preds": (all_preds, all_preds_years),
            "obs_preds": (all_obs, preds_for_obs, all_obs_years),
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
    middle = base[len(prefix):-len(suffix)]
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

        # Two stacked versions
        img_std_html = (
            f'<img src="{plot_std_src}" alt="Evaluation {lvl} {year}" class="img-fit rounded ver-{group_id}-std" style="display:;">'
            if plot_std_src
            else ""
        )
        img_npc_html = (
            f'<img src="{plot_npc_src}" alt="Evaluation {lvl} {year} (no-pixel)" class="img-fit rounded ver-{group_id}-npc" style="display:none;">'
            if plot_npc_src
            else ""
        )

        tbl_std_html = f"<div class='ver-{group_id}-std' style='display:;'>{table_std}</div>"
        tbl_npc_html = f"<div class='ver-{group_id}-npc' style='display:none;'>{table_npc}</div>" if has_alt else ""

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

    toggle_btn = f"""
        <button class='btn btn-sm btn-outline-dark ml-3' type='button'
        onclick="
            document.querySelectorAll('.ver-{group_id}-std').forEach(function(e){chr(123)}e.style.display = (e.style.display==='none'?'':'none');{chr(125)});
            document.querySelectorAll('.ver-{group_id}-npc').forEach(function(e){chr(123)}e.style.display = (e.style.display==='none'?'':'none');{chr(125)});
        ">
        Toggle no-pixel conversion
        </button>
    """

    return (
        f"""
    <div class="d-flex align-items-center mb-2">
      <strong>Yearly evaluation ({lvl})</strong>
       {toggle_btn if has_alt_any else ""}
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
            all_preds, preds_years = data["only_preds"]
            if len(all_preds) == 0:
                continue

            pred_fig = create_predictions_plot(all_preds, preds_years)
            pred_html = pio.to_html(pred_fig, include_plotlyjs="cdn", full_html=False)
            multiyear_metrics_html = "<p><em>No ground-truth available for multiyear metrics.</em></p>"

            yearly_html_for_lvl, imgs_used, name_map_lvl, _has_alt = render_yearly_eval_html(yearly_eval_data, lvl, tp)
            referenced_images.extend(imgs_used)
            desired_name_map.update(name_map_lvl)

            # Wrap yearly evaluation in a collapsed section if it exists
            eval_section = ""
            if "No yearly evaluation available" not in yearly_html_for_lvl:
                eval_collapse_id = f"collapse_eval_{sanitize(tp)}_{sanitize(lvl)}"
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

            obs, preds, yrs = data["obs_preds"]
            if obs:
                scatter = create_scatter_plot(preds, obs, yrs)
                scatter_html = pio.to_html(scatter, include_plotlyjs="cdn", full_html=False)
                metrics = compute_metrics(np.array(preds), np.array(obs))
                metrics_rows = "".join(f"<tr><th scope='row'>{k}</th><td>{v:.3f}</td></tr>" for k, v in metrics.items())
                multiyear_metrics_html = f"""
                <div class='metrics-table mb-3'>
                    <strong>Metrics ({lvl}):</strong>
                    <div class="table-wrap">
                      <table class='table table-sm table-bordered mt-2'>
                        <thead class='thead-light'><tr><th>Metric</th><th>Value</th></tr></thead>
                        <tbody>{metrics_rows}</tbody>
                      </table>
                    </div>
                </div>
                {scatter_html}
                """

            content.append(
                f"""
                <div class='card mb-4'>
                  <div class='card-header'><h3>{tp} - Predictions {lvl}</h3></div>
                  <div class='card-body plot-container'>
                    {pred_html}
                    {eval_section}
                    {multiyear_metrics_html}
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
