import os
from glob import glob
import click
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from vercye_ops.matching_sim_real.evaluate_yield_estimates import (
    compute_metrics, create_scatter_plot, get_preds_obs, load_csv
)
from vercye_ops.matching_sim_real.generate_lai_plot import load_lai_files, parse_lai_file

# Use Plotly's qualitative palette
color_palette = px.colors.qualitative.Plotly
mean_palette  = px.colors.qualitative.Set1


# HTML template with Bootstrap for a modern, responsive layout
# Note: double braces {{ }} in CSS to escape Python formatting
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Multiyear Interactive Summary</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ background-color: #f8f9fa; font-family: 'Arial', sans-serif; padding: 20px; }}
        h1, h2, h3 {{ color: #343a40; }}
        .card {{ box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .plot-container {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="my-4 text-center">Multiyear Interactive Summary</h1>
        {content}
    </div>
</body>
</html>
"""


def get_available_years(input_dir):
    return sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])


def get_available_timepoints(reference_year_dir):
    return sorted([d for d in os.listdir(reference_year_dir) if os.path.isdir(os.path.join(reference_year_dir, d))])


def plot_lai_figure(input_dir, timepoint, years, lai_agg_type, adjusted):
    combined = []
    for year in years:
        for fp in load_lai_files(os.path.join(input_dir, year, timepoint)):
            df, region, _ = parse_lai_file(fp, lai_agg_type)
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            start = df['Date'].min()
            df['DaysAfterStart'] = (df['Date'] - start).dt.days
            df['Year']   = int(year)
            df['Region'] = region
            combined.append(df)
    full_df = pd.concat(combined, ignore_index=True)

    col = 'LAI ' + ('Mean' if lai_agg_type=='mean' else 'Median')
    if adjusted:
        col += ' Adjusted'

    fig = go.Figure()
    year_traces = {y: [] for y in years}

    for (region, year), grp in full_df.groupby(['Region','Year']):
        idx = len(fig.data)
        fig.add_trace(go.Scatter(
            x=grp['DaysAfterStart'], y=grp[col], mode='lines',
            name=f"{region} {year}",
            legendgroup=str(year),
            line=dict(width=2, color=color_palette[int(year) % len(color_palette)]),
            opacity=0.5,
            showlegend=True,   # hide from legend to avoid clutter
            visible=False
        ))
        year_traces[str(year)].append(idx)

    for i, year in enumerate(years):
        df_y = full_df[full_df['Year']==int(year)]
        if df_y.empty:
            continue
        m = df_y.groupby('DaysAfterStart')[col].mean().reset_index()
        fig.add_trace(go.Scatter(
            x=m['DaysAfterStart'], y=m[col],
            mode='lines',
            name=f"{year} Mean",
            legendgroup=str(year),
            line=dict(width=6, color=mean_palette[i % len(mean_palette)]),
            opacity=1,
            showlegend=True,
            visible=True
        ))

    year_buttons = []
    for year, idxs in year_traces.items():
        if not idxs:
            continue
        year_buttons.append(dict(
            label=year,
            method='restyle',
            args=[{'visible': True}, idxs],
            args2=[{'visible': False}, idxs]
        ))

    fig.update_layout(
        title=dict(text=f"{col} by Days after Simulation Start", x=0.5),
        xaxis_title='Days after Simulation Start',
        yaxis_title=f'LAI {"Adjusted" if adjusted else "Non-Adjusted"}',
        template='plotly_white',
        margin=dict(l=40, r=40, t=80, b=40),
        height=600,
        font=dict(family='Arial, sans-serif', size=12),
        updatemenus=[
            dict(
                type='buttons',
                direction='down',
                buttons=year_buttons,
                x=1.1, y=1.1,
                showactive=False,
                bgcolor='white',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=12)
            )
        ]
    )

    global_start = full_df['Date'].min().strftime('%d/%m/%Y')
    fig.add_annotation(
        x=0, y=1.05, xref='x', yref='paper',
        text=f"Simulation start: {global_start}",
        showarrow=False, font=dict(size=12), align='left'
    )

    return fig

def load_obs_preds(input_dir, timepoint, years, agg_levels):
    results = {}
    for lvl in agg_levels:
        all_obs, all_preds, yrs = [], [], []
        for year in years:
            base = os.path.join(input_dir, year, timepoint)
            est = glob(os.path.join(base, f'agg_yield_estimates_{lvl}*.csv'))
            if not est:
                continue
            preds_df = load_csv(est[0])
            all_preds.extend(preds_df['mean_yield_kg_ha'])
            yrs.extend([year]*len(preds_df))
            val = glob(os.path.join(input_dir, year, f'groundtruth-{lvl}*.csv'))
            if val:
                data = get_preds_obs(est[0], val[0])
                all_obs.extend(data['obs'])
        results[lvl] = (all_obs, all_preds, yrs)
    return results


def create_predictions_plot(preds, years):
    df = pd.DataFrame({'Predictions': preds, 'Year': years})
    fig = go.Figure()
    for yr, grp in df.groupby('Year'):
        fig.add_trace(
            go.Violin(
                x=[yr]*len(grp), y=grp['Predictions'], name=yr,
                box_visible=True, meanline_visible=True
            )
        )
    fig.update_layout(
        title=dict(text='Yield Predictions Distribution by Year from all simulation regions.', x=0.5),
        template='plotly_white', xaxis_title='Year', yaxis_title='Yield (kg/ha)',
        margin=dict(l=40, r=40, t=60, b=40), height=500,
        font=dict(family='Arial, sans-serif', size=12)
    )
    return fig


def identify_eval_levels(input_dir, years):
    lvls = set(['primary'])
    for y in years:
        files = glob(os.path.join(input_dir, y, 'groundtruth*.csv'))
        for f in files:
            lvl = os.path.basename(f).split('-')[1]
            lvls.add(lvl)
    return sorted(lvls)


@click.command()
@click.option('--input-dir', type=click.Path(exists=True), required=True)
@click.option('--lai-agg-type', type=click.Choice(['mean', 'median']), default='mean')
@click.option('--adjusted', is_flag=True)
@click.option('--output-file', type=click.Path(), required=True)
def main(input_dir, lai_agg_type, adjusted, output_file):
    years = get_available_years(input_dir)
    reference = os.path.join(input_dir, years[0])
    timepoints = get_available_timepoints(reference)
    agg_levels = identify_eval_levels(input_dir, years)

    content = []
    for tp in timepoints:
        # LAI Plot
        lai_fig = plot_lai_figure(input_dir, tp, years, lai_agg_type, adjusted)
        lai_html = pio.to_html(lai_fig, include_plotlyjs='cdn', full_html=False)
        content.append(
            f"""
            <div class='card mb-4'>
              <div class='card-header'><h2>{tp} - LAI</h2></div>
              <div class='card-body plot-container'>
                {lai_html}
                <p><em>Click on the 'year' buttons (top right) to toggle visibility of all individual LAI traces from that year (this might take a few seconds to load).</br>
                Notice: You might have to click twice.</em></p>
            </div>
            </div>
            """
        )
        # Predictions + Metrics per aggregation level
        obs_preds = load_obs_preds(input_dir, tp, years, agg_levels)
        for lvl, (obs, preds, yrs) in obs_preds.items():
            pred_fig = create_predictions_plot(preds, yrs)
            pred_html = pio.to_html(pred_fig, include_plotlyjs='cdn', full_html=False)
            metrics_html = "<p><em>No ground-truth available for metrics.</em></p>"
            if obs:
                scatter = create_scatter_plot(preds, obs, yrs)
                scatter_html = pio.to_html(scatter, include_plotlyjs='cdn', full_html=False)
                metrics = compute_metrics(preds, obs)
                metrics_html = f"""<pre><strong>Metrics ({lvl}):</strong>\n{metrics}</pre>{scatter_html}"""

            content.append(
                f"""
                <div class='card mb-4'>
                  <div class='card-header'><h3>{tp} - Predictions {lvl}</h3></div>
                  <div class='card-body plot-container'>{pred_html}{metrics_html}</div>
                </div>
                """
            )

    # Render full HTML and write
    full_html = HTML_TEMPLATE.format(content=''.join(content))
    with open(output_file, 'w') as f:
        f.write(full_html)

if __name__ == '__main__':
    main()