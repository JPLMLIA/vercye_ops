import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import click

from vercye_ops.matching_sim_real.utils import load_simulation_data, load_simulation_units
from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


def generate_report(apsim_filtered_fpath, rs_lai_csv_fpath, apsim_db_fpath, total_yield_csv_fpath, html_fpath=None, png_fpath=None):
    """
    Generate a plot report from APSIM and database time series data.

    Parameters
    ----------
    apsim_filtered_fpath : str
        Filepath to the filtered CSV.
    rs_lai_csv : str
        Filepath to the remotely sensed CSV with LAI data.
    apsim_db_fpath : str
        Filepath to the APSIM SQLite database.
    total_yield_csv_fpath : str
        Filepath to the CSV containing total yield and conversion factor.
    html_fpath : str, optional
        Filepath to save the interactive HTML report, by default None.
    png_fpath : str, optional
        Filepath to save the PNG image report, by default None.
    """

    logger.info("Loading APSIM filtered data from CSV.")
    apsim_filtered = pd.read_csv(apsim_filtered_fpath)
    if not len(apsim_filtered):
        logger.error(f'Simulation matches in {apsim_filtered_fpath} contains no valid matches.')
    
    logger.info("Loading Sentinel-2 RS CSV.")
    rs_df = pd.read_csv(rs_lai_csv_fpath, index_col='Date', parse_dates=['Date'], dayfirst=True)
    
    logger.info("Loading report data from database.")
    report_data = load_simulation_data(apsim_db_fpath)

    logger.info("Grouping report data by SimulationID.")
    grouped_data = report_data.groupby('SimulationID')

    ###########
    logger.info(f"Reading total yield/conversion factor from {total_yield_csv_fpath}")

    df = pd.read_csv(total_yield_csv_fpath)
    if 'total_yield_kg' not in df.columns:
        raise KeyError("CSV file must contain a 'total_yield_kg' column.")

    total_yield_kg = df['total_yield_kg'].iloc[0]
    total_yield_metric_tons = total_yield_kg / 1000
    mean_yield_kg_ha = df['mean_yield_kg_ha'].round().iloc[0]
    logger.info(f"Total yield: {total_yield_kg} kg")
    
    ###########

    # Style dictionary based on StepFilteredOut
    style_dict = {pd.NA: {'line_color': 'DodgerBlue', 'line_dash': 'solid', 'line_width': 1, 'opacity': 1, 'zorder':0, 'show_group': True},
                  5: {'line_color': 'Navy', 'line_dash': 'solid', 'line_width': 0.5, 'opacity': 0.5, 'zorder':-1, 'show_group': True},
                  4: {'line_color': 'Navy', 'line_dash': 'solid', 'line_width': 0.5, 'opacity': 0.5, 'zorder':-2, 'show_group': True},
                  3: {'line_color': 'Navy', 'line_dash': 'solid', 'line_width': 0.5, 'opacity': 0.5, 'zorder':-3, 'show_group': True},
                  2: {'line_color': 'Navy', 'line_dash': 'solid', 'line_width': 0.5, 'opacity': 0.5, 'zorder':-4, 'show_group': True}}

    # Create figure with two subplots: one for LAI and one for Yield
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing = 0.1,
                        subplot_titles=("APSIM and RS LAI", "APSIM Yield"))

    ###################################
    logger.info("Plotting individual SimulationID series.")
    for _, row in apsim_filtered.iterrows():
        sim_id = int(row['SimulationID'])
        step_filter = row['StepFilteredOut']
        sim_data = grouped_data.get_group(sim_id)
        
        # Define legend group by StepFilteredOut
        legend_group = str(int(step_filter)) if not pd.isna(step_filter) else "N/A"
 
        # Add LAI line
        style = style_dict.get(step_filter, style_dict[pd.NA])
        fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data['Wheat.Leaf.LAI'], mode='lines', name=f'LAI/Yield (Filtered on Step: {legend_group})', opacity=style['opacity'], zorder=style['zorder'],
                                 legendgroup=legend_group, showlegend=style['show_group'],
                                 line=dict(color='DarkGreen', dash=style['line_dash'], width=style['line_width'])),
                      row=1, col=1)
        style['show_group'] = False  # Set this to False so only the first group is shown

        # Add Yield line
        fig.add_trace(go.Scatter(x=sim_data.index, y=sim_data['Yield'], mode='lines', name=f'Yield (Filtered on Step: {legend_group})', opacity=style['opacity'], zorder=style['zorder'],
                                 legendgroup=legend_group, showlegend=False,
                                 line=dict(color=style['line_color'], dash=style['line_dash'], width=style['line_width'])),
                      row=2, col=1)

    ###################################
    logger.info("Plotting RS data.")
    fig.add_trace(go.Scatter(x=rs_df.index, y=rs_df['LAI Mean'], mode='lines', name='RS Mean LAI', 
                             line=dict(color='black', width=3)), row=1, col=1)

    ###################################
    logger.info("Calculating and plotting mean series for simulations not filtered out.")
    good_sim_ids = apsim_filtered[apsim_filtered['StepFilteredOut'].isna()]['SimulationID']
    mean_data = report_data[report_data['SimulationID'].isin(good_sim_ids)].groupby('Date').mean()

    fig.add_trace(go.Scatter(x=mean_data.index, y=mean_data['Wheat.Leaf.LAI'], mode='lines', name='Mean LAI',
                             line=dict(color='chartreuse', width=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=mean_data.index, y=mean_data['Yield'], mode='lines', name='Mean Yield',
                             line=dict(color='deepskyblue', width=4)), row=2, col=1)

    ###################################
    # Set hovermode to compare across all traces (this enables syncing between subplots)

    # Add y-labels for both subplots
    fig.update_yaxes(title_text="LAI", row=1, col=1)
    fig.update_yaxes(title_text="Yield Rate (kg/ha)", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Create metadata for the title
    start_date = mean_data.index.min().strftime('%Y-%m-%d')
    end_date = mean_data.index.max().strftime('%Y-%m-%d')
    n_simulations = len(good_sim_ids)
    
    title_text = (f"<b>Sim/Real (APSIM/S2-LAI) Matching</b><br>"
                  f"Input CSV: <i>{apsim_filtered_fpath}</i><br>"
                  f"Input DB: <i>{apsim_db_fpath}</i><br>"
                  f"Number of simulation traces in mean data: {n_simulations}<br>"
                  f"Date range: {start_date} to {end_date}<br>"
                  f"Mean yield rate: {mean_yield_kg_ha:0.0f} kg/ha<br>"
                  f"Production: <b>{total_yield_metric_tons:0.0f} metric tons</b>")

    fig.update_layout(title=dict(text=title_text, font=dict(size=10)),
                      margin={'t': 275})  # Adjust the top margin to avoid overlap
                      #hovermode='x unified')

    # Save outputs
    if html_fpath:
        logger.info(f"Saving HTML report to {html_fpath}")
        fig.write_html(html_fpath)

    if png_fpath:
        logger.info(f"Saving PNG report to {png_fpath}")
        fig.update_layout(width=1000, height=1200)  # Adjust the top margin to avoid overlap
        fig.write_image(png_fpath)


@click.command()
@click.option('--apsim_filtered_fpath', required=True, type=click.Path(exists=True), help='Filepath to the filtered CSV.')
@click.option('--rs_lai_csv_fpath', required=True, type=click.Path(exists=True), help='Path to remotely sensed LAI CSV file')
@click.option('--apsim_db_fpath', required=True, type=click.Path(exists=True), help='Filepath to the APSIM SQLite database.')
@click.option('--total_yield_csv_fpath', required=True, type=click.Path(exists=True), help='Filepath to CSV with the conversion factor and total yield.')
@click.option('--html_fpath', type=click.Path(), help='Optional filepath to save the HTML report.')
@click.option('--png_fpath', type=click.Path(), help='Optional filepath to save the PNG report.')
@click.option('--verbose', is_flag=True, help='Enable verbose logging.')
def cli(apsim_filtered_fpath, rs_lai_csv_fpath, apsim_db_fpath, total_yield_csv_fpath, html_fpath, png_fpath, verbose):
    """
    CLI wrapper for generating the APSIM report from a CSV and SQLite database.

    This function calls the core `generate_report` function and provides
    command-line arguments for setting the input paths and optional outputs.
    """
    if verbose:
        logger.setLevel('INFO')

    generate_report(apsim_filtered_fpath, rs_lai_csv_fpath, apsim_db_fpath, total_yield_csv_fpath, html_fpath, png_fpath)


if __name__ == "__main__":
    cli()