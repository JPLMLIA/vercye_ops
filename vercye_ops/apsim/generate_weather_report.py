"""CLI to generate a weather report from CSV of weather"""

import click
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

MET_FILE_DTYPES = {
    'year': int,
    'day': int,
    'radn': float,
    'maxt': float,
    'mint': float,
    'meant': float,
    'rain': float,
    'wind': float,
    'data_type': str
}

@click.command()
@click.option('--input_fpath', required=True, type=click.Path(exists=True), help='Path to the input .met file.')
@click.option('--output_fpath', required=True, type=click.Path(writable=True), help='Path to save the output HTML plot.')
@click.option('--precipitation_source', required=True, type=click.Choice(['NASA_POWER', 'CHIRPS']), help='Source of precipitation data. "NASA_Power" or "CHIRPS".')
@click.option('--precipitation_agg', required=True, type=click.Choice(['centroid', 'mean']), help='Aggregation method for precipitation data. "centroid" or "mean".')
@click.option('--nasapower_fallback', default=False, help='Boolean specifying wether to fallback on nasa power data if chirps data is not available.')
@click.option('--nasapower_fpath', required=False, type=click.Path(exists=True), default=None, help='Path to the NASA Power CSV file.')
@click.option('--header_lines', default=8, show_default=True, help='Number of lines in the header.')
@click.option('--column_line', default=6, show_default=True, help='Line number of the column headers (1-based index).')
def cli(input_fpath, output_fpath, precipitation_source, precipitation_agg, nasapower_fallback, nasapower_fpath, header_lines, column_line):
    """
    CLI wrapper to plot weather data from a .met file and save as an interactive HTML plot.
    """
    data, metadata = plot_weather_data(input_fpath, precipitation_source, precipitation_agg, nasapower_fallback, nasapower_fpath, header_lines, column_line)
    
    metadata.update({'input_fpath': input_fpath,
                     'output_fpath': output_fpath})
    fig = create_plots(data, metadata)
    fig.write_html(output_fpath)


def plot_weather_data(file_path, precipitation_source, precipitation_agg, nasapower_fallback, nasapower_fpath, header_lines=8, column_line=6):
    """
    Plot weather data from a .met file and save as an interactive HTML plot.

    Parameters
    ----------
    file_path : str
        Path to the input .met file.
    precipitation_source : str
        Source of precipitation data. "NASA_POWER" or "CHIRPS".
    precipitation_agg : str
        Aggregation method for precipitation data. "centroid" or "mean".
    nasapower_fpath : str
        Path to the NASA Power CSV file.
    header_lines : int
        Number of lines in the header.
    column_line : int
        Line number of the column headers (1-based index).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed weather data.
    dict
        Dictionary containing metadata (latitude, longitude, tav, amp).
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata
    metadata = {
        'latitude': float(lines[1].split('=')[1].strip().split()[0]),
        'longitude': float(lines[2].split('=')[1].strip().split()[0]),
        'tav': float(lines[3].split('=')[1].strip().split()[0]),
        'amp': float(lines[4].split('=')[1].strip().split()[0])}
    
    # Extract column names
    columns = lines[column_line - 1].strip().split()
    
    # Extract units
    units_line = lines[column_line].strip().split()
    units = dict(zip(columns, units_line))
    metadata['units'] = units
    
    # Read the data
    data = [line.strip().split() for line in lines[header_lines:]]
    
    # Create DataFrame, convert types
    df = pd.DataFrame(data, columns=columns)
    df = df.astype(MET_FILE_DTYPES)
    df['date'] = pd.to_datetime(df['year'].astype(str) + df['day'].astype(str), format='%Y%j')

    # Add unused NASA Power precipitation data for comparison if available
    metadata['precipitation_source'] = precipitation_source
    metadata['precipitation_agg'] = precipitation_agg

    if precipitation_source.lower() == 'chirps':
        df_nasapower = pd.read_csv(nasapower_fpath)
        if not 'NASA_POWER_PRECTOTCORR_UNUSED' in df_nasapower.columns:
            if not nasapower_fallback:
                raise ValueError('NASA Power CSV file does not contain the required column "NASA_POWER_PRECTOTCORR_UNUSED".')
            metadata['precipitation_source'] = 'NASA Power fallback.'
            metadata['precipitation_agg'] = 'centroid'
        else:
            df['rain_nasapower_unused'] = df_nasapower['NASA_POWER_PRECTOTCORR_UNUSED']
            metadata['precipitation_source'] = 'CHIRPS'
    
    # Extract date information
    metadata['met_start_date'] = df.iloc[0]['date']
    metadata['met_end_date'] = df.iloc[-1]['date']
    metadata['met_meas_end_date'] = None

    measured_rows = df[df['data_type'] == 'measured']
    if not measured_rows.empty:
        metadata['met_meas_end_date'] = measured_rows.iloc[-1]['date']
    
    return df, metadata

def create_plots(df, metadata):
    """
    Create interactive subplots for the weather data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the weather data.
    metadata : dict
        Dictionary containing metadata (latitude, longitude, tav, amp).

    Returns
    -------
    go.Figure
        Plotly figure with interactive subplots.
    """
    # Create subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Temperature (Max, Mean, Min)', 'Radiance', 'Rain', 'Surface Wind'))

    # Convenience function for plotting
    def add_traces(entire_df, data_type, fig):
        if data_type == 'measured':
            line_dash = 'solid'
        elif data_type == 'projected':
            line_dash = 'dot'
        else:
            raise ValueError('Data type not understood. Expected `measured` or `projected`.')
        
        df = entire_df[entire_df['data_type'] == data_type].copy()

        # Temperature
        fig.add_trace(go.Scatter(x=df['date'], y=df['maxt'], mode='lines', name=f'Max Temp ({data_type})',
                                 line=dict(color='red', dash=line_dash, width=0.75)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['mint'], mode='lines', name=f'Min Temp ({data_type})',
                                 line=dict(color='blue', dash=line_dash, width=0.75)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['meant'], mode='lines', name=f'Mean Temp ({data_type})',
                                 line=dict(color='black', dash=line_dash, width=0.75)), row=1, col=1)

        # Radiance
        fig.add_trace(go.Scatter(x=df['date'], y=df['radn'], mode='lines', name=f'Radiance ({data_type})',
                                 line=dict(color='goldenrod', dash=line_dash, width=1.5)), row=2, col=1)

        # Precipitation
        if 'rain_nasapower_unused' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['rain_nasapower_unused'], mode='lines', name=f'Unused NASA_Power Precipitation ({data_type})',
                                     line=dict(color='darkgreen', dash=line_dash, width=1.5)), row=3, col=1)

            fig.add_trace(go.Scatter(x=df['date'], y=df['rain'], mode='lines', name=f'Chirps Precipitation (Corrected) ({data_type})',
                                    line=dict(color='dodgerblue', dash=line_dash, width=1.5)), row=3, col=1)
        else:
            fig.add_trace(go.Scatter(x=df['date'], y=df['rain'], mode='lines', name=f'Precipitation (Corrected) ({data_type})',
                                    line=dict(color='navy', dash=line_dash, width=1.5)), row=3, col=1)

        # Wind
        fig.add_trace(go.Scatter(x=df['date'], y=df['wind'], mode='lines', name=f'Surface Wind ({data_type})',
                                 line=dict(color='purple', dash=line_dash, width=1.5)), row=4, col=1)

    # Add traces for measured data
    add_traces(df, 'measured', fig)
    
    # Add traces for projected data
    add_traces(df, 'projected', fig)
    
    # Add vertical line for the transition date
    fig.add_vline(x=metadata['met_meas_end_date'].timestamp() * 1000,  # This is a bug workaround: https://github.com/plotly/plotly.py/issues/3065
                  row='all',
                  line=dict(color='gray', dash='dash'),
                  annotation_text='End Met. Data',
                  annotation_position='top left')

    # Update text, layout, and hover details
    sim_date_information = f"Met File Start Date: {metadata['met_start_date'].date()}<br>Met File Last Measurement Date: {metadata['met_meas_end_date'].date()}<br>Met File End Date: {metadata['met_end_date'].date()}<br>Precipitation Config: {metadata['precipitation_source']}, {metadata['precipitation_agg']}"
    met_header_information = f"Lat {metadata['latitude']}, Lon {metadata['longitude']}, Tav {metadata['tav']}°C, Amp {metadata['amp']}°C"

    fig.update_layout(
        hovermode='x unified',
        title_text=f"<b>Weather Data</b><br>Input file: <i>{metadata['input_fpath']}</i><br>{met_header_information}<br>{sim_date_information}<br>",
        title={'y':0.95},  # Adjust the title position to avoid overlap
        margin=dict(t=200),  # Adjust the top margin to avoid overlap
        yaxis={'title':f'Temperature {metadata["units"]["meant"]}'},
        yaxis2={'title':f'Radiance {metadata["units"]["radn"]}'},
        yaxis3={'title':f'Rain {metadata["units"]["rain"]}'},
        yaxis4={'title':f'Surface Wind {metadata["units"]["wind"]}'})
    
    return fig


if __name__ == '__main__':
    cli()