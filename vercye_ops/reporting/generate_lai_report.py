from collections import defaultdict
import os
import glob
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns
import click

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_date(date_str):
    """Parse date string in DD/MM/YYYY format to datetime"""
    try:
        return pd.to_datetime(date_str, format='%d/%m/%Y')
    except:
        return pd.NaT

def load_and_process_lai_data(file_path):
    """Load LAI data and process dates"""
    df = pd.read_csv(file_path)
    df['Date_parsed'] = df['Date'].apply(parse_date)
    df = df.dropna(subset=['Date_parsed'])
    df = df.sort_values('Date_parsed')
    return df

def calculate_mean_curve_for_agglvl(file_paths, lai_column):
    """Calculate mean LAI curve across multiple regions within an aggregation lvl"""
    common_dates = None
    
    # Load all files and find common dates
    dfs = []
    for file_path in file_paths:
        try:
            df = load_and_process_lai_data(file_path)
            if len(df) > 0:
                dfs.append(df)
                if common_dates is None:
                    common_dates = set(df['Date_parsed'])
                else:
                    common_dates = common_dates.intersection(set(df['Date_parsed']))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not dfs or not common_dates:
        return None, None
    
    # Convert to sorted list
    common_dates = sorted(list(common_dates))
    
    # Extract LAI values for common dates
    lai_curves = []
    for df in dfs:
        df_subset = df[df['Date_parsed'].isin(common_dates)].copy()
        df_subset = df_subset.sort_values('Date_parsed')
        
        # Use LAI Median Adjusted, fall back to LAI Median if not available
        lai_values = df_subset[lai_column].values

            
        # Only include if we have valid values
        if len(lai_values) == len(common_dates):
            lai_curves.append(lai_values)
    
    if not lai_curves:
        return None, None
    
    # Calculate mean curve
    lai_matrix = np.array(lai_curves)
    mean_curve = np.nanmean(lai_matrix, axis=0)
    
    return common_dates, mean_curve


def get_axis(axes, row, col, total_rows, n_cols):
    """Helper function to safely get the correct axis from the axes array"""
    if total_rows == 1 and n_cols == 1:
        return axes
    elif total_rows == 1:
        return axes[col]
    elif n_cols == 1:
        return axes[row]
    else:
        return axes[row, col]


def create_agg_plots(basedir, out_path, admin_agg_column, lai_column):
    """Create plots for each timepoint, with subplots for each admin unit showing mean curves across years"""
    
    # Collect all data organized by timepoint
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print("Scanning directories...")
    for year in os.listdir(basedir):
        year_path = os.path.join(basedir, year)
        if not os.path.isdir(year_path):
            continue
        
        for timepoint in os.listdir(year_path):
            timepoint_path = os.path.join(year_path, timepoint)
            if not os.path.isdir(timepoint_path):
                continue

            for region in os.listdir(timepoint_path):
                region_path = os.path.join(timepoint_path, region)
                if not os.path.isdir(region_path):
                    continue
                    
                lai_stats_file = os.path.join(region_path, f'{region}_LAI_STATS.csv')
                geojson_file = os.path.join(region_path, f'{region}.geojson')
                
                if os.path.exists(lai_stats_file) and os.path.exists(geojson_file):
                    try:
                        gdf = gpd.read_file(geojson_file)
                        if admin_agg_column in gdf.columns and len(gdf) > 0:
                            admin_name = gdf[admin_agg_column].values[0]
                            results[timepoint][year][admin_name].append(lai_stats_file)
                        else:
                            raise KeyError(f'{admin_agg_column} column not found or no data in shapefile.')
                    except Exception as e:
                        print(f"Error reading {geojson_file}: {e}")
    
    # Get all unique timepoints and admin units
    all_timepoints = sorted(results.keys())
    all_admin_names = set()
    for timepoint_data in results.values():
        for year_data in timepoint_data.values():
            all_admin_names.update(year_data.keys())
    all_admin_names = sorted(all_admin_names)
    
    print(f"Found {len(all_timepoints)} timepoints: {all_timepoints}")
    print(f"Found {len(all_admin_names)} admin units: {all_admin_names}")
    
    if len(all_timepoints) == 0 or len(all_admin_names) == 0:
        print("No data found!")
        return
    
    # Calculate grid dimensions for admin units within each timepoint section
    n_admin = len(all_admin_names)
    n_cols = min(3, n_admin)
    n_rows_per_timepoint = (n_admin + n_cols - 1) // n_cols
    
    # Create figure with sections for each timepoint
    total_rows = len(all_timepoints) * n_rows_per_timepoint
    fig, axes = plt.subplots(total_rows, n_cols, figsize=(6*n_cols, 4*total_rows))
    
    # Color mapping for years
    all_years = set()
    for timepoint_data in results.values():
        all_years.update(timepoint_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_years)))
    year_color_map = dict(zip(sorted(all_years), colors))
    
    # Process each timepoint
    for timepoint_idx, timepoint in enumerate(all_timepoints):
        print(f"\nProcessing timepoint: {timepoint}")
        
        # Add section title
        section_start_row = timepoint_idx * n_rows_per_timepoint
        if n_cols > 1:
            fig.text(0.5, 1 - (section_start_row + 0.5) / total_rows, 
                    f'Timepoint: {timepoint}', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    transform=fig.transFigure)
        
        # Plot each admin unit for this timepoint
        for admin_idx, admin_unit in enumerate(all_admin_names):
            row_in_section = admin_idx // n_cols
            col = admin_idx % n_cols
            global_row = section_start_row + row_in_section
            
            # Get the appropriate axis using the helper function
            ax = get_axis(axes, global_row, col, total_rows, n_cols)
            
            print(f"  Processing admin unit: {admin_unit}")
            
            # Plot curves for each year in this timepoint
            plotted_any = False
            for year in sorted(all_years):
                if timepoint in results and year in results[timepoint] and admin_unit in results[timepoint][year]:
                    file_paths = results[timepoint][year][admin_unit]
                    dates, mean_curve = calculate_mean_curve_for_agglvl(file_paths, lai_column)
                    
                    if dates is not None and mean_curve is not None:
                        # Convert dates to day of year for consistent x-axis
                        day_of_year = [d.timetuple().tm_yday for d in dates]
                        
                        ax.plot(day_of_year, mean_curve, 
                               label=f'{year} (n={len(file_paths)})',
                               color=year_color_map[year],
                               linewidth=2,
                               alpha=0.8)
                        plotted_any = True
                        print(f"    Plotted {year}: {len(file_paths)} regions, {len(dates)} dates")
            
            # Configure the subplot
            ax.set_title(f'{admin_unit}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel(f'Mean {lai_column}')
            ax.grid(True, alpha=0.3)
            
            if plotted_any:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
            
            ax.set_ylim(0, None)
        
        # Hide empty subplots in this timepoint section
        used_plots = len(all_admin_names)
        for i in range(used_plots, n_rows_per_timepoint * n_cols):
            row_in_section = i // n_cols
            col = i % n_cols
            global_row = section_start_row + row_in_section
            
            if global_row < total_rows:
                ax = get_axis(axes, global_row, col, total_rows, n_cols)
                ax.set_visible(False)
    
    plt.tight_layout()
    
    plt.subplots_adjust(top=0.95, hspace=0.4)
    
    # Save the plot
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {out_path}")
    print(f"Created {len(all_timepoints)} sections with {len(all_admin_names)} admin units each")


@click.command()
@click.option('--base-dir', type=click.Path(exists=True), required=True, help='Yield Study base directory. Must be following the vercye output structure as produced by the snakemake pipeline.')
@click.option('--out-path', type=click.Path(), required=True, help='Path to the output file. Must be .pdf')
@click.option('--admin-agg-column', type=str, required=True, help='Shapefile column name that contains the admin names to aggregate by.')
@click.option('--lai-agg-type', type=click.Choice(['Mean', 'Median']),  help="Column of the LAI traces to use - either Mean or Median.")
@click.option('--adjusted', is_flag=True, default=False, help="Use the adjusted column in the LAI data.")
def main(base_dir, out_path, admin_agg_column, lai_agg_type, adjusted):
    lai_column = 'LAI Mean' if lai_agg_type == 'Mean' else 'LAI Median'
    lai_column = lai_column + ' Adjusted' if adjusted else lai_column
    create_agg_plots(base_dir, out_path, admin_agg_column, lai_column)


if __name__ == "__main__":
    main()