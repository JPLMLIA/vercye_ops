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
    all_curves = []
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



def create_agg_plots(basedir, out_path, admin_agg_column, lai_column):
    """Create plots for each admin unit showing mean curves across years"""
    
    # Collect all data
    results = defaultdict(lambda: defaultdict(list))
    
    print("Scanning directories...")
    for year in os.listdir(basedir):
        year_path = os.path.join(basedir, year)
        if not os.path.isdir(year_path):
            continue
            
        tp_path = os.path.join(year_path, 'T-0')
        if not os.path.exists(tp_path):
            continue
        for region in os.listdir(tp_path):
            region_path = os.path.join(tp_path, region)
            if not os.path.isdir(region_path):
                continue
                
            lai_stats_file = os.path.join(region_path, f'{region}_LAI_STATS.csv')
            geojson_file = os.path.join(region_path, f'{region}.geojson')
            
            if os.path.exists(lai_stats_file) and os.path.exists(geojson_file):
                try:
                    gdf = gpd.read_file(geojson_file)
                    if admin_agg_column in gdf.columns and len(gdf) > 0:
                        admin_name = gdf[admin_agg_column].values[0]
                        results[year][admin_name].append(lai_stats_file)
                    else:
                        raise KeyError(f'{admin_agg_column} column not found or no data in shapefile.')
                except Exception as e:
                    print(f"Error reading {geojson_file}: {e}")
    
    # Get all unique admin units
    all_admin_names = set()
    for year_data in results.values():
        all_admin_names.update(year_data.keys())
    
    print(f"Found {len(all_admin_names)} admin units: {sorted(all_admin_names)}")
    
    # Create plots for each admin unit
    n_admin = len(all_admin_names)
    if n_admin == 0:
        print("No data found!")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_admin)
    n_rows = (n_admin + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_admin == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results.keys())))
    year_color_map = dict(zip(sorted(results.keys()), colors))
    
    plot_idx = 0
    for admin_unit in sorted(all_admin_names):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        print(f"Processing admin unit: {admin_unit}")
        
        # Plot curves for each year
        for year in sorted(results.keys()):
            if admin_unit in results[year]:
                file_paths = results[year][admin_unit]
                dates, mean_curve = calculate_mean_curve_for_agglvl(file_paths, lai_column)
                
                if dates is not None and mean_curve is not None:
                    # Convert dates to day of year for consistent x-axis
                    day_of_year = [d.timetuple().tm_yday for d in dates]
                    
                    ax.plot(day_of_year, mean_curve, 
                           label=f'{year} (n={len(file_paths)})',
                           color=year_color_map[year],
                           linewidth=2,
                           alpha=0.8)
                    print(f"  Plotted {year}: {len(file_paths)} regions, {len(dates)} dates")
        
        ax.set_title(f'{admin_unit}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('LAI Median')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set reasonable y-axis limits
        ax.set_ylim(0, None)
        
        plot_idx += 1
    
    # Hide empty subplots
    for i in range(plot_idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row][col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")


@click.command()
@click.option('--base-dir', type=click.Path(exists=True), required=True, help='Yield Study base directory. Must be following the vercye output structure as produced by the snakemake pipeline.')
@click.option('--out-path', type=click.Path(), require=True, help='Path to the output file. Must be .pdf')
@click.option('--admin-agg-column', type=str, required=True, help='Shapefile column name that contains the admin names to aggregate by.')
@click.option('--lai-agg-type', type=click.Choice(['Mean', 'Median']),  help="Column of the LAI traces to use - either Mean or Median.")
@click.option('--adjusted', is_flag=True, default=False, help="Use the adjusted column in the LAI data.")
def main(base_dir, out_path, admin_agg_column, lai_agg_type, adjusted):
    lai_column = 'Mean LAI' if lai_agg_type =='Mean' else 'Median'
    lai_column = lai_column + ' Adjusted' if adjusted else lai_column
    create_agg_plots(base_dir, out_path, admin_agg_column, lai_column)


if __name__ == "__main__":
    main()
    basedir = '/gpfs/data1/cmongp2/wronk/Data/vercye_ops/yieldstudy_20241108'
    outdir = '/gpfs/data1/cmongp2/sawahnr/nasa-harvest/vercye/experiments/validate_LAI_JRC/outs'
    
    