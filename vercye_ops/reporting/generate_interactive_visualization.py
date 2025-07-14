from collections import defaultdict
import os
import shutil
import click
import geopandas as gpd
import pandas as pd
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

DEFAULT_LAI_COLUMNS = [
    "LAI Median:Median LAI",
    "LAI Mean:Mean LAI"
]

class InteractiveMapGenerator:
    def __init__(self, shapefile_path: str, basedir_path: str, output_dir: str, lai_columns: List[str], lai_column_names: List[str], agg_levels: Dict[str, Tuple[str, str]], simplify_tolerance: float = 0.01, zip: bool = False):
        """Initialize the map generator."""
        self.shapefile_path = shapefile_path
        self.basedir_path = basedir_path
        self.output_dir = output_dir
        self.gdf = None
        self.lai_data = None
        self.sim_data = None
        self.aggregation_levels = []
        self.aggregated_estimates = {}
        self.lai_columns = lai_columns
        self.lai_column_names = lai_column_names
        self.simplify_tolerance = simplify_tolerance
        self.zip = zip

        os.makedirs(self.output_dir, exist_ok=True)

        for agg_name, (agg_column, agg_estimates_fpath) in agg_levels.items():
            self.add_aggregation_level(agg_name, agg_column, agg_estimates_fpath)

        self.load_data()
        
    def load_data(self):
        """Load shapefile and LAI data."""
        print("Loading shapefile...")
        self.gdf = gpd.read_file(self.shapefile_path)

        # Select columns to keep & append aggregation columns if they exist
        keep_cols = ["cleaned_region_name_vercye", "estimated_mean_yield_kg_ha", "total_area_ha",
                     "estimated_median_yield_kg_ha", "geometry", "reported_mean_yield_kg_ha"]
        keep_cols += [self.aggregation_levels[i]['column'] for i in range(len(self.aggregation_levels)) if self.aggregation_levels[i]['column'] not in keep_cols]
        print(keep_cols)

        # Filter columns to reduce file size
        self.gdf = self.gdf[keep_cols]
        
        # Ensure CRS is WGS84 for web mapping
        if self.gdf.crs != 'EPSG:4326':
            print(f"Reprojecting from {self.gdf.crs} to EPSG:4326...")
            self.gdf = self.gdf.to_crs('EPSG:4326')
        print(f"Loaded {len(self.gdf)} features from shapefile")

        # Load aggregated estimates
        print("Loading aggregated estimates...")
        self.aggregated_estimates = self.load_aggregated_estimates()
        print('Loaded aggregated estimates for all levels')
        
        print("Loading LAI data...")
        self.lai_data = self.load_lai_data()
        print(f"Loaded LAI data")

        print("Loading simulations data")
        self.sim_data = self.load_sim_data()
        
    
    def add_aggregation_level(self, level_name: str, column_name: str, agg_estimate_fpath: str):
        """
        Add an aggregation level.
        
        Args:
            level_name: Display name for this level (e.g., "States", "Counties")
            column_name: Column name to aggregate by
        """

        # Currently hardcoded - TODO make dynamic aswell
        errors_fpath = str(Path(agg_estimate_fpath).parent / f'errors_{Path(agg_estimate_fpath).name.split("_")[3]}.csv')
        
        self.aggregation_levels.append({
            'name': level_name,
            'column': column_name,
            'agg_estimates_file': agg_estimate_fpath,
            'errors_file': errors_fpath
        })

        print(f"Added aggregation level: {level_name} (by {column_name})")
    
    
    def aggregate_lai_data(self, regions: List[str]) -> Tuple[List[float], List[str], List[bool]]:
        """
        Aggregate LAI data for a list of regions by taking mean values per timepoint.
        
        Args:
            regions: List of region names to aggregate
            
        Returns:
            Tuple of (LAI values, date labels, interpolation flags)
        """

        # Filter LAI data for the specified regions
        region_data = self.lai_data[self.lai_data['cleaned_region_name_vercye'].isin(regions)]

        if region_data.empty:
            return [], [], []
        
        # Sort by date to ensure chronological order
        region_data = region_data.sort_values('Date')
        
        # Calculate mean LAI Median and interpolation status for each date
        aggregated_values = defaultdict(list)
        date_labels = []
        interpolation_flags = []
        
        date_groups = region_data.groupby('Date')
        
        for date, group in date_groups:
            # Collect LAI values

            for display_name, column_name in zip(self.lai_column_names, self.lai_columns):
                if column_name in group.columns:
                    mean_lai_ = float(np.nanmean(group[column_name].values))
                    aggregated_values[display_name].append(mean_lai_)
            
            # Format date for display
            date_labels.append(date.strftime('%m/%d'))
            
            if group['interpolated'].isin([1]).any():
                # If any value is interpolated, mark as interpolated
                interpolation_flags.append(1)
            else:
                interpolation_flags.append(0)

        return aggregated_values, date_labels, interpolation_flags
    
    def load_agg_geometries(self, agg_col):
        # Group polygon by aggregation column and union into single geom
        grouped = self.gdf.groupby(agg_col).agg({
            'cleaned_region_name_vercye': lambda x: list(x),
            'geometry': lambda x: x.unary_union,
        }).reset_index()

        grouped['subregions'] = grouped['cleaned_region_name_vercye']

        return grouped
    
    def create_geojson_level(self, level_idx: int) -> Dict[str, Any]:
        """
        Create GeoJSON data for a specific aggregation level.
        
        Args:
            level_idx: Index of the aggregation level
            
        Returns:
            GeoJSON FeatureCollection
        """

        # Load geometries
        is_base_level = level_idx == len(self.aggregation_levels) - 1
        agg_col = self.aggregation_levels[level_idx]['column']

        if is_base_level:
            geometries = self.gdf
            geometries['subregions'] = geometries[agg_col].apply(lambda x: [x])
        else:
            geometries = self.load_agg_geometries(agg_col)

        # Create feature for every entry
        features = []
        for idx, row in geometries.iterrows():
            geometry = row["geometry"]
            geometry = geometry.simplify(
                tolerance=self.simplify_tolerance,
                preserve_topology=True
            )

            # Get LAI data for all regions in this group
            lai_values, date_labels, interpolation_flags = self.aggregate_lai_data(row['subregions'])

            # Extract data to display
            region_name = str(row[agg_col])
            if region_name in self.aggregated_estimates[level_idx]:
                mean_estimated_yield_kg_ha = self.aggregated_estimates[level_idx][region_name]['estimated_mean_yield_kg_ha']
                median_estimated_yield_kg_ha = self.aggregated_estimates[level_idx][region_name]['estimated_median_yield_kg_ha']
                sum_area = self.aggregated_estimates[level_idx][region_name]['total_area_ha']
            else:
                mean_estimated_yield_kg_ha = np.nan
                median_estimated_yield_kg_ha = np.nan
                sum_area = np.nan

            # Create polygon feature vector to display              
            feature = {
                "type": "Feature",
                "properties": {
                    "id": f"agg_{level_idx}_{idx}",
                    "name": region_name,
                    "estimated_mean_yield_kg_ha": float(mean_estimated_yield_kg_ha),
                    "estimated_median_yield_kg_ha": float(median_estimated_yield_kg_ha),
                    "total_area": sum_area,
                    "timeSeries": lai_values,
                    "dateLabels": date_labels,
                    "interpolationFlags": interpolation_flags,
                    "subregions": row['subregions'] if not is_base_level else [],
                    "isAggregated": False if is_base_level else True,
                    "error": np.nan,
                    "relative_error": np.nan
                },
                "geometry": geometry.__geo_interface__
            }
            
            # If base level, link the simulations report
            if is_base_level and region_name in self.sim_data:
                feature["properties"]["simulationsImgPath"] = self.sim_data[region_name]

            # Add reference data if available
            if region_name in self.aggregated_estimates[level_idx] and 'reported_mean_yield_kg_ha' in self.aggregated_estimates[level_idx][region_name]:
                feature['properties']['reported_mean_yield_kg_ha'] = float(self.aggregated_estimates[level_idx][region_name]['reported_mean_yield_kg_ha'])
            
            if region_name in self.aggregated_estimates[level_idx] and 'error' in self.aggregated_estimates[level_idx][region_name]:
                feature['properties']["error"] = float(self.aggregated_estimates[level_idx][region_name]['error'])
                feature['properties']["relative_error"] =  float(self.aggregated_estimates[level_idx][region_name]['relative_error'])

            features.append(feature)
   
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def create_hierarchical_data(self) -> Dict[str, Any]:
        """
        Create hierarchical data structure for all levels.
        
        Returns:
            Dictionary with level data and mappings
        """
        levels = {
            "levels": [],
            "level_data": {},
            "level_mappings": {}
        }
        
        # Add level information
        for i, level in enumerate(self.aggregation_levels):
            levels["levels"].append({
                "index": i,
                "name": level['name'],
                "column": level['column']
            })
            
            # Create GeoJSON for this level
            level_geojson = self.create_geojson_level(i)
            levels["level_data"][f"level_{i}"] = level_geojson

        # Create mappings between levels
        if len(self.aggregation_levels) > 1:
            for i in range(len(self.aggregation_levels) - 1):
                print(f"level_{i}")
                print(self.aggregation_levels[i])
                parent_level = levels["level_data"][f"level_{i}"]
                next_level_data = levels["level_data"][f"level_{i+1}"]
                
                # Create mapping from parent features to child regions
                for feature in parent_level["features"]:
                    parent_id = feature["properties"]["id"]
                    parent_regions = feature["properties"].get("subregions", [])

                    child_features = [f for f in next_level_data["features"] if f["properties"]['name'] in parent_regions]
                    
                    if child_features:
                        levels["level_mappings"][parent_id] = {
                            "type": "FeatureCollection",
                            "features": child_features
                        }
        
        return levels
    
    def compute_value_ranges(self, levels):
        value_ranges = {}

        # Calculate value range for color scaling
        for valueType in ['estimated_mean_yield_kg_ha', 'estimated_median_yield_kg_ha', 'error', 'relative_error']:
            all_values = []
            for level_key, level_data in levels["level_data"].items():
                for feature in level_data["features"]:
                    all_values.append(feature["properties"][valueType])

            for mapping_data in levels["level_mappings"].values():
                for feature in mapping_data["features"]:
                    all_values.append(feature["properties"][valueType])
            
            all_values = [v for v in all_values if not np.isnan(v)]

            if len(all_values) == 0:
                min_val = 0
                max_val = 0
                p95_val = 0
            else:
                # Compute min, max, and the 95th percentile
                min_val = float(np.min(all_values))
                max_val = float(np.max(all_values))
                p95_val = float(np.percentile(all_values, 95))

            value_ranges[valueType] = {
                'min_val': min_val,
                'max_val': max_val,
                'p95_val': p95_val
            }

        return value_ranges
    
    def generate_html_template(self, levels: Dict[str, Any], title: str) -> str:
        """
        Generate HTML template with embedded data.
        
        Args:
            levels: Hierarchical data structure with polygons and data per level
            
        Returns:
            Complete HTML string
        """

        value_ranges = self.compute_value_ranges(levels)


        title = title + ' Analysis'

        lai_column_options = ''.join(
            f'<option value="{col}">{col}</option>'
            for col in self.lai_column_names
        )

        dark_primary = '#324e47'
        dark_white = '#F5F8F5'
        dark_gray = ''
        light_grey = '#4a5568'
        red = '#f56565'
        red_dark ='#e52b50'
        blue = '#4974a5'
        blue_light = ''
        white = '#FFF'

        
        template = f'''<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" />
                <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.1.0/chroma.min.js"></script>
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Lexend+Exa:wght@100..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">
                <link href="https://fonts.googleapis.com/css2?family=Lexend+Exa:wght@100..900&family=Merriweather:ital,opsz,wght@0,18..144,300..900;1,18..144,300..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&display=swap" rel="stylesheet">
                <link href="https://fonts.googleapis.com/css2?family=Lexend+Exa:wght@100..900&family=Merriweather:ital,opsz,wght@0,18..144,300..900;1,18..144,300..900&family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Roboto:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet">
                
                <style>
                    body {{
                        margin: 0;
                        font-family: "Merriweather", serif;
                        background: #f9f9f9;
                        color: #2d3748;
                    }}

                    p {{
                        font-size: 14px;
                    }}
                    
                    .container {{
                        display: flex;
                        height: 100vh;
                    }}

                    .title {{
                        color: {dark_primary};
                        font-family: "Roboto", sans-serif;
                        weight: 400;

                    }}
                    
                    .map-container {{
                        flex: 1;
                        position: relative;
                    }}
                    
                    #map {{
                        height: 100%;
                        width: 100%;
                    }}
                    
                    .sidebar {{
                        width: 500px;
                        background: #ffffff;
                        border-left: 2px solid #e2e8f0;
                        display: flex;
                        flex-direction: column;
                    }}

                    .sidebar-header {{
                        padding: 20px;
                        border-bottom: 1px solid #e2e8f0;
                        background: {dark_white};
                    }}
                    
                    .sidebar-content {{
                        flex: 1;
                        padding: 20px;
                        overflow-y: auto;
                    }}
                    
                    .breadcrumb {{
                        display: flex;
                        align-items: center;
                        font-size: 14px;
                        color: #718096;
                        flex-wrap: wrap;
                    }}
                    
                    .breadcrumb-item {{
                        cursor: pointer;
                        padding: 4px 8px;
                        border-radius: 4px;
                        transition: all 0.2s;
                        margin: 2px;
                    }}
                    
                    .breadcrumb-item:hover {{
                        background: #edf2f7;
                        color: #2d3748;
                    }}
                    
                    .breadcrumb-separator {{
                        margin: 0 4px;
                        color: #cbd5e0;
                    }}

                    .selected-regions {{
                        margin: 20px 0;
                        border-radius: 12px;
                        padding: 15px;
                        background: {dark_white};
                    }}

                    .selected-regions h3 {{
                        color: #2b6cb0;
                        margin-bottom: 15px;
                        font-size: 18px;
                        font-weight: 600;
                    }}

                    .selected-region-item {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        padding: 8px 12px;
                        margin-bottom: 8px;
                        background: #e6f0fa;
                        border-radius: 8px;
                        transition: all 0.2s ease;
                    }}

                    .selected-region-name {{
                        color: #2d3748;
                        font-size: 14px;
                    }}

                    .remove-region-btn {{
                        background: {red};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-size: 12px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    }}

                    .remove-region-btn:hover {{
                        background: {red_dark};
                    }}

                    .heatmap-selector label {{
                        display: block;
                        color: {light_grey};
                        font-size: 14px;
                        margin-bottom: 8px;
                        margin-top: 10px;
                    }}

                    .heatmap-selector select {{
                        width: 100%;
                        padding: 8px 12px;
                        background: {white};
                        border: 1px solid {dark_primary};
                        border-radius: 8px;
                        color: #2d3748;
                        font-size: 14px;
                    }}

                    .heatmap-selector select:focus-visible {{
                        outline: 0px!important;
                    }}

                    .heatmap-selector select option {{
                        background: {white};
                        color: {dark_gray};
                    }}

                    .lai-selector {{
                        margin-bottom: 15px;
                    }}

                    .lai-selector label {{
                        display: block;
                        color: {light_grey};
                        font-size: 14px;
                        margin-bottom: 8px;
                    }}

                    .lai-selector select {{
                        width: 100%;
                        padding: 8px 12px;
                        background: {white};
                        border: 1px solid #4a5568;
                        border-radius: 8px;
                        color: #2d3748;
                        font-size: 14px;
                    }}

                    .lai-selector select:focus-visible {{
                        outline: 0px!important;
                    }}

                    .lai-selector select option {{
                        background: {white};
                        color: {dark_gray};
                    }}

                    .compare-button {{
                        width: 100%;
                        padding: 12px;
                        background: {blue};
                        color: white;
                        border: none;
                        border-radius: 10px;
                        cursor: pointer;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        margin-bottom: 15px;
                    }}

                    .compare-button:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 5px 15px rgba(104, 211, 145, 0.4);
                    }}

                    .compare-button:disabled {{
                        background: {light_grey};
                        cursor: not-allowed;
                        transform: none;
                        box-shadow: none;
                    }}

                    .clear-selection-btn {{
                        width: 100%;
                        padding: 8px;
                        background: transparent;
                        color: {red};
                        border: 1px solid #f56565;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.2s ease;
                    }}

                    .clear-selection-btn:hover {{
                        background: {red};
                        color: white;
                    }}
                    
                    .tooltip-content {{
                        background: {dark_white};
                        padding: 15px;
                        border-radius: 8px;                      
                    }}
                    
                    .tooltip-title {{
                        font-size: 16px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    
                    .tooltip-stats {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 10px;
                        margin-bottom: 15px;
                    }}
                    
                    .stat-item {{
                        background: rgba(255,255,255,0.05);
                        padding: 8px;
                        border-radius: 4px;
                        text-align: center;
                    }}
                    
                    .stat-value {{
                        font-size: 18px;
                        font-weight: bold;
                        color: #38a169;
                    }}
                    
                    .stat-label {{
                        font-size: 12px;
                        color: #718096;
                        margin-top: 2px;
                    }}
                    
                    .chart-container {{
                        height: 250px;
                        margin-top: 10px;
                    }}
                    
                    .legend {{
                        background: {dark_white};
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }}
                    
                    .legend-title {{
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    
                    .legend-item {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 5px;
                        font-size: 12px;
                    }}
                    
                    .legend-color {{
                        width: 20px;
                        height: 15px;
                        margin-right: 8px;
                        border-radius: 2px;
                    }}

                    .legend-color-bar {{
                        height: 20px;
                        border-radius: 4px;
                        margin-bottom: 6px;
                    }}
                    
                    .control-panel {{
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        z-index: 1000;
                        padding: 15px;
                        border-radius: 8px;
                        backdrop-filter: blur(10px);
                        min-width: 200px;
                        background: {dark_white};
                    }}
                    
                    .level-indicator {{
                        font-weight: bold;
                        font-size: 14px;
                        margin-bottom: 8px;
                    }}
                    
                    .back-button {{
                        background: {blue};
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 6px;
                        cursor: pointer;
                        transition: all 0.2s;
                        width: 100%;
                        margin-top: 8px;
                    }}
                    
                    .back-button:hover {{
                        background: {blue_light};
                        transform: translateY(-1px);
                    }}
                    
                    .stats-summary {{
                        margin-top: 10px;
                        font-size: 12px;
                        color: #4a5568;
                    }}
                    
                    .stats-summary div {{
                        margin: 2px 0;
                    }}

                    .detail-overlay {{
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        bottom: 0;
                        width: calc(100% - 500px); /* Sidebar width is 500px */
                        background: rgba(255, 255, 255);
                        color: #2d3748;
                        backdrop-filter: blur(5px);
                        z-index: 10000;
                        transition: transform 0.3s ease;
                        overflow-y: auto;
                        scroll-behavior: smooth;
                    }}

                    .detail-overlay.show {{
                        display: block;
                    }}

                    .detail-overlay-close {{
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        background: {red};
                        border: none;
                        color: #fff;
                        font-size: 16px;
                        padding: 8px 12px;
                        border-radius: 4px;
                        cursor: pointer;
                    }}

                    .detail-overlay-content {{
                        height: 100%;
                        padding: 40px;
                    }}

                    .detail-overlay-content hr{{
                        color: {dark_primary}
                    }}

                    .detail-overlay-header {{
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 10px;
                    }}

                    .detail-overlay-stats {{
                        display: flex;
                        gap: 20px;
                        margin-bottom: 12px;
                    }}

                    .detail-overlay-stat-item {{
                        flex: 1;
                        background: rgba(255,255,255,0.5);
                        padding: 15px;
                        border-radius: 6px;
                        text-align: center;
                    }}

                    .detail-overlay-stat-label {{
                        font-size: 14px;
                        color: #4a5568;
                        margin-top: 4px;
                    }}

                    .detail-overlay-stat-value {{
                        font-size: 20px;
                        font-weight: bold;
                        color: #38a169;
                    }}

                    .detail-overlay-chart {{
                        height: 500px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 15px;
                        padding: 20px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        margin-top: 12px;
                    }}

                    .comparison-overlay {{
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        bottom: 0;
                        width: calc(100% - 500px); /* Sidebar width is 500px */
                        background: rgba(255, 255, 255, 0.95);
                        color: #2d3748;
                        backdrop-filter: blur(5px);
                        z-index: 10000;
                        overflow-y: auto;
                        scroll-behavior: smooth;
                    }}

                    .comparison-overlay-content {{
                        height: 100%;
                        padding: 40px;
                        overflow-y: auto;
                        backdrop-filter: blur(15px);
                        border-radius: 0 20px 20px 0;
                        border-right: 1px solid rgba(255, 255, 255, 0.1);
                        border-top: none;
                        border-bottom: none;
                        border-left: none;
                    }}

                    .comparison-overlay-header {{
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 30px;
                    }}

                    .comparison-overlay-chart {{
                        height: 500px;
                        background: rgba(255, 255, 255, 0.05);
                        border-radius: 15px;
                        padding: 20px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                    }}

                    /* Selected region highlighting */
                    .leaflet-interactive.selected-region {{
                        stroke: #68d391 !important;
                        stroke-width: 4px !important;
                        stroke-dasharray: none !important;
                    }}

                    .simulations-overview {{
                        margin-top: 30px;
                        margin-bottom: 30px;
                    }}

                    .simulations-overview img {{
                        display: block;
                        max-width: 100%;
                        border-radius: 10px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                    }}

                    .help-icon {{
                        position: relative;
                        display: inline-block;
                        cursor: help;
                    }}

                    .help-button {{
                        width: 24px;
                        height: 24px;
                        border-radius: 50%;
                        background-color: {dark_primary};
                        color: white;
                        border: none;
                        font-size: 14px;
                        font-weight: bold;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: background-color 0.2s ease;
                    }}

                    .help-button:hover {{
                        background-color: #5a6268;
                    }}

                    .help-tooltip {{
                        position: absolute;
                        top: 100%;
                        left: 50%;
                        transform: translateX(-50%);
                        margin-top: 8px;
                        background-color: #343a40;
                        color: white;
                        padding: 12px 16px;
                        border-radius: 6px;
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                        z-index: 1000;
                        width: 280px;
                        opacity: 0;
                        visibility: hidden;
                        transition: opacity 0.2s ease, visibility 0.2s ease;
                        font-size: 14px;
                        line-height: 1.4;
                    }}

                    .help-tooltip::before {{
                        content: '';
                        position: absolute;
                        top: -6px;
                        left: 50%;
                        transform: translateX(-50%);
                        width: 0;
                        height: 0;
                        border-left: 6px solid transparent;
                        border-right: 6px solid transparent;
                        border-bottom: 6px solid #343a40;
                    }}

                    .help-icon:hover .help-tooltip {{
                        opacity: 1;
                        visibility: visible;
                    }}

                    .help-tooltip ul {{
                        margin: 0;
                        padding-left: 16px;
                    }}

                    .help-tooltip li {{
                        margin-bottom: 8px;
                    }}

                    .help-tooltip li:last-child {{
                        margin-bottom: 0;
                    }}

                    .title-container {{
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }}


                </style>
            </head>
            <body>
                <div class="container">
                    <div class="map-container">
                        <div id="map"></div>
                        <div class="control-panel">
                            <div class="level-indicator title" id="levelIndicator">Loading...</div>
                            <div class="stats-summary" id="statsSummary"></div>
                            <button class="back-button" id="backButton" style="display: none;">← Back</button>
                        </div>
                    </div>
                    
                    <div class="sidebar">
                        <div class="sidebar-header">
                            <div class="title-container">
                                <h2 class="title">{title}</h2>
                                <div class="help-icon">
                                    <button class="help-button">?</button>
                                    <div class="help-tooltip">
                                        <ul>
                                            <li>Click on a region to view nested regions.</li>
                                            <li>Double click on a region to see the regions stat details.</li>
                                            <li>Use CTRL+Click to select multiple regions for comparing their data.</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            <div class="breadcrumb" id="breadcrumb"></div>
                        </div>
                        
                        <div class="sidebar-content">
                            <div id="hoverInfo" style="display: none;">
                                <div class="tooltip-content">
                                    <div class="tooltip-title title" id="hoverTitle"></div>
                                    <div class="tooltip-stats">
                                        <div class="stat-item">
                                            <div class="stat-value" id="hoverValue1"></div>
                                            <div class="stat-label">Mean Yield (kg/ha)</div>
                                        </div>
                                        <div class="stat-item">
                                            <div class="stat-value" id="hoverValue2"></div>
                                            <div class="stat-label">Median Yield (kg/ha)</div>
                                        </div>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="hoverChart"></canvas>
                                    </div>
                                    <div class="legend-note" style="font-size: 11px; margin-top: 6px; color: #4a5568;">
                                        <p>★ denotes aggregated LAI series computed as the daily mean of all LAI series from subregions.</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="legend">
                                <div class="legend-title title">Yield Legend (kg/ha)</div>
                                <div id="legendItems"></div>
                                <div class="heatmap-selector">
                                    <label for="heatmapSelector">Select Heatmap Type:</label>
                                    <select id="heatmapSelector">
                                        <option value="estimated_mean_yield_kg_ha">Estimated Mean Yield (kg/ha)</option>
                                        <option value="estimated_median_yield_kg_ha">Estimated Median Yield (kg/ha)</option>
                                        <option value="error">Error (kg)</option>
                                        <option value="relative_error">Relative Error (%)</option>
                                    </select>
                                </div>
                            </div>

                            <div class="selected-regions">
                                <div class="legend-title title">Selected Regions (<span id="selectedCount">0</span>)</div>
                                <p>Select a number of regions using CTRL+Click to compare their LAI curves.</p>
                                <div id="selectedRegionsList"></div>
                                
                                <div class="comparison-controls">
                                    <div class="lai-selector">
                                        <label for="laiColumnSelect">Select LAI Column to Compare:</label>
                                        <select id="laiColumnSelect">
                                            {lai_column_options}
                                        </select>
                                    </div>
                                    
                                    <button class="compare-button" id="compareButton" disabled>Compare Selected Regions</button>
                                    <button class="clear-selection-btn" id="clearSelectionBtn">Clear Selection</button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div id="detailOverlay" class="detail-overlay">
                    <div class="detail-overlay-content">
                        <button id="overlayCloseBtn" class="detail-overlay-close">✕ Close</button>
                        <div class="detail-overlay-header title" id="overlayTitle">Region Details</div>
                        <div class="detail-overlay-stats">
                            <div class="detail-overlay-stat-item">
                                <div class="detail-overlay-stat-value" id="overlayValue1">0.0</div>
                                <div class="detail-overlay-stat-label">Mean Yield (kg/ha)</div>
                            </div>
                            <div class="detail-overlay-stat-item">
                                <div class="detail-overlay-stat-value" id="overlayValue2">0.0</div>
                                <div class="detail-overlay-stat-label">Median Yield (kg/ha)</div>
                            </div>
                            <div class="detail-overlay-stat-item">
                                <div class="detail-overlay-stat-value" id="overlayValue3">0.0</div>
                                <div class="detail-overlay-stat-label">Cropland Area (ha)</div>
                            </div>
                        </div>
                        <hr></hr>
                        <div class="detail-overlay-chart">
                            <canvas id="overlayChart"></canvas>
                        </div>
                         <div class="legend-note" style="font-size: 11px; margin-top: 6px; color: #4a5568;">
                            <p>★ denotes aggregated LAI series computed as the daily mean of all LAI series from subregions.</p>
                        </div>
                        <div id="simulationsOverview" class="simulations-overview" style="margin-top: 30px;">
                        </div>
                    </div>
                </div>

                <div id="comparisonOverlay" class="comparison-overlay">
                    <div class="comparison-overlay-content">
                        <button id="comparisonCloseBtn" class="detail-overlay-close">✕ Close</button>
                        <div class="comparison-overlay-header title" id="comparisonTitle">Region Comparison</div>
                        <div class="comparison-overlay-chart">
                            <canvas id="comparisonChart"></canvas>
                        </div>
                        <div id="comparisonStats" style="margin-top: 30px;"></div>
                    </div>
                </div>

                <script>
                    // Embedded data
                    const mapData = {json.dumps(levels, separators=(",", ":"), ensure_ascii=False)};

                    const allValueRanges = {json.dumps(value_ranges)}

                    let valueRange = {{
                        min_val: {value_ranges['estimated_mean_yield_kg_ha']['min_val']},
                        max_val: {value_ranges['estimated_mean_yield_kg_ha']['max_val']},
                        p95_val: {value_ranges['estimated_mean_yield_kg_ha']['p95_val']}
                    }};

                    const SERIES_COLORS = [
                        '#63b3ed', // blue
                        '#68d391', // green
                        '#f6ad55', // orange
                        '#d53f8c'  // pink/red
                    ];

                    let clickTimer = null;
                    const CLICK_DELAY = 300;  // milliseconds
                    
                    // Global state
                    let map;
                    let currentLevel = 0;
                    let currentParent = null;
                    let currentLayer = null;
                    let hoverChart = null;
                    let breadcrumbPath = [];
                    let heatmapType = 'estimated_mean_yield_kg_ha';

                    // Selection state for comparison
                    let selectedRegions = new Map(); // regionId -> name, properties, layer
                    let comparisonChart = null;
                    let availableLaiColumns = [];

                    // Color ramp where Chroma.js interpolates smoothly between these stops
                    const yieldRamp = chroma
                        .scale('viridis')
                        .domain([0, 1])
                        .mode('lrgb');

                    const errorRamp = chroma
                        .scale(['#2166ac', '#f7f7f7', '#b2182b']) // blue → white → red
                        .domain([-1, 0, 1])
                        .mode('lrgb');

                    let ramp = yieldRamp;

                    // Color scale function based on actual data range
                    function getColor(value) {{
                        let min = valueRange.min_val;
                        let max = valueRange.p95_val;
                        if (heatmapType.includes('error')) {{
                            const absMax = Math.max(Math.abs(min), Math.abs(max));
                            min = -absMax;
                            max = absMax;
                        }}

                        // Clamp value between min and max
                        const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
                       

                        return ramp(t).hex();
                    }}

                    const capitalize = str => str.charAt(0).toUpperCase() + str.slice(1);

                    // Style each feature using getColor(...)
                    function style(feature) {{
                        return {{
                            fillColor: getColor(feature.properties[heatmapType]),
                            weight: 2,
                            opacity: 1,
                            color: 'white',
                            fillOpacity: 0.85
                        }};
                    }}

                    // Event handlers
                    function highlightFeature(e) {{
                        const layer = e.target;
                        const props = layer.feature.properties;
                        
                        layer.setStyle({{
                            weight: 3,
                            color: '{dark_primary}',
                            fillOpacity: 0.95
                        }});

                        if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {{
                            layer.bringToFront();
                        }}

                        showHoverInfo(props);
                    }}

                    function resetHighlight(e) {{
                        currentLayer.resetStyle(e.target);
                        hideHoverInfo();
                    }}

                    let overlayChart = null;

                    function showOverlay(props) {{
                        document.getElementById('overlayTitle').textContent = props.name;
                        document.getElementById('overlayValue1').textContent = props.estimated_mean_yield_kg_ha.toFixed(1);
                        document.getElementById('overlayValue2').textContent = (props.estimated_median_yield_kg_ha || 0).toFixed(1);
                        document.getElementById('overlayValue3').textContent = (props.total_area || 0).toFixed(1);

                        updateOverlayChart(props.timeSeries, props.dateLabels, props.interpolationFlags, props.isAggregated);

                        // Show simulation image if present
                        const simContainer = document.getElementById('simulationsOverview');
                        simContainer.innerHTML = '';
                        if (props.simulationsImgPath) {{
                            const img = document.createElement('img');
                            img.src = props.simulationsImgPath;
                            img.alt = 'Simulation Overview';
                            img.style.maxWidth = '100%';
                            img.style.borderRadius = '8px';
                            img.style.marginTop = '20px';
                            img.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
                            simContainer.appendChild(img);
                        }}

                        const overlay = document.getElementById('detailOverlay');
                        overlay.classList.add('show');
                    }}

                    function hideOverlay() {{
                        // Destroy chart instance if exists
                        if (overlayChart) {{
                            overlayChart.destroy();
                            overlayChart = null;
                        }}
                        document.getElementById('detailOverlay').classList.remove('show');
                    }}                    

                    function updateOverlayChart(timeSeriesObj, dateLabels, interpolationFlags, isAggregated) {{
                        const ctx = document.getElementById('overlayChart').getContext('2d');

                        // Destroy any existing instance
                        if (overlayChart) {{
                            overlayChart.destroy();
                            overlayChart = null;
                        }}

                        const datasets = [];
                        const seriesNames = Object.keys(timeSeriesObj || {{}});

                        console.log(interpolationFlags);

                        seriesNames.forEach((seriesName, idx) => {{
                            const rawData = timeSeriesObj[seriesName] || [];
                            const color = SERIES_COLORS[idx % SERIES_COLORS.length];

                            datasets.push({{
                                label: seriesName + (isAggregated ? ' ★' : ''),
                                data: rawData,
                                borderColor: color,
                                backgroundColor: color.replace(/([0-9a-f]{{6}})$/, '20'), // ~12% opacity
                                borderWidth: 2,
                                fill: false,
                                tension: 0.4,
                                pointRadius: 0,      // no default points on the line
                                pointHoverRadius: 0, // no hover points on the curve
                                showLine: true       // draw the connecting line
                            }});

                            datasets.push({{
                                label: seriesName + ' (Day with RS Data)',
                                data: rawData.map((val, i) => (interpolationFlags?.[i] === 0 ? val : null)),
                                showLine: false,                 // no connecting line
                                pointRadius: 4,                  // radius of each dot
                                pointHoverRadius: 6,             // hoverradius
                                pointBackgroundColor: color,     // <— must be set to force draw circles
                                pointBorderColor: color,         // <— ditto
                                borderWidth: 0
                            }});
                        }});

                        overlayChart = new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: dateLabels || seriesNames.map((_, i) => `T${{i+1}}`),
                                datasets: datasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {{
                                    intersect: false,
                                    mode: 'index'
                                }},
                                plugins: {{
                                    legend: {{
                                        display: true,
                                        position: 'bottom',
                                        labels: {{
                                            color: '#000',
                                            font: {{ size: 11 }},
                                            filter: function(legendItem, chartData) {{
                                                const txt = legendItem.text || '';
                                                return !txt.includes('Day with RS');
                                            }},
                                            usePointStyle: true,
                                            pointStyle: 'circle',
                                            generateLabels: function(chart) {{
                                                const datasets = chart.data.datasets;
                                                return datasets
                                                    .map((dataset, i) => {{
                                                        const label = dataset.label || '';
                                                        if (label.includes('Day with RS')) return null;

                                                        return {{
                                                            text: label,
                                                            fillStyle: dataset.borderColor,
                                                            strokeStyle: dataset.borderColor,
                                                            lineWidth: 1,
                                                            pointStyle: 'circle',
                                                            hidden: !chart.isDatasetVisible(i),
                                                            datasetIndex: i
                                                        }};
                                                    }})
                                                    .filter(Boolean);
                                            }}
                                        }}
                                    }},
                                    tooltip: {{
                                        backgroundColor: 'rgba(0,0,0,0.8)',
                                        titleColor: '#63b3ed',
                                        bodyColor: '#fff'
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 10 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 14, weight: 'bold'}}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI',
                                            color: '{dark_primary}',
                                            font: {{ size: 21, weight: 'bold' }}
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    function updateRegionSelection(regionId, regionName, properties, layer, add = true) {{
                        if (add) {{
                            selectedRegions.set(regionId, {{
                                name: regionName,
                                properties: properties,
                                layer: layer
                            }});
                            // Add visual selection indicator
                            layer.getElement().classList.add('selected-region');
                        }} else {{
                            selectedRegions.delete(regionId);
                            // Remove visual selection indicator
                            if (layer && layer.getElement()) {{
                                layer.getElement().classList.remove('selected-region');
                            }}
                        }}
                        
                        updateSelectedRegionsUI();
                        updateCompareButtonState();
                    }}

                    function updateCompareButtonState() {{
                        const compareBtn = document.getElementById('compareButton');
                        const laiSelect = document.getElementById('laiColumnSelect');
                        
                        const canCompare = selectedRegions.size >= 2 && laiSelect.value !== '';
                        compareBtn.disabled = !canCompare;
                    }}

                    function updateSelectedRegionsUI() {{
                        const listContainer = document.getElementById('selectedRegionsList');
                        const countSpan = document.getElementById('selectedCount');
                        
                        countSpan.textContent = selectedRegions.size;
                        listContainer.innerHTML = '';
                        
                        selectedRegions.forEach((regionData, regionId) => {{
                            const item = document.createElement('div');
                            item.className = 'selected-region-item';
                            
                            const nameSpan = document.createElement('span');
                            nameSpan.className = 'selected-region-name';
                            nameSpan.textContent = regionData.name;
                            
                            const removeBtn = document.createElement('button');
                            removeBtn.className = 'remove-region-btn';
                            removeBtn.textContent = 'x';
                            removeBtn.onclick = () => {{
                                updateRegionSelection(regionId, regionData.name, regionData.properties, regionData.layer, false);
                            }};
                            
                            item.appendChild(nameSpan);
                            item.appendChild(removeBtn);
                            listContainer.appendChild(item);
                        }});
                    }}

                    function clearSelection() {{
                        // Remove visual indicators from all selected layers
                        selectedRegions.forEach((regionData, regionId) => {{
                            if (regionData.layer && regionData.layer.getElement()) {{
                                regionData.layer.getElement().classList.remove('selected-region');
                            }}
                        }});
                        
                        selectedRegions.clear();
                        updateSelectedRegionsUI();
                        updateCompareButtonState();
                    }}

                    function showComparison() {{
                        const selectedColumn = document.getElementById('laiColumnSelect').value;
                        if (!selectedColumn || selectedRegions.size < 2) return;
                        
                        const datasets = [];
                        let dateLabels = null;
                        
                        Array.from(selectedRegions.values()).forEach((regionData, index) => {{
                            const properties = regionData.properties;
                            const timeSeries = properties.timeSeries;
                            const interpolationFlags = properties.interpolationFlags;
                            
                            if (timeSeries && timeSeries[selectedColumn]) {{
                                const data = timeSeries[selectedColumn];
                                const color = SERIES_COLORS[index % SERIES_COLORS.length];
                                
                                // Use date labels from first region
                                if (!dateLabels) {{
                                    dateLabels = properties.dateLabels || data.map((_, i) => `T${{i+1}}`);
                                }}
                                
                                // Main line dataset
                                datasets.push({{
                                    label: regionData.name + (properties.isAggregated ? ' ★' : ''),
                                    data: data,
                                    borderColor: color,
                                    backgroundColor: color.replace(/([0-9a-f]{{6}})$/, '20'),
                                    borderWidth: 3,
                                    fill: false,
                                    tension: 0.4,
                                    pointRadius: 0,
                                    pointHoverRadius: 5
                                }});
                                
                                // Points for actual RS data
                                datasets.push({{
                                    label: regionData.name + ' (RS Data)',
                                    data: data.map((val, i) => interpolationFlags?.[i] === 0 ? val : null),
                                    showLine: false,
                                    pointRadius: 4,
                                    pointHoverRadius: 6,
                                    pointBackgroundColor: color,
                                    pointBorderColor: color,
                                    borderWidth: 0
                                }});
                            }}
                        }});
                        
                        // Update title and show overlay
                        document.getElementById('comparisonTitle').textContent = 
                            `Comparison: ${{selectedColumn}} (${{selectedRegions.size}} regions)`;
                        
                        updateComparisonChart(datasets, dateLabels);

                        // Update stats container with selected regions estimated yield
                        const statsContainer = document.getElementById('comparisonStats');
                            statsContainer.innerHTML = ''; // Clear previous stats

                            Array.from(selectedRegions.values()).forEach((regionData, index) => {{
                                const props = regionData.properties;
                                const color = SERIES_COLORS[index % SERIES_COLORS.length];
                                const name = regionData.name + (props.isAggregated ? ' ★' : '');

                                const mean = props.estimated_mean_yield_kg_ha?.toFixed(1) || '-';
                                const median = props.estimated_median_yield_kg_ha?.toFixed(1) || '-';

                                const regionStats = document.createElement('div');
                                regionStats.style.marginTop = '10px';
                                regionStats.style.display = 'flex';
                                regionStats.style.alignItems = 'center';
                                regionStats.style.gap = '10px';

                                regionStats.innerHTML = `
                                    <div style="width: 12px; height: 12px; background-color: ${{color}}; border-radius: 2px;"></div>
                                    <strong style="color: ${{color}};">${{name}}</strong>
                                    <span style="color: #4a5568;">Mean Yield: <strong>${{mean}}</strong> kg/ha</span>
                                    <span style="color: #4a5568;">Median Yield: <strong>${{median}}</strong> kg/ha</span>
                                `;

                                statsContainer.appendChild(regionStats);
                            }});
                        document.getElementById('comparisonOverlay').style.display = 'block';
                    }}

                    function updateComparisonChart(datasets, dateLabels) {{
                        const ctx = document.getElementById('comparisonChart').getContext('2d');
                        
                        if (comparisonChart) {{
                            comparisonChart.destroy();
                            comparisonChart = null;
                        }}
                        
                        comparisonChart = new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: dateLabels,
                                datasets: datasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {{
                                    intersect: false,
                                    mode: 'index'
                                }},
                                plugins: {{
                                    legend: {{
                                        display: true,
                                        position: 'bottom',
                                        labels: {{
                                            color: '#4a5568',
                                            font: {{ size: 12 }},
                                            filter: function(legendItem, chartData) {{
                                                const txt = legendItem.text || '';
                                                return !txt.includes('RS Data');
                                            }}
                                        }}
                                    }},
                                    tooltip: {{
                                        backgroundColor: 'rgba(0,0,0,0.8)',
                                        titleColor: '#63b3ed',
                                        bodyColor: '#fff'
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 11 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 11 }}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI Value',
                                            color: '#4a5568'
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    function hideComparisonOverlay() {{
                        if (comparisonChart) {{
                            comparisonChart.destroy();
                            comparisonChart = null;
                        }}
                        document.getElementById('comparisonOverlay').style.display = 'none';
                    }}

                    // Navigation functions
                    function drillDown(parentId, parentName) {{
                        clearSelection();
                        // Check if we have child data for this parent (drill into detailed view of same level)
                        console.log('mapData', mapData);
                        console.log(parentId);
                        if (mapData.level_mappings[parentId]) {{
                            console.log('has mappings');
                            // We're drilling into a subset of the current level
                            currentLevel++;
                            currentParent = parentId;
                            breadcrumbPath.push(capitalize(parentName));
                            
                            console.log('subset drilldown', parentId, parentName);
                            loadLevel(mapData.level_mappings[parentId]);
                            updateUI();
                            
                            // Fit map to new bounds
                            const bounds = L.geoJSON(mapData.level_mappings[parentId]).getBounds();
                            map.fitBounds(bounds, {{ padding: [20, 20] }});
                            
                            document.getElementById('backButton').style.display = 'block';
                        }} else if (currentLevel < mapData.levels.length - 1) {{
                            // Move to next aggregation level
                            currentLevel++;
                            currentParent = null; // Reset parent when moving to new level
                            breadcrumbPath.push(capitalize(parentName));
                            
                            console.log('subset drilldown', parentId, parentName);
                            loadLevel(mapData.level_data[`level_${{currentLevel}}`]);
                            updateUI();
                            
                            document.getElementById('backButton').style.display = 'block';
                        }}
                    }}

                    function goBack() {{
                        clearSelection();
                        if (breadcrumbPath.length > 0) {{
                            breadcrumbPath.pop();
                            
                            if (breadcrumbPath.length === 0) {{
                                // Back to top level
                                currentLevel = 0;
                                currentParent = null;
                                document.getElementById('backButton').style.display = 'none';
                                loadLevel(mapData.level_data[`level_0`]);
                                
                                // Reset map view
                                const bounds = L.geoJSON(mapData.level_data.level_0).getBounds();
                                map.fitBounds(bounds, {{ padding: [20, 20] }});
                            }} else {{
                                // Go back one step in current level or previous level
                                if (currentParent) {{
                                    // We were in a subset, go back to full level
                                    currentLevel--;
                                    currentParent = null;
                                    loadLevel(mapData.level_data[`level_${{currentLevel}}`]);
                                }} else if (currentLevel > 0) {{
                                    // Go back to previous level
                                    currentLevel--;
                                    loadLevel(mapData.level_data[`level_${{currentLevel}}`]);
                                }}
                                const bounds = L.geoJSON(mapData.level_data[`level_${{currentLevel}}`]).getBounds();
                                map.fitBounds(bounds, {{ padding: [20, 20] }});
                            }}
                            
                            updateUI();
                        }}
                    }}

                    function onEachFeature(feature, layer) {{
                        layer.on({{
                        mouseover: highlightFeature,
                        mouseout: resetHighlight,

                        // SINGLE-click: start a timer. If no dblclick arrives within 300ms, drill down.
                        click: function(e) {{
                            const props = e.target.feature.properties;
                            const regionId = props.id;
                            const regionName = props.name;
                            
                            if (clickTimer === null) {{
                                clickTimer = setTimeout(() => {{
                                    // Check if Ctrl key was held during click
                                    if (e.originalEvent.ctrlKey || e.originalEvent.metaKey) {{
                                        // Toggle region selection
                                        if (selectedRegions.has(regionId)) {{
                                            updateRegionSelection(regionId, regionName, props, layer, false);
                                        }} else {{
                                            updateRegionSelection(regionId, regionName, props, layer, true);
                                        }}
                                    }} else {{
                                        // Normal drill down behavior
                                        if (currentLevel < mapData.levels.length - 1) {{
                                            drillDown(regionId, regionName);
                                        }}
                                    }}
                                    clickTimer = null;
                                }}, CLICK_DELAY);
                            }}
                        }},

                        // DOUBLE-click: clear the single-click timer (if still pending), then show overlay.
                        dblclick: function(e) {{
                            const props = e.target.feature.properties;
                            if (clickTimer) {{
                                clearTimeout(clickTimer);
                                clickTimer = null;
                            }}
                            showOverlay(props);
                        }}
                        }});
                    }}

                    // Data loading
                    function loadLevel(levelData) {{
                        if (currentLayer) {{
                            map.removeLayer(currentLayer);
                        }}

                        console.log(currentLayer, levelData);

                        currentLayer = L.geoJSON(levelData, {{
                            style: style,
                            onEachFeature: onEachFeature
                        }}).addTo(map);

                        updateLegend();
                        updateStats(levelData);
                    }}

                    // UI updates
                    function updateUI() {{
                        updateBreadcrumb();
                        updateLevelIndicator();
                    }}

                    function updateBreadcrumb() {{
                        const breadcrumb = document.getElementById('breadcrumb');
                        breadcrumb.innerHTML = '';
                        
                        const fullPath = [mapData.levels[0].name, ...breadcrumbPath];
                        
                        fullPath.forEach((item, index) => {{
                            if (index > 0) {{
                                const separator = document.createElement('span');
                                separator.className = 'breadcrumb-separator';
                                separator.textContent = '>';
                                breadcrumb.appendChild(separator);
                            }}
                            
                            const breadcrumbItem = document.createElement('span');
                            breadcrumbItem.className = 'breadcrumb-item';
                            breadcrumbItem.textContent = capitalize(item);
                            
                            if (index < fullPath.length - 1) {{
                                breadcrumbItem.onclick = () => {{
                                    // Navigate back to this level
                                    const targetLevel = index;
                                    while (currentLevel > targetLevel) {{
                                        goBack();
                                    }}
                                }};
                            }}
                            
                            breadcrumb.appendChild(breadcrumbItem);
                        }});
                    }}

                    function updateLevelIndicator() {{
                        const indicator = document.getElementById('levelIndicator');
                        const levelInfo = mapData.levels[currentLevel];
                        indicator.textContent = `Level ${{currentLevel + 1}}: ${{capitalize(levelInfo.name)}}`;
                    }}

                    function updateStats(levelData) {{
                        const values = levelData.features.map(f => f.properties.estimated_mean_yield_kg_ha);
                        const avgYield = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
                        const minYield = Math.min(...values).toFixed(1);
                        const maxYield = Math.max(...values).toFixed(1);
                        
                        const statsDiv = document.getElementById('statsSummary');
                        statsDiv.innerHTML = `
                            <div><b>Regions:</b> ${{values.length}}</div>
                           
                            <div><b>Range:</b> ${{minYield}} - ${{maxYield}} kg/ha</div>
                        `;
                    }}

                    function selectHeatmapType(newHeatmapType) {{
                        heatmapType = newHeatmapType;
                        if (newHeatmapType.includes('error')) {{
                            ramp = errorRamp;
                        }} else{{
                            ramp = yieldRamp
                        }}

                        console.log(allValueRanges);
                        console.log(newHeatmapType);
                        valueRange = allValueRanges[newHeatmapType];
                        console.log(valueRange);

                        updateLegend();
                        if (currentLayer) {{
                            currentLayer.setStyle(style);
                        }}
                    }}

                    function updateLegend() {{
                        const legendItems = document.getElementById('legendItems');
                        legendItems.innerHTML = '';

                        const min = valueRange.min_val.toFixed(0);
                        const max = valueRange.p95_val.toFixed(0);

                        const gradientDiv = document.createElement('div');
                        gradientDiv.style.height = '20px';
                        gradientDiv.style.background = `linear-gradient(to right, 
                            ${{ramp(0).hex()}}, 
                            ${{ramp(0.2).hex()}}, 
                            ${{ramp(0.4).hex()}}, 
                            ${{ramp(0.6).hex()}}, 
                            ${{ramp(0.8).hex()}}, 
                            ${{ramp(1).hex()}})`;
                        gradientDiv.style.marginBottom = '5px';

                        const labelContainer = document.createElement('div');
                        labelContainer.style.display = 'flex';
                        labelContainer.style.justifyContent = 'space-between';
                        labelContainer.style.fontSize = '12px';
                        labelContainer.style.color = '#4a5568';

                        const minLabel = document.createElement('span');
                        minLabel.textContent = `${{min}} kg/ha`;
                        const maxLabel = document.createElement('span');
                        maxLabel.textContent = `≥ ${{max}} kg/ha`;

                        labelContainer.appendChild(minLabel);
                        labelContainer.appendChild(maxLabel);

                        legendItems.appendChild(gradientDiv);
                        legendItems.appendChild(labelContainer);
                    }}

                    // Hover info functions
                    function showHoverInfo(props) {{
                        document.getElementById('hoverInfo').style.display = 'block';
                        document.getElementById('hoverTitle').textContent = capitalize(props.name);
                        document.getElementById('hoverValue1').textContent = props.estimated_mean_yield_kg_ha.toFixed(1);
                        document.getElementById('hoverValue2').textContent = (props.estimated_median_yield_kg_ha || 0).toFixed(1);
                        
                        updateHoverChart(props.timeSeries, props.name, props.dateLabels, props.interpolationFlags, props.isAggregated);
                    }}

                    function hideHoverInfo() {{
                        document.getElementById('hoverInfo').style.display = 'none';
                    }}

                    function updateHoverChart(timeSeriesObj, label, dateLabels, interpolationFlags, isAggregated) {{
                        const ctx = document.getElementById('hoverChart').getContext('2d');

                        // Destroy any existing hoverChart
                        if (hoverChart) {{
                            hoverChart.destroy();
                            hoverChart = null;
                        }}

                        // Build two datasets PER SERIES
                        const datasets = [];
                        const seriesNames = Object.keys(timeSeriesObj || {{}});

                        seriesNames.forEach((seriesName, idx) => {{
                            const rawData = timeSeriesObj[seriesName] || [];
                            const color = SERIES_COLORS[idx % SERIES_COLORS.length];

                            datasets.push({{
                                label: seriesName + (isAggregated ? ' ★' : ''),
                                data: rawData,
                                borderColor: color,
                                backgroundColor: color.replace(/([0-9a-f]{{6}})$/, '20'),
                                borderWidth: 2,
                                fill: false,
                                tension: 0.4,
                                pointRadius: 0,
                                pointHoverRadius: 0
                            }});
                        }});

                        hoverChart = new Chart(ctx, {{
                            type: 'line',
                            data: {{
                                labels: dateLabels || seriesNames.map((_, i) => `T${{i+1}}`),
                                datasets: datasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                interaction: {{
                                    intersect: false,
                                    mode: 'index'
                                }},
                                plugins: {{
                                    legend: {{
                                        display: true,
                                        position: 'bottom',
                                        labels: {{
                                            color: '#000',
                                            font: {{ size: 11 }},
                                            filter: function(legendItem, chartData) {{
                                                const txt = legendItem.text || '';
                                                return !txt.includes('Day with RS');
                                            }},
                                            usePointStyle: true,
                                            pointStyle: 'circle',
                                            generateLabels: function(chart) {{
                                                const datasets = chart.data.datasets;
                                                return datasets
                                                    .map((dataset, i) => {{
                                                        const label = dataset.label || '';
                                                        if (label.includes('Day with RS')) return null;

                                                        return {{
                                                            text: label,
                                                            fillStyle: dataset.borderColor,
                                                            strokeStyle: dataset.borderColor,
                                                            lineWidth: 1,
                                                            pointStyle: 'circle',
                                                            hidden: !chart.isDatasetVisible(i),
                                                            datasetIndex: i
                                                        }};
                                                    }})
                                                    .filter(Boolean);
                                            }}
                                        }}
                                    }},
                                    tooltip: {{
                                        backgroundColor: 'rgba(0,0,0,0.8)',
                                        titleColor: '#63b3ed',
                                        bodyColor: '#fff'
                                    }}
                                }},
                                scales: {{
                                    x: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 10 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#4a5568',
                                            font: {{ size: 10 }}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI Value',
                                            color: '#4a5568'
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    // Event listeners
                    document.getElementById('heatmapSelector').addEventListener("change", function () {{
                        selectHeatmapType(this.value);
                    }});

                    document.addEventListener('keydown', function(event) {{
                        if (event.key === 'Escape') {{
                            hideOverlay();
                            hideComparisonOverlay();
                        }}
                    }});

                    document.getElementById('backButton').onclick = goBack;
                    document.getElementById('clearSelectionBtn').onclick = clearSelection;
                    document.getElementById('laiColumnSelect').onchange = updateCompareButtonState;
                    document.getElementById('compareButton').onclick = showComparison;

                    document.getElementById('overlayCloseBtn').onclick = hideOverlay;
                    document.getElementById('comparisonCloseBtn').onclick = hideComparisonOverlay;

                    // Initialize map
                    window.addEventListener('load', function () {{
                        setTimeout(() => {{
                        map = L.map('map');
                        map.doubleClickZoom.disable();

                        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                            attribution: '&copy; OpenStreetMap contributors & Carto',
                            subdomains: 'abcd',
                            maxZoom: 19
                            }}).addTo(map);

                        // Load initial (level 0)
                        loadLevel(mapData.level_data.level_0);

                        // Fit map to initial bounds
                        const bounds = L.geoJSON(mapData.level_data.level_0).getBounds();
                        map.fitBounds(bounds, {{ padding: [20, 20] }});

                        updateUI();
                        }}, 100);
                    }});
                </script>
            </body>
            </html>'''
        return template
    
    def generate_map(self, title):
        """
        Generate the complete interactive map.
        """
        
        print("Creating hierarchical data structure...")
        levels = self.create_hierarchical_data()
        
        print("Generating HTML template...")
        html_content = self.generate_html_template(levels, title)
        
        output_path = os.path.join(self.output_dir, 'vercye_results_map.html')
        print(f"Writing to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        if self.zip:
            shutil.make_archive(self.output_dir, 'zip', self.output_dir)
            print(f"Interactive map generated and zipped successfully: {self.output_dir}.zip")
        else:
            print(f"Interactive map generated successfully: {output_path}")

        return output_path
    
    def load_aggregated_estimates(self):
        agg_estimates = {}
        for idx, level in enumerate(self.aggregation_levels):
            
            agg_estimates_df = pd.read_csv(level['agg_estimates_file'])

            if os.path.exists(level['errors_file']):
                errors_df = pd.read_csv(level['errors_file'])
                agg_estimates_df = agg_estimates_df.merge(errors_df, on='region', how='left')

            # create dict with region names as keys and aggregated data as values
            agg_data = {}
            for _, row in agg_estimates_df.iterrows():
                region_name = row['region']
                agg_data[region_name] = {
                    'estimated_mean_yield_kg_ha': row['mean_yield_kg_ha'],
                    'estimated_median_yield_kg_ha': row['median_yield_kg_ha'],
                    'total_area_ha': row['total_area_ha']
                }

                if 'error_kg_ha' in row and 'rel_error_percent' in row:
                    agg_data[region_name]["error"] = row["error_kg_ha"]
                    agg_data[region_name]["relative_error"] =  row["rel_error_percent"]

                if 'reported_mean_yield_kg_ha' in row: 
                    agg_data[region_name]['reported_mean_yield_kg_ha'] = row['reported_mean_yield_kg_ha']



            agg_estimates[idx] = agg_data
        return agg_estimates
    
    def load_lai_data(self) -> pd.DataFrame:
        """
        Load LAI data from CSV files in the specified directory.
        
        Args:
            basedir_path: Study base directory
            
        Returns:
            Combined DataFrame with all LAI data
        """
        lai_files = list(Path(self.basedir_path).rglob('*_LAI_STATS.csv'))
        
        if not lai_files:
            raise FileNotFoundError(f"No LAI data files found in {self.basedir_path}")
        
        all_lai_data = []
        
        for file in lai_files:
            df = pd.read_csv(file, parse_dates=['Date'], dayfirst=True)
            # Extract region name from file path
            cleaned_region_name_vercye = file.stem.replace('_LAI_STATS', '')
            df['cleaned_region_name_vercye'] = cleaned_region_name_vercye
            all_lai_data.append(df)
        
        # Combine all data into single DataFrame
        combined_df = pd.concat(all_lai_data, ignore_index=True)
        return combined_df
    
    def load_sim_data(self):
        all_sim_data = {}

        sim_pngs = list(Path(self.basedir_path).rglob('*_yield_report.png'))
        for sim_png in sim_pngs:
            region_name = sim_png.stem.rsplit('_yield_report', 1)[0]

            if region_name in all_sim_data:
                raise ValueError(f"Duplicate report for '{region_name}':\n"
                                 f"  {all_sim_data[region_name]}\n  {sim_png}")
            
            # Copy all the simulations pngs to the output folder so it is self contained
            new_path = Path(self.output_dir) / sim_png.name
            shutil.copy(sim_png, new_path)

            all_sim_data[region_name] = str(new_path)
        
        return all_sim_data

def parse_agg_level(ctx, param, value):
    # Parses CLI parameters for aggregation level in passed order
    agg_dict = {}
    for item in value:
        try:
            level_name, column_name, csv_path = item.split(':', 2)
            csv_path = Path(csv_path)
            if not csv_path.exists():
                raise click.BadParameter(f"CSV path does not exist: {csv_path}")
            agg_dict[level_name] = (column_name, str(csv_path))
        except ValueError:
            raise click.BadParameter("Each --agg-level must be in format level:column:path")
    return agg_dict

def parse_lai_column(ctx, param, value):
    # Parses CLI parameters for lai columns
    internal = []
    display = []
    for item in value:
        try:
            internal_name, display_name = item.split(':', 1)
            internal.append(internal_name)
            display.append(display_name)
        except ValueError:
            raise click.BadParameter("Each --lai-column must be in format internal_name:display_name")
    return internal, display


@click.command()
@click.option('--shapefile-path', required=True, type=click.Path(exists=True), help='Path to the shapefile (.geojson).')
@click.option('--basedir-path', required=True, type=click.Path(exists=True), help='Path to the base directory of the timepoint in the study.')
@click.option('--adjusted/--no-adjusted', default=True, help='Include adjusted LAI columns.')
@click.option('--smoothed/--no-smoothed', default=True, help='Include smoothed LAI columns.')
@click.option('--agg-level', multiple=True, callback=parse_agg_level,
              help='Aggregation level in format level:column:path. Can be used multiple times.')
@click.option('--lai-column', multiple=True, default=DEFAULT_LAI_COLUMNS, callback=parse_lai_column,
              help='LAI column in format internal_name:display_name. Can be used multiple times.')
@click.option('--title', type=str, default='Yield Study', help='Title for the interactive map.')
@click.option('--output-dir', type=click.Path(),  help='Output directory.')
@click.option('--zipped', is_flag=True, help='Add flag for creating a .zip from the output directory.')
def main(shapefile_path, basedir_path, adjusted, smoothed, agg_level, lai_column, title, output_dir, zipped):

    lai_columns, lai_column_names = lai_column

    if adjusted:
        lai_columns = [col + ' Adjusted' for col in lai_columns]
        lai_column_names = [name + ' Adjusted' for name in lai_column_names]

    if smoothed:
        lai_columns += [col + ' Unsmoothed' for col in lai_columns]
        lai_column_names += [name + ' Smoothed' for name in lai_column_names] + [name + ' Unsmoothed' for name in lai_column_names]

    generator = InteractiveMapGenerator(
        shapefile_path=shapefile_path,
        basedir_path=basedir_path,
        output_dir=output_dir,
        lai_columns=lai_columns,
        lai_column_names=lai_column_names,
        agg_levels=agg_level,
        zip=zipped
    )

    try:
        output_file = generator.generate_map(title)
        click.echo(f"Map successfully created: {output_file}")
        click.echo("Open the HTML file in a web browser to view the interactive map.")
    except Exception as e:
        click.echo(f"Error generating map: {e}", err=True)
        raise e

if __name__ == "__main__":
    main()