from collections import defaultdict
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
    def __init__(self, shapefile_path: str, lai_basedir_path: str, output_path: str, lai_columns: List[str], lai_column_names: List[str], agg_levels: Dict[str, Tuple[str, str]], simplify_tolerance: float = 0.01):
        """Initialize the map generator."""
        self.shapefile_path = shapefile_path
        self.lai_basedir_path = lai_basedir_path
        self.output_path = output_path
        self.gdf = None
        self.lai_data = None
        self.aggregation_levels = []
        self.aggregated_estimates = {}
        self.lai_columns = lai_columns
        self.lai_column_names = lai_column_names
        self.simplify_tolerance = simplify_tolerance

        for agg_name, (agg_column, agg_estimates_fpath) in agg_levels.items():
            self.add_aggregation_level(agg_name, agg_column, agg_estimates_fpath)

        self.load_data()
        
    def load_data(self):
        """Load shapefile and LAI data."""
        print("Loading shapefile...")
        self.gdf = gpd.read_file(self.shapefile_path)

        # Todo add total_area_ha once fixed in prod
        keep_cols = ["cleaned_region_name_vercye", "estimated_mean_yield_kg_ha", "estimated_median_yield_kg_ha", "geometry"]

        # Append aggregation columns if they exist
        keep_cols += [self.aggregation_levels[i]['column'] for i in range(len(self.aggregation_levels))]

        self.gdf = self.gdf[keep_cols]
        #self.gdf["geometry"] = self.gdf["geometry"].simplify(tolerance=0.01)
        
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
        self.lai_data = self.load_lai_data(self.lai_basedir_path)
        print(f"Loaded LAI data")
        
    
    def add_aggregation_level(self, level_name: str, column_name: str, agg_estimate_fpath: str):
        """
        Add an aggregation level.
        
        Args:
            level_name: Display name for this level (e.g., "States", "Counties")
            column_name: Column name to aggregate by
        """
        
        self.aggregation_levels.append({
            'name': level_name,
            'column': column_name,
            'agg_estimates_file': agg_estimate_fpath
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
            date_labels.append(pd.to_datetime(date, format='%d/%m/%Y').strftime('%m/%d'))
            
            if group['interpolated'].isin([1]).any():
                # If any value is interpolated, mark as interpolated
                interpolation_flags.append(1)
            else:
                interpolation_flags.append(0)

        
        return aggregated_values, date_labels, interpolation_flags
    
    def create_geojson_level(self, level_idx: int) -> Dict[str, Any]:
        """
        Create GeoJSON data for a specific aggregation level.
        
        Args:
            level_idx: Index of the aggregation level (-1 for base level)
            
        Returns:
            GeoJSON FeatureCollection
        """
        if level_idx == -1:
            # Base level - use original data. Loading estimated yield from shapefile.
            features = []

            simplified = self.gdf.copy()
            simplified["geometry"] = simplified["geometry"].simplify(
                tolerance=self.simplify_tolerance,
                preserve_topology=True
            )
            
            for idx, row in simplified.iterrows():
                # Get LAI data for this region
                lai_values, date_labels, interpolation_flags = self.aggregate_lai_data([row['cleaned_region_name_vercye']])
                
                # Calculate additional metrics
                area = row.get('total_area_ha', 0)
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": f"region_{idx}",
                        "name": str(row['cleaned_region_name_vercye']),
                        "detail1": float(row['estimated_median_yield_kg_ha']),
                        "detail2": area,
                        "heatValue": float(row['estimated_mean_yield_kg_ha']),
                        "timeSeries": lai_values,
                        "dateLabels": date_labels,
                        "interpolationFlags": interpolation_flags,
                        "isAggregated": False
                    },
                    "geometry": row.geometry.__geo_interface__
                }
                features.append(feature)
            
            return {
                "type": "FeatureCollection",
                "features": features
            }
        
        else:
            # Aggregated level
            agg_col = self.aggregation_levels[level_idx]['column']

            # Loading aggregated values from already aggregated csv file
            
            # Group by aggregation column and aggregate
            grouped = self.gdf.groupby(agg_col).agg({
                'cleaned_region_name_vercye': lambda x: list(x),
                'geometry': lambda x: x.unary_union
            }).reset_index()

            
            features = []
            
            for idx, row in grouped.iterrows():
                merged_geom = row["geometry"]  # this is the full union of raw pieces
                simplified_geom = merged_geom.simplify(
                    tolerance=self.simplify_tolerance,
                    preserve_topology=True
                )
                # Get LAI data for all regions in this group
                lai_values, date_labels, interpolation_flags = self.aggregate_lai_data(row['cleaned_region_name_vercye'])

                region_name = str(row[agg_col])
                mean_estimated_yield_kg_ha = self.aggregated_estimates[level_idx][region_name]['estimated_mean_yield_kg_ha']
                median_estimated_yield_kg_ha = self.aggregated_estimates[level_idx][region_name]['estimated_median_yield_kg_ha']
                sum_area = self.aggregated_estimates[level_idx][region_name]['total_area_ha']
                                                                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "id": f"agg_{level_idx}_{idx}",
                        "name": str(row[agg_col]),
                        "heatValue": float(mean_estimated_yield_kg_ha),
                        "detail1": float(median_estimated_yield_kg_ha),
                        "detail2": sum_area,
                        "timeSeries": lai_values,
                        "dateLabels": date_labels,
                        "interpolationFlags": interpolation_flags,
                        "subregions": row['cleaned_region_name_vercye'],
                        "isAggregated": True
                    },
                    "geometry": simplified_geom.__geo_interface__
                }
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
        data_structure = {
            "levels": [],
            "level_data": {},
            "level_mappings": {}
        }
        
        # Add level information
        if self.aggregation_levels:
            # Add aggregation levels (highest to lowest)
            for i, level in enumerate(self.aggregation_levels):
                data_structure["levels"].append({
                    "index": i,
                    "name": level['name'],
                    "column": level['column']
                })
                
                # Create GeoJSON for this level
                level_geojson = self.create_geojson_level(i)
                data_structure["level_data"][f"level_{i}"] = level_geojson
        
        # Add base level (most detailed)
        base_level_idx = len(self.aggregation_levels)
        data_structure["levels"].append({
            "index": base_level_idx,
            "name": "Regions",
            "column": "cleaned_region_name_vercye"
        })
        
        base_geojson = self.create_geojson_level(-1)
        data_structure["level_data"][f"level_{base_level_idx}"] = base_geojson

        # Create mappings between levels
        if self.aggregation_levels:
            for i in range(len(self.aggregation_levels)):
                parent_level = self.create_geojson_level(i)
                
                # Create mapping from parent features to child regions
                for feature in parent_level["features"]:
                    parent_id = feature["properties"]["id"]
                    
                    # Determine what the next level should be
                    if i < len(self.aggregation_levels) - 1:
                        # Map to next aggregation level
                        next_level_data = self.create_geojson_level(i + 1)
                        # Filter features that belong to this parent
                        child_features = []
                        parent_column = self.aggregation_levels[i]['column']
                        next_column = self.aggregation_levels[i + 1]['column']
                        
                        # Get the parent region name
                        parent_name = feature["properties"]["name"]
                        
                        # Find all features in next level that belong to this parent
                        for next_feature in next_level_data["features"]:
                            # Check if this feature belongs to the parent
                            # This requires checking the original shapefile data
                            regions_in_next_feature = next_feature["properties"].get("subregions", [])
                            
                            # Check if any of these regions belong to our parent
                            parent_regions = feature["properties"].get("subregions", [])
                            if any(region in parent_regions for region in regions_in_next_feature):
                                child_features.append(next_feature)
                        
                        if child_features:
                            data_structure["level_mappings"][parent_id] = {
                                "type": "FeatureCollection",
                                "features": child_features
                            }
                    else:
                        # Map to base level (individual regions)
                        if "subregions" in feature["properties"]:
                            child_regions = feature["properties"]["subregions"]
                            child_features = []

                            for cleaned_region_name_vercye in child_regions:
                                region_data = self.gdf[self.gdf['cleaned_region_name_vercye'] == cleaned_region_name_vercye]
                                if not region_data.empty:
                                    row = region_data.iloc[0]
                                    lai_values, date_labels, interpolation_flags = self.aggregate_lai_data([cleaned_region_name_vercye])
                                    
                                    area = row.get('total_area_ha', 0)                                
                                    child_feature = {
                                        "type": "Feature",
                                        "properties": {
                                            "id": f"child_{parent_id}_{cleaned_region_name_vercye}",
                                            "name": str(cleaned_region_name_vercye),
                                            "parentId": parent_id,
                                            "heatValue": float(row['estimated_mean_yield_kg_ha']),
                                            "detail1": float(row['estimated_median_yield_kg_ha']),
                                            "detail2": area,
                                            "timeSeries": lai_values,
                                            "dateLabels": date_labels,
                                            "interpolationFlags": interpolation_flags,
                                        },
                                        "geometry": row.geometry.__geo_interface__
                                    }
                                    child_features.append(child_feature)

                            if child_features:
                                if parent_id not in data_structure["level_mappings"]:
                                    data_structure["level_mappings"][parent_id] = {
                                        "type": "FeatureCollection",
                                        "features": child_features
                                    }
        
        return data_structure
    
    def generate_html_template(self, data_structure: Dict[str, Any], title: str) -> str:
        """
        Generate HTML template with embedded data.
        
        Args:
            data_structure: Hierarchical data structure
            
        Returns:
            Complete HTML string
        """
        # Calculate value range for color scaling
        all_values = []
        for level_key, level_data in data_structure["level_data"].items():
            for feature in level_data["features"]:
                all_values.append(feature["properties"]["heatValue"])
        for mapping_data in data_structure["level_mappings"].values():
            for feature in mapping_data["features"]:
                all_values.append(feature["properties"]["heatValue"])

        # Compute min, max, and the 95th percentile in Python
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))
        p95_val = float(np.percentile(all_values, 95))

        title = title + ' Analysis'

        lai_column_options = ''.join(
            f'<option value="{col}">{col}</option>'
            for col in self.lai_column_names
        ) 
        
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

                <style>
                    body {{
                        margin: 0;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: #1a1a1a;
                        color: #fff;
                    }}
                    
                    .container {{
                        display: flex;
                        height: 100vh;
                    }}

                    .title {{
                        color: #63b3ed;
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
                        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
                        border-left: 2px solid #4a5568;
                        display: flex;
                        flex-direction: column;
                    }}
                    
                    .sidebar-header {{
                        padding: 20px;
                        border-bottom: 1px solid #4a5568;
                        background: rgba(0,0,0,0.3);
                    }}
                    
                    .sidebar-content {{
                        flex: 1;
                        padding: 20px;
                        overflow-y: auto;
                    }}
                    
                    .breadcrumb {{
                        display: flex;
                        align-items: center;
                        margin-bottom: 15px;
                        font-size: 14px;
                        color: #a0aec0;
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
                        background: rgba(255,255,255,0.1);
                        color: #fff;
                    }}
                    
                    .breadcrumb-separator {{
                        margin: 0 4px;
                        color: #4a5568;
                    }}

                    .selected-regions {{
                        margin-bottom: 30px;
                        margin-top: 30px;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 12px;
                        padding: 20px;
                        background: rgba(255, 255, 255, 0.05);
                    }}

                    .selected-regions h3 {{
                        color: #63b3ed;
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
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 8px;
                        transition: all 0.2s ease;
                    }}

                    .selected-region-item:hover {{
                        background: rgba(255, 255, 255, 0.15);
                    }}

                    .selected-region-name {{
                        color: #e2e8f0;
                        font-size: 14px;
                    }}

                    .remove-region-btn {{
                        background: #f56565;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 4px 8px;
                        font-size: 12px;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    }}

                    .remove-region-btn:hover {{
                        background: #e53e3e;
                    }}

                    .comparison-controls {{
                        margin-bottom: 20px;
                    }}

                    .lai-selector {{
                        margin-bottom: 15px;
                    }}

                    .lai-selector label {{
                        display: block;
                        color: #a0aec0;
                        font-size: 14px;
                        margin-bottom: 8px;
                    }}

                    .lai-selector select {{
                        width: 100%;
                        padding: 8px 12px;
                        background: rgba(255, 255, 255, 0.1);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        border-radius: 8px;
                        color: white;
                        font-size: 14px;
                    }}

                    .lai-selector select option {{
                        background: #2d3748;
                        color: white;
                    }}

                    .compare-button {{
                        width: 100%;
                        padding: 12px;
                        background: linear-gradient(135deg, #63b3ed, #38b2ac);
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
                        background: #4a5568;
                        cursor: not-allowed;
                        transform: none;
                        box-shadow: none;
                    }}

                    .clear-selection-btn {{
                        width: 100%;
                        padding: 8px;
                        background: transparent;
                        color: #f56565;
                        border: 1px solid #f56565;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        transition: all 0.2s ease;
                    }}

                    .clear-selection-btn:hover {{
                        background: #f56565;
                        color: white;
                    }}
                    
                    .tooltip-content {{
                        background: rgba(0,0,0,0.9);
                        padding: 15px;
                        border-radius: 8px;
                        border: 1px solid #4a5568;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                        backdrop-filter: blur(10px);
                    }}
                    
                    .tooltip-title {{
                        font-size: 16px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #63b3ed;
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
                        color: #68d391;
                    }}
                    
                    .stat-label {{
                        font-size: 12px;
                        color: #a0aec0;
                        margin-top: 2px;
                    }}
                    
                    .chart-container {{
                        height: 150px;
                        margin-top: 10px;
                    }}
                    
                    .legend {{
                        background: rgba(0,0,0,0.8);
                        padding: 15px;
                        border-radius: 8px;
                        margin-top: 20px;
                    }}
                    
                    .legend-title {{
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #63b3ed;
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
                        background: rgba(0,0,0,0.8);
                        padding: 15px;
                        border-radius: 8px;
                        backdrop-filter: blur(10px);
                        min-width: 200px;
                    }}
                    
                    .level-indicator {{
                        color: #63b3ed;
                        font-weight: bold;
                        font-size: 14px;
                        margin-bottom: 8px;
                    }}
                    
                    .back-button {{
                        background: #4299e1;
                        color: white;
                        border: none;
                        padding: 8px 15px;
                        border-radius: 6px;
                        cursor: pointer;
                        transition: all 0.2s;
                        width: 100%;
                    }}
                    
                    .back-button:hover {{
                        background: #3182ce;
                        transform: translateY(-1px);
                    }}
                    
                    .stats-summary {{
                        margin-top: 10px;
                        font-size: 12px;
                        color: #a0aec0;
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
                        background: rgba(0, 0, 0, 0.8);
                        backdrop-filter: blur(5px);
                        z-index: 10000;
                        transition: transform 0.3s ease;
                    }}

                    .detail-overlay.show {{
                        display: block;
                    }}

                    .detail-overlay-close {{
                        position: absolute;
                        top: 20px;
                        right: 20px;
                        background: #e53e3e;
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

                    .detail-overlay-header {{
                        font-size: 28px;
                        font-weight: 700;
                        margin-bottom: 10px;
                        color: #63b3ed;
                    }}

                    .detail-overlay-stats {{
                        display: flex;
                        gap: 20px;
                        margin-bottom: 20px;
                    }}

                    .detail-overlay-stat-item {{
                        flex: 1;
                        background: rgba(255,255,255,0.05);
                        padding: 15px;
                        border-radius: 6px;
                        text-align: center;
                    }}

                    .detail-overlay-stat-label {{
                        font-size: 14px;
                        color: #a0aec0;
                        margin-top: 4px;
                    }}

                    .detail-overlay-stat-value {{
                        font-size: 20px;
                        font-weight: bold;
                        color: #68d391;
                    }}

                    .detail-overlay-chart {{
                        width: 100%;
                        height: 250px;
                    }}

                    .comparison-overlay {{
                        display: none;
                        position: fixed;
                        top: 0;
                        left: 0;
                        bottom: 0;
                        width: calc(100% - 500px); /* Sidebar width is 500px */
                        background: rgba(0, 0, 0, 0.8);
                        backdrop-filter: blur(5px);
                        z-index: 10000;
                    }}

                    .comparison-overlay-content {{
                        height: 100%;
                        padding: 40px;
                        overflow-y: auto;
                        background: rgba(26, 32, 44, 0.95);
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
                        color: #63b3ed;
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

                </style>
            </head>
            <body>
                <div class="container">
                    <div class="map-container">
                        <div id="map"></div>
                        <div class="control-panel">
                            <div class="level-indicator" id="levelIndicator">Loading...</div>
                            <div class="stats-summary" id="statsSummary"></div>
                            <button class="back-button" id="backButton" style="display: none;">← Back</button>
                        </div>
                    </div>
                    
                    <div class="sidebar">
                        <div class="sidebar-header">
                            <h2 class="title">{title}</h2>
                            <p>
                                <ul>
                                    <li>Click on a region to view nested regions.</li>
                                    <li>Double click on a region to see the regions stat details.</li>
                                    <li>Use CTRL+Click to select multiple regions for comparing their LAI curves.</li>
                                </ul>
                            </p>
                            <div class="breadcrumb" id="breadcrumb"></div>
                        </div>
                        
                        <div class="sidebar-content">
                            <div id="hoverInfo" style="display: none;">
                                <div class="tooltip-content">
                                    <div class="tooltip-title" id="hoverTitle"></div>
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
                                    <div class="legend-note" style="font-size: 11px; margin-top: 6px; color: #a0aec0;">
                                        <p>★ denotes aggregated LAI series computed as the daily mean of all LAI series from subregions.</p>
                                    </div>
                                    <div class="chart-container">
                                        <canvas id="hoverChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="legend">
                                <div class="legend-title">Yield Legend (kg/ha)</div>
                                <div id="legendItems"></div>
                            </div>

                            <div class="selected-regions">
                                <h3>Comparison Selection</h3>
                                <h4>Selected Regions (<span id="selectedCount">0</span>)</h3>
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
                        <div class="detail-overlay-header" id="overlayTitle">Region Details</div>
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
                                <div class="detail-overlay-stat-label">Area (km²)</div>
                            </div>
                        </div>
                        <div class="legend-note" style="font-size: 11px; margin-top: 6px; color: #a0aec0;">
                            <p>★ denotes aggregated LAI series computed as the daily mean of all LAI series from subregions.</p>
                        </div>
                        <div class="detail-overlay-chart">
                            <canvas id="overlayChart"></canvas>
                        </div>
                    </div>
                </div>

                <div id="comparisonOverlay" class="comparison-overlay">
                    <div class="comparison-overlay-content">
                        <button id="comparisonCloseBtn" class="detail-overlay-close">✕ Close</button>
                        <div class="comparison-overlay-header" id="comparisonTitle">Region Comparison</div>
                        <div class="comparison-overlay-chart">
                            <canvas id="comparisonChart"></canvas>
                        </div>
                        <div id="comparisonStats" style="margin-top: 30px;"></div>
                    </div>
                </div>

                <script>
                    // Embedded data
                    const mapData = {json.dumps(data_structure, separators=(",", ":"), ensure_ascii=False)};
                    const valueRange = {{
                        min: {min_val},
                        max: {max_val},
                        p95: {p95_val}
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

                    // Selection state for comparison
                    let selectedRegions = new Map(); // regionId -> name, properties, layer
                    let comparisonChart = null;
                    let availableLaiColumns = [];

                    // Color ramp where Chroma.js interpolates smoothly between these stops
                    const ramp = chroma
                        .scale('viridis')
                        .domain([0, 1])
                        .mode('lrgb');

                    // Color scale function based on actual data range
                    function getColor(value) {{
                        const min = valueRange.min;
                        const max = valueRange.p95;


                        //  Clamp value between min and max
                        const t = Math.max(0, Math.min(1, (value - min) / (max - min)));

                        return ramp(t).hex();
                    }}

                    // Style each feature using getColor(...)
                    function style(feature) {{
                        return {{
                            fillColor: getColor(feature.properties.heatValue),
                            weight: 2,
                            opacity: 1,
                            color: 'white',
                            dashArray: '3',
                            fillOpacity: 0.85
                        }};
                    }}

                    // Event handlers
                    function highlightFeature(e) {{
                        const layer = e.target;
                        const props = layer.feature.properties;
                        
                        layer.setStyle({{
                            weight: 4,
                            color: '#666',
                            dashArray: '',
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
                        document.getElementById('overlayValue1').textContent = props.heatValue.toFixed(1);
                        document.getElementById('overlayValue2').textContent = (props.detail1 || 0).toFixed(1);
                        document.getElementById('overlayValue3').textContent = (props.detail2 || 0).toFixed(1);

                        updateOverlayChart(props.timeSeries, props.dateLabels, props.interpolationFlags, props.isAggregated);

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
                                fill: true,
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
                                        position: 'top',
                                        labels: {{
                                            color: '#a0aec0',
                                            font: {{ size: 11 }},
                                            filter: function(legendItem, chartData) {{
                                                const txt = legendItem.text || '';
                                                return !txt.includes('Day with RS');
                                            }}
                                        }},
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
                                            color: '#a0aec0',
                                            font: {{ size: 10 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#a0aec0',
                                            font: {{ size: 10 }}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI Value',
                                            color: '#a0aec0'
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

                                const mean = props.heatValue?.toFixed(1) || '-';
                                const median = props.detail1?.toFixed(1) || '-';

                                const regionStats = document.createElement('div');
                                regionStats.style.marginTop = '10px';
                                regionStats.style.display = 'flex';
                                regionStats.style.alignItems = 'center';
                                regionStats.style.gap = '10px';

                                regionStats.innerHTML = `
                                    <div style="width: 12px; height: 12px; background-color: ${{color}}; border-radius: 2px;"></div>
                                    <strong style="color: ${{color}};">${{name}}</strong>
                                    <span style="color: #a0aec0;">Mean Yield: <strong>${{mean}}</strong> kg/ha</span>
                                    <span style="color: #a0aec0;">Median Yield: <strong>${{median}}</strong> kg/ha</span>
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
                                        position: 'top',
                                        labels: {{
                                            color: '#a0aec0',
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
                                            color: '#a0aec0',
                                            font: {{ size: 11 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#a0aec0',
                                            font: {{ size: 11 }}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI Value',
                                            color: '#a0aec0'
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
                        if (mapData.level_mappings[parentId]) {{
                            // We're drilling into a subset of the current level
                            currentLevel++;
                            currentParent = parentId;
                            breadcrumbPath.push(parentName);
                            
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
                            breadcrumbPath.push(parentName);
                            
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
                            breadcrumbItem.textContent = item;
                            
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
                        console.log(mapData.levels);
                        console.log(`Current Level: ${{currentLevel}}, Name: ${{levelInfo.name}}`);
                        indicator.textContent = `Level ${{currentLevel + 1}}: ${{levelInfo.name}}`;
                    }}

                    function updateStats(levelData) {{
                        const values = levelData.features.map(f => f.properties.heatValue);
                        const avgYield = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
                        const minYield = Math.min(...values).toFixed(1);
                        const maxYield = Math.max(...values).toFixed(1);
                        
                        const statsDiv = document.getElementById('statsSummary');
                        statsDiv.innerHTML = `
                            <div>Regions: ${{values.length}}</div>
                            <div>Avg Yield: ${{avgYield}} kg/ha</div>
                            <div>Range: ${{minYield}} - ${{maxYield}} kg/ha</div>
                        `;
                    }}

                    function updateLegend() {{
                        const legendItems = document.getElementById('legendItems');
                        legendItems.innerHTML = '';

                        const min = valueRange.min.toFixed(0);
                        const max = valueRange.p95.toFixed(0);

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
                        labelContainer.style.color = '#a0aec0';

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
                        document.getElementById('hoverTitle').textContent = props.name;
                        document.getElementById('hoverValue1').textContent = props.heatValue.toFixed(1);
                        document.getElementById('hoverValue2').textContent = (props.detail1 || 0).toFixed(1);
                        
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
                                fill: true,
                                tension: 0.4,
                                pointRadius: 0,
                                pointHoverRadius: 0
                            }});

                            datasets.push({{
                                label: seriesName + ' (Day with RS Data)',
                                data: rawData.map((val, i) => interpolationFlags?.[i] === 0 ? val : null),
                                showLine: false,
                                pointRadius: 3,
                                pointHoverRadius: 3,
                                pointBackgroundColor: color,
                                pointBorderColor: color,
                                borderWidth: 0
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
                                        position: 'top',
                                        labels: {{
                                            color: '#a0aec0',
                                            font: {{ size: 10 }},
                                            filter: function(legendItem, chartData) {{
                                                const txt = legendItem.text || '';
                                                return !txt.includes('Day with RS');
                                            }}
                                        }},
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
                                            color: '#a0aec0',
                                            font: {{ size: 10 }}
                                        }}
                                    }},
                                    y: {{
                                        grid: {{
                                            color: 'rgba(255,255,255,0.1)',
                                            drawBorder: false
                                        }},
                                        ticks: {{
                                            color: '#a0aec0',
                                            font: {{ size: 10 }}
                                        }},
                                        title: {{
                                            display: true,
                                            text: 'LAI Value',
                                            color: '#a0aec0'
                                        }}
                                    }}
                                }}
                            }}
                        }});
                    }}

                    // Event listeners
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

                        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                            attribution: '© OpenStreetMap contributors'
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
        data_structure = self.create_hierarchical_data()
        
        print("Generating HTML template...")
        html_content = self.generate_html_template(data_structure, title)
        
        print(f"Writing to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive map generated successfully: {self.output_path}")
        return self.output_path
    
    def load_aggregated_estimates(self):
        agg_estimates = {}
        for idx, level in enumerate(self.aggregation_levels):
            
            agg_estimates_df = pd.read_csv(level['agg_estimates_file'])

            # create dict with region names as keys and aggregated data as values
            agg_data = {}
            for _, row in agg_estimates_df.iterrows():
                region_name = row['region']
                agg_data[region_name] = {
                    'estimated_mean_yield_kg_ha': row['mean_yield_kg_ha'],
                    'estimated_median_yield_kg_ha': row['median_yield_kg_ha'],
                    'total_area_ha': row['total_area_ha']
                }

            agg_estimates[idx] = agg_data
        return agg_estimates
    
    def load_lai_data(self, lai_basedir_path: str) -> pd.DataFrame:
        """
        Load LAI data from CSV files in the specified directory.
        
        Args:
            lai_basedir_path: Directory containing LAI CSV files
            
        Returns:
            Combined DataFrame with all LAI data
        """
        lai_files = list(Path(lai_basedir_path).glob('*_LAI_STATS.csv'))
        
        if not lai_files:
            raise FileNotFoundError(f"No LAI data files found in {lai_basedir_path}")
        
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

def parse_agg_level(ctx, param, value):
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
@click.option('--lai-basedir-path', required=True, type=click.Path(exists=True), help='Path to the base directory containing LAI data.')
@click.option('--adjusted/--no-adjusted', default=True, help='Include adjusted LAI columns.')
@click.option('--smoothed/--no-smoothed', default=True, help='Include smoothed LAI columns.')
@click.option('--agg-level', multiple=True, callback=parse_agg_level,
              help='Aggregation level in format level:column:path. Can be used multiple times.')
@click.option('--lai-column', multiple=True, default=DEFAULT_LAI_COLUMNS, callback=parse_lai_column,
              help='LAI column in format internal_name:display_name. Can be used multiple times.')
@click.option('--title', type=str, default='Yield Study', help='Title for the interactive map.')
@click.option('--output-path', type=click.Path(), default='interactive_map.html', help='Output path for the generated HTML file.')
def main(shapefile_path, lai_basedir_path, adjusted, smoothed, agg_level, lai_column, title, output_path):

    lai_columns, lai_column_names = lai_column

    if adjusted:
        lai_columns = [col + ' Adjusted' for col in lai_columns]
        lai_column_names = [name + ' Adjusted' for name in lai_column_names]

    if smoothed:
        lai_columns += [col + ' Unsmoothed' for col in lai_columns]
        lai_column_names += [name + ' Smoothed' for name in lai_column_names] + [name + ' Unsmoothed' for name in lai_column_names]

    generator = InteractiveMapGenerator(
        shapefile_path=shapefile_path,
        lai_basedir_path=lai_basedir_path,
        output_path=output_path,
        lai_columns=lai_columns,
        lai_column_names=lai_column_names,
        agg_levels=agg_level
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