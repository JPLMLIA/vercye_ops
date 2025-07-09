import os
import re
import yaml
import glob
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def validate_run_config(config_file):
    """
    Validate APSIM run configuration file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Raises:
        FileNotFoundError: If required files/directories don't exist
        ValueError: If configuration values are invalid
        RuntimeError: If dependencies are not available
    """
    
    # Load configuration
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_file}: {e}")
    
    print("Validating APSIM configuration...")
    
    # 1. Validate simulation study head directory
    _validate_sim_study_head_dir(config)
    
    # 2. Validate study metadata
    _validate_study_metadata(config)
    
    # 3. Validate years and timepoints
    _validate_years_timepoints(config)
    
    # 4. Validate regions and directory structure
    _validate_regions_structure(config)
    
    # 5. Validate APSIM parameters
    _validate_apsim_params(config)
    
    # 6. Validate LAI parameters
    _validate_lai_params(config)
    
    # 7. Validate crop masks
    _validate_crop_masks(config)
    
    # 8. Validate evaluation parameters
    _validate_eval_params(config)
    
    # 9. Validate script paths
    _validate_script_paths(config)
    
    # 10. Check for template placeholders
    _check_template_placeholders(config)
    
    # 11. Validate external dependencies
    _validate_external_dependencies(config)
    

def _validate_sim_study_head_dir(config):
    """Validate simulation study head directory."""
    head_dir = config.get('sim_study_head_dir', '')
    
    if not head_dir:
        raise ValueError("sim_study_head_dir is required")
    
    if not os.path.exists(head_dir):
        raise FileNotFoundError(f"Simulation head dir {head_dir} does not exist")
    
    if not os.path.isdir(head_dir):
        raise FileNotFoundError(f"Simulation head dir {head_dir} is not a directory")
    
    if not os.listdir(head_dir):
        raise FileNotFoundError(f"Simulation head dir {head_dir} is empty, but should contain entries for each year/timepoint")
    
    print(f"✓ Simulation head directory validated: {head_dir}")


def _validate_study_metadata(config):
    """Validate study metadata fields."""
    study_id = config.get('study_id', '')
    
    if not study_id:
        raise ValueError("study_id is required")
    
    if len(study_id) > 25:
        raise ValueError("study_id should not exceed 25 characters")
    
    # Check for special characters that might cause issues
    if not re.match(r'^[a-zA-Z0-9_-]+$', study_id):
        raise ValueError("study_id should only contain alphanumeric characters, underscores, and hyphens")
    
    required_fields = ['title', 'description']
    for field in required_fields:
        if not config.get(field):
            raise ValueError(f"{field} is required")
    
    print(f"✓ Study metadata validated: {study_id}")


def _validate_years_timepoints(config):
    """Validate years and timepoints."""
    years = config.get('years', [])
    timepoints = config.get('timepoints', [])
    
    if not years:
        raise ValueError("At least one year must be specified")
    
    if not timepoints:
        raise ValueError("At least one timepoint must be specified")
    
    # Validate years are integers
    for year in years:
        if not isinstance(year, int) or year < 1900 or year > 2030:
            raise ValueError(f"Invalid year: {year}. Must be an integer between 1900 and 2030")
    
    # Validate timepoint format
    for tp in timepoints:
        if not re.match(r'^T-\d+$', tp):
            raise ValueError(f"Invalid timepoint format: {tp}. Must be in format 'T-N' where N is a number")
    
    print(f"✓ Years and timepoints validated: {years}, {timepoints}")


def _validate_regions_structure(config):
    """Validate regions and their directory structure."""
    regions = config.get('regions', [])
    years = config.get('years', [])
    timepoints = config.get('timepoints', [])
    head_dir = config.get('sim_study_head_dir', '')
    
    if not regions:
        raise ValueError("At least one region must be specified")
    
    # Check that each year/timepoint/region combination has required files
    for year in years:
        for timepoint in timepoints:
            for region in regions:
                region_dir = os.path.join(head_dir, str(year), timepoint, region)
                
                if not os.path.exists(region_dir):
                    raise FileNotFoundError(f"Region directory not found: {region_dir}")
                
                # Check for required files
                geojson_files = glob.glob(os.path.join(region_dir, "*.geojson"))
                apsim_files = glob.glob(os.path.join(region_dir, "*.apsimx"))
                
                if not geojson_files:
                    raise FileNotFoundError(f"No GeoJSON file found in {region_dir}")
                
                if not apsim_files:
                    raise FileNotFoundError(f"No APSIM file found in {region_dir}")
    
    print(f"✓ Region structure validated for {len(regions)} regions")


def _validate_apsim_params(config):
    """Validate APSIM parameters."""
    apsim_params = config.get('apsim_params', {})
    
    # Validate met_source options
    met_source = apsim_params.get('met_source', '')
    valid_met_sources = ['NASA_POWER', 'ERA5']
    if met_source not in valid_met_sources:
        raise ValueError(f"Invalid met_source: {met_source}. Must be one of {valid_met_sources}")
    
    # Validate aggregation methods
    met_agg = apsim_params.get('met_agg_method', '')
    valid_agg_methods = ['mean', 'centroid']
    if met_agg not in valid_agg_methods:
        raise ValueError(f"Invalid met_agg_method: {met_agg}. Must be one of {valid_agg_methods}")
    
    # Validate precipitation source
    precip_source = apsim_params.get('precipitation_source', '')
    valid_precip_sources = ['NASA_POWER', 'CHIRPS', 'ERA5']
    if precip_source not in valid_precip_sources:
        raise ValueError(f"Invalid precipitation_source: {precip_source}. Must be one of {valid_precip_sources}")
    
    # Validate cache directories
    cache_dirs = ['nasapower_cache_dir', 'era5_cache_dir']
    for cache_dir_key in cache_dirs:
        cache_dir = apsim_params.get(cache_dir_key, '')
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✓ Created cache directory: {cache_dir}")
    
    # Validate CHIRPS directory if needed
    if precip_source == 'CHIRPS':
        chirps_dir = apsim_params.get('chirps_dir', '')
        if not chirps_dir:
            raise ValueError("chirps_dir is required when precipitation_source is 'CHIRPS'")
        if not os.path.exists(chirps_dir):
            raise FileNotFoundError(f"CHIRPS directory not found: {chirps_dir}")
    
    # Validate Earth Engine project if needed
    if met_source == 'ERA5':
        ee_project = apsim_params.get('ee_project', '')
        if not ee_project:
            raise ValueError("ee_project is required when met_source is 'ERA5'")
    
    # Validate time bounds
    _validate_time_bounds(config)
    
    # Validate APSIM execution settings
    _validate_apsim_execution(config)
    
    print("✓ APSIM parameters validated")


def _validate_time_bounds(config):
    """Validate APSIM time bounds."""
    years = config.get('years', [])
    timepoints = config.get('timepoints', [])
    time_bounds = config.get('apsim_params', {}).get('time_bounds', {})
    
    for year in years:
        if str(year) not in time_bounds:
            raise ValueError(f"Time bounds missing for year {year}")
        
        year_bounds = time_bounds[str(year)]
        
        for timepoint in timepoints:
            if timepoint not in year_bounds:
                raise ValueError(f"Time bounds missing for year {year}, timepoint {timepoint}")
            
            bounds = year_bounds[timepoint]
            required_dates = ['sim_start_date', 'sim_end_date', 'met_start_date', 'met_end_date']
            
            for date_key in required_dates:
                if date_key not in bounds:
                    raise ValueError(f"Missing {date_key} for year {year}, timepoint {timepoint}")
                
                # Validate date format
                try:
                    datetime.strptime(bounds[date_key], '%Y-%m-%d')
                except ValueError:
                    raise ValueError(f"Invalid date format for {date_key} in year {year}, timepoint {timepoint}")
            
            # Check date logic
            sim_start = datetime.strptime(bounds['sim_start_date'], '%Y-%m-%d')
            sim_end = datetime.strptime(bounds['sim_end_date'], '%Y-%m-%d')
            met_start = datetime.strptime(bounds['met_start_date'], '%Y-%m-%d')
            met_end = datetime.strptime(bounds['met_end_date'], '%Y-%m-%d')
            
            if sim_start >= sim_end:
                raise ValueError(f"sim_start_date must be before sim_end_date for year {year}, timepoint {timepoint}")
            
            if met_start >= met_end:
                raise ValueError(f"met_start_date must be before met_end_date for year {year}, timepoint {timepoint}")


def _validate_apsim_execution(config):
    """Validate APSIM execution settings."""
    apsim_exec = config.get('apsim_execution', {})
    use_docker = apsim_exec.get('use_docker', False)
    
    if use_docker:
        # Validate Docker settings
        docker_config = apsim_exec.get('docker', {})
        if not docker_config.get('image'):
            raise ValueError("Docker image is required when use_docker is True")
        
        # Check if Docker is available
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Docker is not available but use_docker is True")
        
        # Check if Docker image exists
        image = docker_config.get('image')
        try:
            subprocess.run(['docker', 'image', 'inspect', image], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print(f"Docker image {image} not found locally. You may need to pull it first.")
    
    else:
        # Validate local execution
        local_config = apsim_exec.get('local', {})
        executable = local_config.get('executable_fpath', '')
        
        if not executable:
            raise ValueError("executable_fpath is required when use_docker is False")
        
        if not os.path.exists(executable):
            raise FileNotFoundError(f"APSIM executable not found: {executable}")
        
        if not os.access(executable, os.X_OK):
            raise ValueError(f"APSIM executable is not executable: {executable}")


def _validate_lai_params(config):
    """Validate LAI parameters."""
    lai_params = config.get('lai_params', {})
    
    # Validate LAI directory
    lai_dir = lai_params.get('lai_dir', '')
    if not lai_dir:
        raise ValueError("lai_dir is required")
    
    if not os.path.exists(lai_dir):
        raise FileNotFoundError(f"LAI directory not found: {lai_dir}")
    
    # Validate LAI parameters
    lai_region = lai_params.get('lai_region', '')
    if not lai_region:
        raise ValueError("lai_region is required")
    
    lai_resolution = lai_params.get('lai_resolution')
    if not isinstance(lai_resolution, int) or lai_resolution <= 0:
        raise ValueError("lai_resolution must be a positive integer")
    
    # Validate crop name
    crop_name = lai_params.get('crop_name', '')
    valid_crops = ['wheat', 'maize']
    if crop_name not in valid_crops:
        raise ValueError(f"Invalid crop_name: {crop_name}. Must be one of {valid_crops}")
    
    # Validate LAI time bounds
    _validate_lai_time_bounds(config)
    
    # Check for LAI files
    _validate_lai_files(config)
    
    print("✓ LAI parameters validated")


def _validate_lai_time_bounds(config):
    """Validate LAI time bounds."""
    years = config.get('years', [])
    timepoints = config.get('timepoints', [])
    lai_time_bounds = config.get('lai_params', {}).get('time_bounds', {})
    
    for year in years:
        if year not in lai_time_bounds:
            raise ValueError(f"LAI time bounds missing for year {year}")
        
        year_bounds = lai_time_bounds[year]
        
        for timepoint in timepoints:
            if timepoint not in year_bounds:
                raise ValueError(f"LAI time bounds missing for year {year}, timepoint {timepoint}")
            
            bounds = year_bounds[timepoint]
            
            if len(bounds) != 2:
                raise ValueError(f"LAI time bounds must have exactly 2 dates for year {year}, timepoint {timepoint}. First one is the start date, the second one the end date.")
            
            # Validate date format and logic
            try:
                start_date = datetime.strptime(bounds[0], '%Y-%m-%d')
                end_date = datetime.strptime(bounds[1], '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid LAI date format for year {year}, timepoint {timepoint}")
            
            if start_date >= end_date:
                raise ValueError(f"LAI start date must be before end date for year {year}, timepoint {timepoint}")


def _validate_lai_files(config):
    """Validate that LAI files exist."""
    lai_params = config.get('lai_params', {})
    lai_dir = lai_params.get('lai_dir', '')
    lai_region = lai_params.get('lai_region', '')
    lai_resolution = lai_params.get('lai_resolution', 0)
    file_ext = lai_params.get('file_ext', '')
    
    # Look for LAI files matching the pattern
    pattern = os.path.join(lai_dir, f"*{lai_region}*_{lai_resolution}m*.{file_ext}")
    lai_files = glob.glob(pattern)

    if len(lai_files) == 0:
        raise FileNotFoundError(f"No LAI files found matching pattern {pattern}."
                f"This typically indicates that either you have provided the wrong "
                f"lai directory, the wrong lai region name (should be your regions prefix) "
                f"the wrong resolution or the wrong extension."
              )
    
    if len(lai_files) < 5:
        print(f"⚠️  Only {len(lai_files)} LAI files found matching pattern {pattern}. "
              f"Consider verifying LAI data availability."
              f"This often indicates that either your date range is too short or "
              f"your cloud coverage threshold is too high."
              )


def _validate_crop_masks(config):
    """Validate crop mask files."""
    years = config.get('years', [])
    crop_masks = config.get('lai_params', {}).get('crop_mask', {})
    
    for year in years:
        if year not in crop_masks:
            raise ValueError(f"Crop mask missing for year {year}")
        
        mask_path = crop_masks[year]
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Crop mask file not found: {mask_path}")
    
    print("✓ Crop masks validated")


def _validate_eval_params(config):
    """Validate evaluation parameters."""
    eval_params = config.get('eval_params', {})
    aggregation_levels = eval_params.get('aggregation_levels', {})
    
    # Check for colons in aggregation level names (not allowed)
    for level_name, column_name in aggregation_levels.items():
        if ':' in level_name:
            raise ValueError(f"Aggregation level name '{level_name}' contains colon, which is not allowed")
        if ':' in column_name:
            raise ValueError(f"Aggregation level column '{column_name}' contains colon, which is not allowed")
    
    # Validate aggregation levels exist in GeoJSON files
    _validate_aggregation_levels_in_geojson(config)
    
    print("✓ Evaluation parameters validated")


def _validate_aggregation_levels_in_geojson(config):
    """Validate that aggregation levels exist in GeoJSON files."""
    regions = config.get('regions', [])
    years = config.get('years', [])
    timepoints = config.get('timepoints', [])
    head_dir = config.get('sim_study_head_dir', '')
    aggregation_levels = config.get('eval_params', {}).get('aggregation_levels', {})
    
    if not aggregation_levels:
        return
    
    # Check the first available GeoJSON file
    for year in years:
        for timepoint in timepoints:
            for region in regions:
                region_dir = os.path.join(head_dir, str(year), timepoint, region)
                geojson_files = glob.glob(os.path.join(region_dir, "*.geojson"))
                
                if geojson_files:
                    try:
                        with open(geojson_files[0], 'r') as f:
                            geojson_data = json.load(f)
                        
                        if 'features' in geojson_data and geojson_data['features']:
                            properties = geojson_data['features'][0].get('properties', {})
                            
                            for level_name, column_name in aggregation_levels.items():
                                if column_name not in properties:
                                    raise ValueError(f"Aggregation level column '{column_name}' not found in GeoJSON properties")
                        
                        return  # Only need to check one file
                    
                    except (json.JSONDecodeError, KeyError) as e:
                        raise ValueError(f"Invalid GeoJSON file {geojson_files[0]}: {e}")


def _validate_script_paths(config):
    """Validate that all script paths exist."""
    scripts = config.get('scripts', {})
    
    for script_name, script_path in scripts.items():
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_name} at {script_path}")
        
        if os.path.getsize(script_path) == 0:
            raise ValueError(f"Script is empty: {script_name} at {script_path}")
    
    print(f"✓ Script paths validated ({len(scripts)} scripts)")


def _check_template_placeholders(config):
    """Check for template placeholder values."""
    def _check_dict(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                _check_dict(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _check_dict(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            if 'XXXX' in obj:
                raise ValueError(f"Template placeholder 'XXXX' found in {path}: {obj}")
    
    _check_dict(config)
    print("✓ No template placeholders found")


def _validate_external_dependencies(config):
    """Validate external dependencies."""
    apsim_params = config.get('apsim_params', {})
    
    # Check Earth Engine authentication if using ERA5
    if apsim_params.get('met_source') == 'ERA5':
        try:
            import ee
            try:
                ee.Initialize()
                print("✓ Earth Engine authentication verified")
            except Exception:
                raise RuntimeError("Earth Engine not authenticated. Please run 'earthengine authenticate'")
        except ImportError:
            raise RuntimeError("Earth Engine Python API not installed. Please install with 'pip install earthengine-api'")
    
    print("✓ External dependencies validated")


def main():
    """Main function to run validation."""
    if len(sys.argv) != 2:
        print("Usage: python validate_config.py <config_file.yaml>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        validate_run_config(config_file)
        print("\n🎉 Configuration validation passed! Your setup is ready to run.")
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()