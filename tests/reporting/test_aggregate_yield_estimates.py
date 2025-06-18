import os
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyarrow as pa
import pyarrow.parquet as pq

from vercye_ops.matching_sim_real.aggregate_yield_estimates import aggregate_yields  # Replace with actual import path


def create_region_dir(base_dir, region_name, yield_value=10000, apsim_mean_yield=3000, lai_value=3.5):
    region_dir = os.path.join(base_dir, region_name)
    os.makedirs(region_dir, exist_ok=True)

    # Yield estimate CSV
    pd.DataFrame({
        'total_yield_production_kg': [yield_value],
        'total_area_ha': [100.0]
    }).to_csv(os.path.join(region_dir, f"{region_name}_converted_map_yield_estimate.csv"), index=False)

    # Conversion factor CSV
    pd.DataFrame({
        'max_rs_lai': [lai_value],
        'max_rs_lai_date': ['2022-06-15'],
        'apsim_max_matched_lai': [lai_value - 0.5],
        'apsim_max_matched_lai_date': ['2022-06-10'],
        'apsim_max_all_lai': [lai_value],
        'apsim_max_all_lai_date': ['2022-06-12'],
        'apsim_mean_yield_estimate_kg_ha': [apsim_mean_yield],
        'apsim_matched_std_yield_estimate_kg_ha': [100],
        'apsim_all_std_yield_estimate_kg_ha': [110],
        'apsim_matched_maxlai_std': [0.4],
        'apsim_all_maxlai_std': [0.6],
    }).to_csv(os.path.join(region_dir, f"{region_name}_conversion_factor.csv"), index=False)

    # LAI stats CSV
    pd.DataFrame({
        'interpolated': [0, 0, 1],
        'Cloud or Snow Percentage': [20.0, 50.0, 100.0]
    }).to_csv(os.path.join(region_dir, f"{region_name}_LAI_STATS.csv"), index=False)

    # GeoJSON
    gdf = gpd.GeoDataFrame({
        'admin_name': [f"{region_name}_admin"],
        'country': ['Testland'],
        'geometry': [Point(0, 0)]
    }, crs="EPSG:4326")
    gdf.to_file(os.path.join(region_dir, f"{region_name}.geojson"), driver='GeoJSON')


def create_chirps_parquet(file_path, regions):
    table = pa.table({region: pa.array([1.0]) for region in regions})
    pq.write_table(table, file_path)


def test_aggregate_yields_basic(tmp_path):
    region_name = "regionA"
    base_dir = tmp_path / "input"
    base_dir.mkdir()
    create_region_dir(base_dir, region_name)

    chirps_file = tmp_path / "chirps.parquet"
    create_chirps_parquet(chirps_file, [region_name])

    result = aggregate_yields(str(base_dir), columns_to_keep="admin_name,country", chirps_path=str(chirps_file))

    assert not result.empty
    assert 'region' in result.columns
    assert 'admin_name' in result.columns
    assert result.loc[0, 'region'] == region_name
    assert result.loc[0, 'precipitation_src'] == 'CHIRPS'
    assert result.loc[0, 'admin_name'] == f"{region_name}_admin"


def test_missing_geojson_column_raises(tmp_path):
    region_name = "regionB"
    base_dir = tmp_path / "input"
    base_dir.mkdir()
    create_region_dir(base_dir, region_name)

    # Remove expected column from geojson
    geojson_path = base_dir / region_name / f"{region_name}.geojson"
    gdf = gpd.read_file(geojson_path)
    gdf = gdf.drop(columns=["admin_name"])
    gdf.to_file(geojson_path, driver='GeoJSON')

    with pytest.raises(Exception, match="Columns .* not found in the geojson"):
        aggregate_yields(str(base_dir), columns_to_keep="admin_name,country")


def test_region_skipped_if_yield_file_missing(tmp_path):
    region_name = "regionC"
    base_dir = tmp_path / "input"
    base_dir.mkdir()
    create_region_dir(base_dir, region_name)

    # Remove yield estimate file
    os.remove(base_dir / region_name / f"{region_name}_converted_map_yield_estimate.csv")

    df = aggregate_yields(str(base_dir), columns_to_keep=None)
    assert df.empty


def test_aggregate_yields_multiple_regions(tmp_path):
    num_regions = 10
    base_dir = tmp_path / "input"
    base_dir.mkdir()

    region_data = []
    for i in range(num_regions):
        region_name = f"region{i}"
        yield_val = i * 1000
        apsim_val = i * 200
        lai_val = i * 0.1
        create_region_dir(base_dir, region_name, yield_val, apsim_val, lai_val)
        region_data.append((region_name, yield_val, apsim_val))

    chirps_file = tmp_path / "chirps.parquet"
    create_chirps_parquet(chirps_file, [f"region{i}" for i in range(5)])

    result = aggregate_yields(str(base_dir), columns_to_keep="admin_name,country", chirps_path=str(chirps_file))

    assert result['region'].nunique() == num_regions

    for region_name, yield_val, apsim_val in region_data:
        region_row = result[result['region'] == region_name]
        assert not region_row.empty
        assert region_row['total_yield_production_kg'].iloc[0] == yield_val
        assert region_row['apsim_mean_yield_estimate_kg_ha'].iloc[0] == apsim_val
        assert region_row['admin_name'].iloc[0] == f"{region_name}_admin"
        expected_src = 'CHIRPS' if region_name in [f"region{i}" for i in range(5)] else 'Met Source'
        assert region_row['precipitation_src'].iloc[0] == expected_src
