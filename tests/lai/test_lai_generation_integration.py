import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from click.testing import CliRunner
from rasterio.transform import from_origin

warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def load_module_from_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_s2_20m_like_tif(out_path: Path, H=4, W=5, num_bands=11):
    """
    Create a small float32 GeoTIFF with 11 bands, nodata=0.
    Last band is non-zero so neither CLI skips it.
    """
    transform = from_origin(500000.0, 4600000.0, 20.0, 20.0)  # arbitrary 20 m grid
    profile = {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": num_bands,
        "dtype": "float32",
        "crs": "EPSG:32631",
        "transform": transform,
        "nodata": 0.0,
        "compress": "lzw",
    }

    # Deterministic data: band k filled with (k+1)
    cube = np.zeros((num_bands, H, W), dtype=np.float32)
    for k in range(num_bands):
        cube[k, :, :] = k + 1

    # Put a few zeros to exercise nodata/NaN masking but keep the last band non-zero overall
    cube[:, 0, 0] = 0.0  # one pixel nodata across bands

    with rio.open(out_path, "w", **profile) as dst:
        dst.write(cube)


def write_simple_vrt(vrt_path: Path, tif_name_in_same_dir: str, width: int, height: int, num_bands: int):
    """
    Minimal VRT that references a TIFF placed in the same directory.
    Uses relativeToVRT=1 so the filename is relative to the VRT location.
    """
    bands_xml = []
    for b in range(1, num_bands + 1):
        bands_xml.append(
            f"""
  <VRTRasterBand dataType="Float32" band="{b}" subClass="VRTSourcedRasterBand">
    <NoDataValue>0</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{tif_name_in_same_dir}</SourceFilename>
      <SourceBand>{b}</SourceBand>
    </SimpleSource>
  </VRTRasterBand>"""
        )
    xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
{''.join(bands_xml)}
</VRTDataset>
"""
    vrt_path.write_text(xml)


@pytest.mark.parametrize("date_str", ["2025-06-15"])
def test_cli_outputs_identical_real_model(tmp_path, date_str):
    """
    Creates a real 11-band TIFF, two VRTs (one per CLI),
    runs both CLIs with the real model code, compares outputs.
    """
    # Make imagery dirs + outputs
    tiled_img_dir = tmp_path / "s2_tiled"
    gee_img_dir = tmp_path / "s2_gee"
    out_tiled = tmp_path / "out_tiled"
    out_gee = tmp_path / "out_gee"
    for p in (tiled_img_dir, gee_img_dir, out_tiled, out_gee):
        p.mkdir(parents=True)

    # Write one TIFF per dir with identical data
    tif_name = "S2_20m_stack.tif"
    tif_tiled = tiled_img_dir / tif_name
    tif_gee = gee_img_dir / tif_name
    write_s2_20m_like_tif(tif_tiled)
    # duplicate exact bytes to keep bitwise identity
    tif_gee.write_bytes(tif_tiled.read_bytes())

    # Write VRTs that match each CLI’s filename pattern
    tiled_vrt = tiled_img_dir / f"T31TCJ_20m_{date_str}.vrt"
    write_simple_vrt(tiled_vrt, tif_name_in_same_dir=tif_name, width=5, height=4, num_bands=11)

    region = "PARIS"
    gee_vrt = gee_img_dir / f"{region}_20m_{date_str}.vrt"
    write_simple_vrt(gee_vrt, tif_name_in_same_dir=tif_name, width=5, height=4, num_bands=11)

    # Load the two CLI modules from the provided files
    root = Path(__file__).resolve().parents[2]  # project root
    base_import_dir = root / "vercye_ops" / "lai"

    stac_cli = load_module_from_path("lai_stac_cli", base_import_dir / "lai_creation_STAC" / "2_1_primary_LAI_tiled.py")
    gee_cli = load_module_from_path("lai_gee_cli", base_import_dir / "lai_creation_GEE" / "2_1_primary_LAI_GEE.py")

    runner = CliRunner()
    res1 = runner.invoke(
        stac_cli.main,
        [
            str(tiled_img_dir),
            str(out_tiled),
            "20",
            "--num-cores",
            "1",
            "--satellite",
            "S2",
        ],
    )
    assert res1.exit_code == 0, f"Tiled CLI failed: {res1.output}\n{res1.exception}"

    res2 = runner.invoke(
        gee_cli.main,
        [
            str(gee_img_dir),
            str(out_gee),
            region,
            "20",
            "--start_date",
            "2025-01-01",
            "--end_date",
            "2025-12-31",
        ],
    )
    assert res2.exit_code == 0, f"GEE CLI failed: {res2.output}\n{res2.exception}"

    out1 = next(out_tiled.glob("*_LAI_tile.tif"))
    out2 = next(out_gee.glob("*_LAI.tif"))

    with rio.open(out1) as d1, rio.open(out2) as d2:
        a1 = d1.read(1)
        a2 = d2.read(1)

    np.testing.assert_allclose(a1, a2, rtol=1e-6, atol=1e-6, equal_nan=True)
