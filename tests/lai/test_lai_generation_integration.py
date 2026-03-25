"""Integration tests for LAI generation pipeline (STAC / Planetary Computer)."""

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


@pytest.mark.slow
@pytest.mark.parametrize("date_str", ["2025-06-15"])
def test_cli_stac_pipeline_produces_output(tmp_path, date_str):
    """
    Creates a real 11-band TIFF and VRT, runs the STAC pipeline CLI,
    and verifies it produces a valid LAI output.
    ATTENTION: Only the inference is tested, not the entire LAI generation process!
    """
    tiled_img_dir = tmp_path / "s2_tiled"
    out_tiled = tmp_path / "out_tiled"
    for p in (tiled_img_dir, out_tiled):
        p.mkdir(parents=True)

    tif_name = "S2_20m_stack.tif"
    tif_tiled = tiled_img_dir / tif_name
    write_s2_20m_like_tif(tif_tiled)

    tiled_vrt = tiled_img_dir / f"T31TCJ_20m_{date_str}.vrt"
    write_simple_vrt(tiled_vrt, tif_name_in_same_dir=tif_name, width=5, height=4, num_bands=11)

    root = Path(__file__).resolve().parents[2]
    base_import_dir = root / "vercye_ops" / "lai"

    stac_cli = load_module_from_path("lai_stac_cli", base_import_dir / "lai_creation_STAC" / "2_1_primary_LAI_tiled.py")

    runner = CliRunner()
    res = runner.invoke(
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
    assert res.exit_code == 0, f"STAC CLI failed: {res.output}\n{res.exception}"

    out_files = list(out_tiled.glob("*_LAI_tile.tif"))
    assert len(out_files) == 1, f"Expected 1 output LAI tile, got {len(out_files)}"

    with rio.open(out_files[0]) as src:
        data = src.read(1)
        assert data.shape == (4, 5)
        # At least some valid (non-NaN, non-zero) LAI values expected
        valid = data[~np.isnan(data) & (data != 0)]
        assert len(valid) > 0, "LAI output has no valid pixels"
