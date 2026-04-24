"""Patch merged-LAI VRTs so they read as Float32 with NaN nodata when the
underlying per-tile standardized-LAI rasters are stored as Int16 with
scale=0.001 and nodata=-32768.

gdalbuildvrt preserves the source dtype (Int16) and does not apply the scale
factor automatically on read. Rasterio/GDAL only apply <ScaleRatio> when the
VRT band's dataType is a float wider than the source and the ComplexSource
declares <ScaleRatio> + <ScaleOffset> + per-source <NODATA>. This script
rewrites the XML accordingly.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

SCALE_RATIO = 0.001
SCALE_OFFSET = 0.0
RAW_NODATA = -32768


def patch_vrt_for_int16_sources(vrt_path):
    """Rewrite a VRT produced by gdalbuildvrt over Int16 tiles so that
    downstream reads return Float32 with NaN at nodata positions.

    Idempotent: safe to run twice. Returns True if modifications were written.
    """
    vrt_path = str(vrt_path)
    tree = ET.parse(vrt_path)
    root = tree.getroot()
    modified = False

    for band in root.findall("VRTRasterBand"):
        if band.get("dataType") != "Float32":
            band.set("dataType", "Float32")
            modified = True

        nd = band.find("NoDataValue")
        if nd is None:
            nd = ET.SubElement(band, "NoDataValue")
            modified = True
        if (nd.text or "").strip().lower() != "nan":
            nd.text = "nan"
            modified = True

        for src in list(band.findall("ComplexSource")) + list(band.findall("SimpleSource")):
            if src.tag == "SimpleSource":
                src.tag = "ComplexSource"
                modified = True

            # Remove any previous ScaleOffset / ScaleRatio so we write known values
            for tag in ("ScaleOffset", "ScaleRatio"):
                for el in list(src.findall(tag)):
                    src.remove(el)
                    modified = True

            nodata_el = src.find("NODATA")
            if nodata_el is None:
                nodata_el = ET.SubElement(src, "NODATA")
                modified = True
            if (nodata_el.text or "").strip() != str(RAW_NODATA):
                nodata_el.text = str(RAW_NODATA)
                modified = True

            idx = list(src).index(nodata_el)
            so = ET.Element("ScaleOffset"); so.text = str(SCALE_OFFSET)
            sr = ET.Element("ScaleRatio");  sr.text = str(SCALE_RATIO)
            src.insert(idx, so)
            src.insert(idx + 1, sr)
            modified = True

            sp = src.find("SourceProperties")
            if sp is not None and sp.get("DataType") != "Int16":
                sp.set("DataType", "Int16")
                modified = True

    if modified:
        tree.write(vrt_path, xml_declaration=False, encoding="utf-8")
    return modified


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        changed = patch_vrt_for_int16_sources(p)
        print(f"{'patched' if changed else 'unchanged'} {p}")
