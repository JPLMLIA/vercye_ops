import numpy as np
import rasterio as rio

RASTER_PATHs = [
    "/gpfs/data1/cmongp2/vercye/data/cropmasks/Ukraine_ver6_1_2018_ww_map.tif",
    "/gpfs/data1/cmongp2/vercye/data/cropmasks/Ukraine_ver6_1_2019_ww_map.tif",
    "/gpfs/data1/cmongp2/vercye/data/cropmasks/Ukraine_ver6_2_2020_ww_map.tif",
]
THRESHOLD = 1


def binarize_raster(raster: np.ndarray, treshold: float):
    raster_binary = np.where(raster >= THRESHOLD, 1, 0)
    return raster_binary


def main():
    for raster_path in RASTER_PATHs:
        with rio.open(raster_path) as src:
            if src.count > 1:
                raise ValueError("Only handling single band rasters")

            raster = src.read(1)
            raster_binary = binarize_raster(raster, THRESHOLD)
            profile = src.profile.copy()

        output_path = raster_path.replace(".tif", "_binary.tif")
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(raster_binary.astype(rio.uint8), 1)
        print(f"Raster sucessfully binarized and saved under {output_path}!")


if __name__ == "__main__":
    main()
