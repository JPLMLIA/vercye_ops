import click
import numpy as np
import rasterio as rio

@click.command()
@click.option('--input-path', help='Path to the binary input raster to validate.', type=click.Path(exists=True))
@click.option('--output-path', help='Path to the output file that indicates whether the raster has a positive pixel.', type=click.Path(exists=False))
@click.option('--px-threshold', help='Minimum number of pixels to be considered as valid.', type=int, default=0, required=False)
def main(input_path, output_path, px_threshold):
    with rio.open(input_path) as src:
        raster = src.read(1)
        valid = np.sum(raster > 0) > px_threshold
        valid_str = 'valid' if valid else 'invalid'

        with open(output_path, 'w') as out_file:
            out_file.write(valid_str)

if __name__ == "__main__":
    main()