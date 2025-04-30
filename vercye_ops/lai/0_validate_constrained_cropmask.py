import click
import numpy as np
import rasterio as rio

@click.command()
@click.option('--input-path', help='Path to the binary input raster to validate.', type=click.Path(exists=True))
@click.option('--output-path', help='Path to the output file that indicates whether the raster has a positive pixel.', type=click.Path(exists=False))
def main(input_path, output_path):
    with rio.open(input_path) as src:
        raster = src.read(1)
        valid = np.any(raster > 0)
        valid_str = 'valid' if valid else 'invalid'

        with open(output_path, '+w') as out_file:
            out_file.write(valid_str)

if __name__ == "__main__":
    main()