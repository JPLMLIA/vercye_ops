"""Helper to take in a template apsimx file and update the .met file (containing weather information)"""

import click
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def recursive_update(json_data, key_to_update, file_suffix, new_value, verbose=False):
    """
    Recursively walk through a dictionary, updating all occurrences of a specific key that end with a certain suffix.

    Parameters
    ----------
    json_data : dict or list
        JSON data as a dictionary or list (if nested).
    key_to_update : str
        The key to look for and update.
    file_suffix : str
        The file suffix to match the current value before updating.
    new_value : str
        The new value to replace with.
    verbose: bool
        Whether or not to log messages.

    Returns
    -------
    None
        Modifies the dictionary in place.
    """

    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if key == key_to_update and isinstance(value, str) and value.endswith(file_suffix):
                if verbose:
                    logging.info('Replacing entry "%s:%s"', key, value)
                json_data[key] = new_value
            elif isinstance(value, (dict, list)):
                recursive_update(value, key_to_update, new_value, file_suffix, verbose)
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                recursive_update(item, key_to_update, new_value, file_suffix, verbose)

                
@click.command()
@click.option('--apsimx_template_fpath', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the .apsimx file which is a JSON.")
@click.option('--apsimx_output_fpath', type=click.Path(writable=True, dir_okay=False), required=True, help="Location to save the modified .apsimx file.")
@click.option('--new_met_fpath', type=click.Path(writable=True, dir_okay=False), required=True, help="Filepath to the new .met data.")
@click.option('--verbose', is_flag=True, help="Enable verbose output.")
def cli(apsimx_template_fpath, apsimx_output_fpath, new_met_fpath, verbose):
    """Update an .apsimx file with new fields and save the updated version."""

    apsimx_template_fpath = Path(apsimx_template_fpath)
    with open(apsimx_template_fpath, 'r') as file:
        json_data = json.load(file)   
    
    recursive_update(json_data, 'FileName', '.met', new_met_fpath, verbose)

    # Save out updated .apsimx json to disk
    with open(apsimx_output_fpath, 'w') as file:
        json.dump(json_data, file, indent=2)

    if verbose:
        logging.info("Updated file saved to %s", apsimx_output_fpath)

if __name__ == "__main__":
    cli()