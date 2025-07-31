"""Helper to take in a template apsimx file and update the .met file (containing weather information)"""

import click
import json
from pathlib import Path

from vercye_ops.utils.init_logger import get_logger

logger = get_logger()


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
                logger.info('Replacing entry "%s:%s"', key, value)
                json_data[key] = new_value
            elif isinstance(value, (dict, list)):
                recursive_update(value, key_to_update, new_value, file_suffix, verbose)
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                recursive_update(item, key_to_update, new_value, file_suffix, verbose)

def get_nested_object_by_name(json_data, object_name):
     """
    Recursively walk through JSON-like structure and find a nested object with a specific name.

    Parameters
    ----------
    json_data : dict or list
        The APSIM JSON data.
    object_name : str
        The name of the object to find.

    Returns
    -------
    dict
        The JSON object with Name==object_name
    """
    if isinstance(json_data, dict):
        if json_data.get("Name") == object_name:
            return json_data
        for value in json_data.values():
            if isinstance(value, (dict, list)):
                return get_nested_object_by_name(value, object_name)

    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, (dict, list)):
                return get_nested_object_by_name(item, object_name)


def update_object_property(json_data, object_property, target_value, verbose=False):
    if not object_property in json_data:
        raise ValueError(f'{object_property} not found in dictionary.')

    json_data[object_property] = target_value
    if verbose:
        print(f'Updated {object_property} to {target_value}')

def update_kv_list(search_list, key, value):
    for item in search_list:
        if item.get('Key') == key:
            item['Value'] = value
            return
    
    raise ValueError(f'No item with key "{key} found in {search_list}.')

                
@click.command()
@click.option('--apsimx_template_fpath', type=click.Path(exists=True, dir_okay=False), required=True, help="Path to the .apsimx file which is a JSON.")
@click.option('--apsimx_output_fpath', type=click.Path(writable=True, dir_okay=False), required=True, help="Location to save the modified .apsimx file.")
@click.option('--new_met_fpath', type=click.Path(writable=True, dir_okay=False), required=True, help="Filepath to the new .met data.")
@click.option('--sowing_date', type=click.DateTime(formats=["%Y-%m-%d"]), required=False, help="True sowing date. Will replace sowing window / sowing date factorial.")
@click.option('--verbose', is_flag=True, help="Enable verbose output.")
def cli(apsimx_template_fpath, apsimx_output_fpath, new_met_fpath, sowing_date, verbose):
    """Update an .apsimx file with new fields and save the updated version."""
    if verbose:
        logger.setLevel('INFO')

    # Load APSIM file
    apsimx_template_fpath = Path(apsimx_template_fpath)
    with open(apsimx_template_fpath, 'r') as file:
        json_data = json.load(file)   
    
    # Update metfile path
    recursive_update(json_data, 'FileName', '.met', new_met_fpath, verbose)

    # Use real sowing date
    if sowing_date:
        # Convert date to string like 15-sep for APSIM
        apsim_date_str = sowing_date.strftime("%-d-%b").lower()

        # Disable sowing date factorial
        sowing_date_obj = get_nested_object_by_name(json_data, object_name='SowingDate')
        update_object_property(object_property='Enabled', target_value=False, vebose=verbose)

        # Enforce fixed sowing date
        sowing_rule_manager_obj = get_nested_object_by_name(json_data, object_name='SowingRule')
        sowing_rule_manager_parameters = sowing_rule_manager_obj['Parameters']
        update_kv_list(sowing_rule_manager_parameters, key='StartDate', value=apsim_date_str)
        update_kv_list(sowing_rule_manager_parameters, key='EndDate', value=apsim_date_str)
        update_kv_list(sowing_rule_manager_parameters, key='ForceSowing', value=apsim_date_str)

    # Save out updated .apsimx json to disk
    with open(apsimx_output_fpath, 'w') as file:
        json.dump(json_data, file, indent=2)

    logger.info("Updated file saved to %s", apsimx_output_fpath)

if __name__ == "__main__":
    cli()