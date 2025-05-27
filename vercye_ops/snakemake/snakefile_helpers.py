import os.path as op
from pathlib import Path
from types import SimpleNamespace

def build_apsim_execution_command(head_dir, use_docker, docker_image, docker_platform, executable_fpath, n_jobs, input_file):
    '''Builds the APSIM execution command depending on whether we are using APSIM in Docker or not'''

    if use_docker:
        return (
            f'docker run -i --rm --platform={docker_platform} '
            f'-v "{head_dir}:{head_dir}" '
            f'-u $(id -u):$(id -g) '
            f'{docker_image} '
            f'{input_file} '
        )
    else:
        return (
            f'{executable_fpath} '
            f'{input_file} '
            f'--cpu-count {n_jobs} '
        )

def get_evaluation_results_path_func(config):
    '''Returns a function with the config hardcoded. The function return the path to the evaluation results file if it exists or an empty list if it does not.
        This allows the evaluation rule to be skipped if the evaluation results file does not exist.'''
    def get_evaluation_results_path(wildcards):
        output_paths = []
        # Check if the evaluation results at simulation level file exists
        # This file must always be called groundtruth_primary
        if op.exists(op.join(config['sim_study_head_dir'], wildcards.year, f'groundtruth_primary-{wildcards.year}.csv')):
            primary_eval_file = op.join(config['sim_study_head_dir'], wildcards.year, wildcards.timepoint, 'evaluation_primary.csv')
            output_paths.append(primary_eval_file)
        
        # Get all other evaluation results files from the config (wildcards)
        for agg_name in config['eval_params']['aggregation_levels']:
            gt_file = op.join(config['sim_study_head_dir'], wildcards.year, f'groundtruth_{agg_name}-{wildcards.year}.csv')
            if op.exists(gt_file):
                eval_file = op.join(config['sim_study_head_dir'], wildcards.year, wildcards.timepoint, f'evaluation_{agg_name}.csv')
                output_paths.append(eval_file)

        return output_paths
        
    return get_evaluation_results_path


def get_multiyear_evaluation_results_path_func(config):
    """ Function to get all evaluation results paths from all configs and timepoints"""
    single_year = get_evaluation_results_path_func(config)

    def _collect_all(wildcards):
        all_paths = []
        for year in config['years']:
            for tp in config['timepoints']:
                # build a fake wildcards object
                fake = SimpleNamespace(year=year, timepoint=tp)
                all_paths.extend(single_year(fake))
        return all_paths

    return _collect_all


def get_required_yield_report_suffix(config):
    """Get the required suffix for the yield report based on the configuration"""

    # Will always create a lightweight downsampled png, but html report is only created if specified
    if config["create_per_region_html_report"]:
        return "html"
    else:
        return "png"