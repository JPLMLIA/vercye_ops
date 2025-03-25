import os.path as op

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
        
        if op.exists(op.join(config['sim_study_head_dir'], wildcards.year, 'groundtruth.csv')):
            return op.join(config['sim_study_head_dir'], wildcards.year, wildcards.timepoint, 'evaluation.csv')
        else:
            return []
        
    return get_evaluation_results_path