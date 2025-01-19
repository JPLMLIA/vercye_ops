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