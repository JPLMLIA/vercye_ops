The `VeRCYe CLI` (command line interface), simplifies data ingestion and execution of yield studies.

### Usage

Make sure you have folled the setup instructions and have installed `vercye` with `pip install -e .` in this project's root directory and have activated your virtual environment.

**Optional: Set up environemental defaults**

If you have a system wide installation and want all studies to be saved in the same directory, copy the `.env_example` file to `.env` in the projects root. And set the paths to your shared storage and cache directories. Then, you will not have to provide the `--dir` option in the following instructions.


0. Activate your virtual environment with the `vercye` instalation (depending on your venv setup). E.g:

```bash
conda activate vercye
```

1. Initialize a new yield study.

```bash
vercye init --name your-study-name --dir /path/to/study/store
```

2. [Optional] Download remotely sensed imagery & create LAI.

- Only required if the LAI data for your region of interest is not yet available locally.
- Fill in the lai configuration under `/path/to/study/store/lai_config.yaml`.
- Run the following command to download the imagery:
```bash
vercye lai --name your-study-name --dir /path/to/study/store
```

3. Prepare your study

- Fill in the run congfiguration under `/path/to/study/store/setup_config.yaml`.
- Run the following command to create your study directory and config template.
```bash
vercye prep --name your-study-name --dir /path/to/study/store
```
4. Set your run options

- Fill in the run congfiguration under `/path/to/study/store/study/config.yaml`.
- If you already had the LAI data available locally, ensure to adapt the `lai_dir`, `lai_region` and `lai_resolution`.


5. [Optional] Download the chirps data.

- Only required if chirps data is not yet downloaded for the complete study range.
- Ensure you have completed step 4, and have set `apsim_params.chirps_dir` correctly in `/path/to/study/store/study/config.yaml`.

```bash
vercye chirps --name your-study-name --dir /path/to/study/store
```

6. [Optional] Authenticate with EarthEngine

- Only required if using ERA5 meteorological data.
- Requires authenticating with your earth engine account to submit jobs.

```bash
earthengine authenticate
```

Run this command and follow the promtes in the terminal.

7. Run your study

- You might want to adapt the number of cores to use in /path/to/study/store depending on your system.

```bash
vercye run --name your-study-name --dir /path/to/study/store/profile/config.yaml
```