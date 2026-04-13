# Meteorologcal Data Inputs

Meteorological data can be acquired from `NASA Power`, `ECMWF ERA5` and `CHIRPS`. Hereby the base meteorological data must be acquired from either `NASA Power` or `ERA5` and the precipitation data can be supplemented with `CHIRPS`.

## NASA Power
The `NASA Power` meteorological data is fetched through the `NASA Power daily point API` (https://power.larc.nasa.gov/api/temporal/daily/point).
The API imposes rather strict limitations on the number of parallel requests and querying the same internal grid cell multiple times. In order to avoid being blacklisted, a caching mechanism is implemented. This represents a polygon by the Nasa Power grid cells centroid of the polygons centroid. So if `(x,y)` is the polygons centroid, we lookup into which `NASA Power` grid cell `(x,y)` would fall and query the grid cells centroid `(x', y')` as a representative of our polygon. The `NASA Power` grid cell is split into cells of `0.5 x 0.625` deg resolution for meteorological products and `1x1` degree resolution for solar parameters. **However, currently we are only using the `0.5 x 0.625`, also for solar parameters, which might possibly lead to misalignment!**

## ERA5
The `ERA5` data is fetched through `Google Earth Engine`. Therefore, this requires you to authenticate with your `Earth Engine` project.
During execution of the pipeline, the meteorological data is then fetched as the centroid or mean of a region from `Earth Engine`.

You can authenticate in two ways:

### Option 1: Service account (recommended for headless/server setups)

Create a service account in your GCP project and download a JSON key. The service account needs to be:

1. **Registered with Earth Engine** at https://signup.earthengine.google.com/#!/service_accounts
2. Granted the following IAM roles on the GCP project:
    - `roles/serviceusage.serviceUsageConsumer` — allows billing API calls to the project
    - `roles/earthengine.viewer` — allows running EE computations and reading public datasets (use `roles/earthengine.writer` instead if you also need to run the GEE LAI export pipeline)

Then set the path to the key file in your `.env`:

```
EE_SERVICE_ACCOUNT_KEY=/path/to/your/service-account-key.json
EE_PROJECT_NAME=your-gcp-project-id
```

The pipeline will pick this up automatically — no interactive login required.

### Option 2: Interactive user authentication

If you are on a machine with a browser available, run `earthengine authenticate` and complete the browser flow before starting the pipeline. Leave `EE_SERVICE_ACCOUNT_KEY` unset in your `.env`.

## CHIRPS

**Downloading CHIRPS Precipitation Data**
If you plan on using CHIRPS precipitation data, you will have to download the complete historical and current CHRIPS archive before starting a yield study.
For this, we provide a helper script in `met_data/download_chirps_data.py`. Here you will have to specify the time range (beginning, -end date) and the data will be downloaded with daily global coverage.

```
Usage: download_chirps_data.py [OPTIONS]

Options:
  --start-date [%Y-%m-%d]  Start date for CHIRPS data collection in YYYY-MM-DD
                           format.  [required]
  --end-date [%Y-%m-%d]    End date for CHIRPS data collection in YYYY-MM-DD
                           format.  [required]
  --output-dir TEXT        Output directory to store the CHIRPS data.
                           [required]
  --num-workers INTEGER    Number of parallel processes. Capped at 10 due to
                           server limitations.  [default: 10]
  --verbose                Enable verbose logging.
  --help                   Show this message and exit.
```

!Attention: This can amount to 100+ GB of data when downloading many years of historical data. Therefore this is rather intended to be run on HPC environments.

This will first try to download all final [CHIRPS v2.0 global daily products](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/) at 0.05 degrees resolution. For days without data available, the downloader will fallback to the [preliminary product](https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/).

The VeRCYe pipeline will then read local regions from these global files during runtime.
