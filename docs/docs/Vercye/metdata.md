# Meteorologcal Data Inputs

Meteorological data can be acquired from `NASA Power`, `ECMWF ERA5` and `CHIRPS`. Hereby the base meteorological data must be acquired from either `NASA Power` or `ERA5` and the precipitation data can be supplemented with `CHIRPS`.

## NASA Power
The `NASA Power` meteorological data is fetched through the `NASA Power daily point API` (https://power.larc.nasa.gov/api/temporal/daily/point). 
The API imposes rather strict limitations on the number of parallel requests and querying the same internal grid cell multiple times. In order to avoid being blacklisted, a caching mechanism is implemented. This represents a polygon by the Nasa Power grid cells centroid of the polygons centroid. So if `(x,y)` is the polygons centroid, we lookup into which `NASA Power` grid cell `(x,y)` would fall and query the grid cells centroid `(x', y')` as a representative of our polygon. The `NASA Power` grid cell is split into cells of `0.5 x 0.625` deg resolution for meteorological products and `1x1` degree resolution for solar parameters. **However, currently we are only using the `0.5 x 0.625`, also for solar parameters, which might possibly lead to misalignment!**

## ERA5
The `ERA5` data is fetched through `Google Earth Engine`. Therefore, this requires you to authenticate with your `Earth Engine` project.
During execution of the pipeline, the meteorological data is then fetched as the centroid or mean of a region from `Earth Engine`.

If you plan on using this approach you will have to run `earthengine authenticate` and complete the steps in the terminal before starting the pipeline.

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

!Attention: This can amount to a few hundred GB of data when downloading many years of historical data. Therefore this is rather intended to be run on HPC environments.

This will first try to download all final [CHIRPS v2.0 global daily products](https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/cogs/p05/) at 0.05 degrees resolution. For days without data available, the downloader will fallback to the [preliminary product](https://data.chc.ucsb.edu/products/CHIRPS-2.0/prelim/global_daily/tifs/p05/).

The VeRCYe pipeline will then read local regions from this global files during runtime.
