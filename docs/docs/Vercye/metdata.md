# Meteorologcal Data Inputs

TODO description of the different met sources will follow soon.

## NASA Power

## ERA5

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
