# Meteorologcal Data Inputs

TODO description of the different met sources will follow soon.

## NASA Power

## ERA5

## CHIRPS

**Downloading CHIRPS Precipitation Data**
If you plan on using CHIRPS precipitation data, you will have to download the complete historical and current CHRIPS archive before starting a yield study.
For this, we provide a helper script in `apsim/download_chirps_data.py`. Here you will have to specify the time range (beginning, -end date) and the data will be downloaded with daily global coverage.

!Attention: This can amount to a few hundred GB of data when downloading many years of historical data. Therefore this is rather intended to be run on HPC environments.