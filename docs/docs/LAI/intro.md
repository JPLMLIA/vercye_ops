# VeRCYe LAI Generation

VeRCYE is designed to identify best matching APSIM simulations with actual remotely sensed Leaf Area Index (LAI) data. This documentation covers the LAI data generation pipeline, which is a separate but essential component of the VeRCYE workflow. The LAI is not a true remotely sensed value, but is rather estimated from a 
combination of bands using neural networks that were converted to Pytorch from the [Leaf Toolbox](https://github.com/rfernand387/LEAF-Toolbox).

The LAI creation pipeline described in this documentation transforms Sentinel-2 satellite imagery into LAI estimates that can be used by the VeRCYE algorithm. Currently only Sentinel-2 data is supported, but a version using Harmonized Landsat-Sentinel Imagery is planned.

We've intentionally decoupled the LAI generation from the standard VeRCYE pipeline for several practical reasons:
1. It requires downloading and processing substantial amounts of satellite data
2. Once generated, LAI data for a region can be reused with different VeRCYE configurations

## LAI Generation
We provide two methods for exporting remotely sensed imagery and deriving LAI products:

A: Exporting RS imagery from **Google Earth Engine**
B: Downloading RS imagery through an open source **STAC catalog** and data hosted on AWS.

### A: Google Earth Engine Exported
This approach lets Google Earth Engine handle the majority of data preprocessing and exports daily Sentinel-2 Mosaics from Google Earth Engine to either a
Google Drive or a Google Cloud Storage Bucket, from which it can be downloaded to the HPC environment for deriving the LAI product. The data in Google Earth Engine is the `S2_SR_Harmonized` collection, in which all data is harmonized to be in the same range, even if different processing baselines are used.

**Pro:**
- Directly Export Mosaics with bands resampled to the same resolution and CRS
- Strong Cloud Masking Algorithm (Google Cloud Score Plus)

**Con**
- Slow for large regions, due to limited number of parallel export processes
- Exported data is exported to either Google Drive (Free) or Google Cloud Storage (Fees apply), and downloaded from there, but requires more manual setup which might be tedious especially on remote systems.

### B: STAC & AWS Export
This approach queries a STAC catalog hosted by Element84 on AWS to identify all Sentinel-2 Tiles intersecting the region of interest within the timespan. The individual tiles are then downloaded from an AWS bucket. This data uses the `Sentinel-2 L2A Colection 1` data in which all historial data was processed using `Processing Baseline 5.0`.

**Pro**:
- Very Fast Download in HPC environment due to high level of parallelism
- Completely free download of data
- Harmonized to data in `Sentinel-2 L2A Colection 1` - all data processed using modern Baseline 5.0.

**Con**:
- Less Accurate Cloudmask in comparison to Google Cloud Score Plus. Cloud mask is based on SCL + S2-Cloudless.
- As of May 27th 2025, `Sentinel-2 L2A Colection 1` does not contain data for 2022 and parts of 2023. According to ESA this backfill is scheduled to be completed until Q2 2025.

Instructions on how to run each LAI creation pieline are detailed in [LAI Generation Instructions](running.md).