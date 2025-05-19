# LAI Generation
The pipeline produces LAI products for VERCYe and is intended for scaled deployment on servers or HPC with minimal human intervention. The Sentinel-2 LAI model is by Fernandes et al. from https://github.com/rfernand387/LEAF-Toolbox. We provide two methods for exporting remotely sensed imagery and deriving LAI products:


A: Exporting RS imagery from **Google Earth Engine**
B: Downloading RS imagery through an open source **STAC catalog** and data hosted on AWS.

The individual advantages are detailed in the [introduction](intro.md#lai-generation). This document details the instruction on how to download remotely sensed imagery and derive LAI data. For both approaches we provide pipelines that simply require specifying a configuration and then handle the complete process from exporting and downloading remotely sensed imagery to cloudmasking and deriving LAI estimates.


### Prequisites

Install the requirements as detailed in [Introduction - Setup](../index.md#setup)

### A - Google Earth Engine Pipeline

1. Extract geosjsons from a shapefile for each region of interest

Todo documentation

Snakemake based pipeline under `lai/lai_generation_GEE`


### B - STAC Catalog & AWS Pipeline

Todo documentation

Pyton based pipeline under `lai/lai_creation_STAC/run_stac_dl_pipeline.py`