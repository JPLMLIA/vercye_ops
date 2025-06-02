# VeRCYe Documentation

This documentation aims to provides a comprehensive guide to running the Versatile Crop Yield Estimate (VeRCYe) pipeline. The original VeRCYe algorithm is [published here](https://doi.org/10.1007/s13593-024-00974-4).

Currently the documentation is a work in progress, so please visit again soon.

### Features
* Tools to greatly reduce manual effort required for executing the VERYCE crop yield estimate pipeline.
* All workflow steps are wrapped in a well-documented CLI interface to permit step by step execution.
* The core CLI steps are also wrapped in a Snakemake-based data processing pipeline to batch execute yield estimates in an easy to run and reproducible manner.


### Overview

The **VeRCYe** pipeline is split into two main components:

- **LAI Generation**: Download remotely sensed imagery and predict Leaf Area Index (LAI) values per pixel.
- **Yield Simulation and Prediction**: Simulate numerous likely configurations using APSIM and identify the best-matching simulations with the LAI data. This step also includes evaluation and reporting tools.

---

### Setup

#### 0. Clone this repository

```bash
git clone https://github.com/JPLMLIA/vercye_ops.git
```

#### 1. Check Python Version and GDAL

This repository has been tested and run with `python 3.10.16` and with `gdal==3.1.0`. Ensure you have installed the corresponding versions (`python --version` and `gdalinfo --version`).
If you are running your code on a shared cluster, you might have to run `module load gdal/3.1.0`, before being able to use `GDAL`.

#### 2. Install the requirements

Navigate to this package's root directory and run:

```bash
conda install --yes --file requirements.txt
# or
pip install -r requirements.txt
```

> **Note**: As of June 2024, if using `conda`, you may also need to install `Snakemake` and a specific dependency manually via pip:
>
> ```bash
> pip install snakemake pulp==2.7.0
> ```

#### 3. Install the VeRCYe package

From the root directory, run:

```bash
pip install -e .
```

#### 4. Install APSIMX

There are two options for running APSIM:

- **Using Docker**: Simply set a parameter during configuration of your yield study. The Docker container will build automatically. (Ensure `docker` is installed.)
- **Building the binary manually**: See instructions in [vercye_ops/apsim/README.md](vercye_ops/apsim/README.md).

> **Note**: If running on UMD systems, APSIM is pre-installed at:
>
> ```
> /gpfs/data1/cmongp2/wronk/Builds/ApsimX/bin/Release/net6.0/Models
> ```

#### 5. Install `jq`

To manipulate JSON files via the command line:

- Install from [https://jqlang.github.io/jq/](https://jqlang.github.io/jq/)
- Or on HPC systems, load it with:

```bash
module load jq
```


### Running your first yield study
You will first have to generate **LAI** data from remotely sensed imagery. Refer to the [LAI Creation Guide](LAI/running.md) for details.

Once you have generated the **LAI** data, you can run your yield study, by following the [Running a Yieldstudy Guide](Vercye/running.md).


### Technical Details
The technical implementation details are outlined in he [Architecture Section](Vercye/architecture.md). Fore more details check out the code in `vercye_ops`.


### Development
Development tipps and best practices are documented under [Development Tipps](devtipps.md)