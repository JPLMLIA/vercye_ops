# VeRCYe Documentation

This documentation aims to provides a comprehensive guide to running the Versatile Crop Yield Estimate (VeRCYe) pipeline. The original VeRCYe algorithm is [published here](https://doi.org/10.1007/s13593-024-00974-4).

Currently the documentation is a work in progress, so please visit again soon.

### Features
* Tools to greatly reduce manual effort required for executing the VERYCE crop yield estimate pipeline.
* All workflow steps are wrapped in a well-documented CLI interface to permit step by step execution.
* The core CLI steps are also wrapped in a Snakemake-based data processing pipeline to batch execute yield estimates in an easy to run and reproducible manner.


### Overview
The VeRCYe pipeline is split into two main components:
- **LAI Generation**: Downloading of remotely sensed imagery and prediction of Leaf Area Index (LAI) values per pixel.
- **Yield Simulation and Prediction**: Simulation of numerous likely configurations with APSIM and identification of the best matching simulations
with the LAI data. Also includes evaluation and reporting tools.

### Setup
1. **Install the requirements**: Navigate to this package's root directory and then run:

`conda install --yes --file requirements.txt`
or
`pip install -r requirements.txt`

Note: As of June 2024, if you use conda, you may need to manually install `Snakemake` and a specific dependency via pip: `pip install snakemake pulp=2.7.0`

2. **Install the VeRCYe package**: Navigate to this package's root directory and then run:
`pip install -e .`

3. **Install APSIMX**: There are two ways of running APSIM. Either by using the APSIM docker container or by downloading and building the APSIM binary. For the first option, you will only have to set a parameter later when configuring your yieldstudy and the docker container will be automatically build (Ensure you have `docker` installed). For the second option, see the instructions at [vercye_ops/apsim/README.md](vercye_ops/apsim/README.md).

Note: If running on the UMD Systems, APSIM is already installed under `/gpfs/data1/cmongp2/wronk/Builds/ApsimX/bin/Release/net6.0/Models`.

4. **Install jq**: To manipulate json files via the command line, install https://jqlang.github.io/jq/ or if on HPC resources, you might be able to activate it with `module load jq`

### Running your first yield study
You will first have to generate **LAI** data from remotely sensed imagery. Refer to the [LAI Creation Guide]() for details.

Once you have generated the **LAI** data, you can run your yield study, by following the [Running a Yieldstudy Guide]().

### Technical Details
Coming soon.


### Development
Development tipps and best practices will be filled in here soon.