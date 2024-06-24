# Manual APSIM Install
Visit [https://www.apsim.info](https://www.apsim.info) and make sure you have the proper license.

The below instructions show how to install onto RHEL. This involves
* DOTNET installation
* Building APSIM from source
* Confirming APSIM example runs

```
export APSIM_SRC_ROOT=/desired/path/to/ApsimX
############################
# DOTNET install

# In future, SAs could install via yum
sudo yum install dotnet-sdk-6.0

# Manual DOTNET install
# Get install script
wget https://dot.net/v1/dotnet-install.sh

# Run the install with latest DOTNET version specified. Available versions here: https://dotnet.microsoft.com/en-us/download/dotnet/6.0
./dotnet-install.sh \
--version 6.0.422 \
--install-dir path/to/dotnet

# Export DOTNET_ROOT to be the same as `install-dir` above
# ADD THIS TO YOUR .bashrc
export DOTNET_ROOT=path/to/dotnet

############################
# Build APSIM
module load git  # Might not be needed

cd ${APSIM_SRC_ROOT}
${DOTNET_ROOT}/dotnet build -c Release ApsimX.sln

############################
# Run APSIM on Example
# Also seems to work with the apsim or ApsimNG executables
${APSIM_SRC_ROOT}/bin/Release/net6.0/Models \
--verbose \
${APSIM_SRC_ROOT}/Examples/Wheat.apsimx

# Check if Wheat simulation results are present
ls ${APSIM_SRC_ROOT}/Examples/
```