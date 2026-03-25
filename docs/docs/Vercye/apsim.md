APSIM is an agricultural modelling framework that can simulate a variety of biophysical processes for different crops.

VeRCYe relies on the APSIM Next-Gen framework for generating various simulations in a realistic range of input parameters (management practices, soils, water, etc. ).

### Installing APSIMX
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
chmod +x dotnet-install.sh

# Run the install with latest DOTNET version specified. Available versions here: https://dotnet.microsoft.com/en-us/download/dotnet/8.0
# APSIM might change it's dependancies in newer versions. As for 14.11.2025, Dotnet 8.0 is required
./dotnet-install.sh \
--version 8.0.416 \
--install-dir path/to/dotnet

# Export DOTNET_ROOT to be the same as `install-dir` above
# ADD THIS TO YOUR .bashrc otherwise you will have to expert this before each run manually
export DOTNET_ROOT=path/to/dotnet

############################
# Build APSIM

git clone https://github.com/APSIMInitiative/ApsimX.git ${APSIM_SRC_ROOT}

cd ${APSIM_SRC_ROOT}

# ATTENTION: Apsim changes it's behavior between different Versions!
# Make sure to install the same version that is being used to create APSIM-File templates. You can identify the commit to checkout as described here: https://github.com/APSIMInitiative/ApsimX/discussions/10663
# The current implementation is validated with the following commit (APSIM Version 2025.11.7927). Adapt to use newer versions!
git checkout 889e997

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
