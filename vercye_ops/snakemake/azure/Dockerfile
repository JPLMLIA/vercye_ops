# Container image for VeRCYe pipeline execution on Azure Batch.
#
# The same image runs on the head VM (driving snakemake) and on every Batch
# node (executing group tasks). Build once per release, push to ACR, and
# reference as AZ_BATCH_CONTAINER_IMAGE in profiles/azure-batch/config.yaml.
#
# Build (from the repo root):
#   docker build -f vercye_ops/snakemake/azure/Dockerfile \
#       --build-arg APSIM_IMAGE=apsiminitiative/apsimng:<tag> \
#       -t <acr>/vercye:<tag> .
# Push:
#   az acr login --name <acr>; docker push <acr>/vercye:<tag>

# Stage 1: grab the APSIM NG publish tree from the pinned official image.
ARG APSIM_IMAGE=apsiminitiative/apsimng:latest
FROM ${APSIM_IMAGE} AS apsim-src
# Nothing to do here - we only COPY --from this stage below.

# Stage 2: vercye runtime image.
FROM mambaorg/micromamba:1.5.8

USER root
SHELL ["/bin/bash", "-c"]

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq tini unzip bzip2 wget gnupg \
    && wget -q https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O /tmp/msprod.deb \
    && dpkg -i /tmp/msprod.deb \
    && rm /tmp/msprod.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        dotnet-runtime-8.0 \
        libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSfL https://aka.ms/downloadazcopy-v10-linux \
        | tar -xz --strip-components=1 -C /usr/local/bin --wildcards '*/azcopy'

COPY --from=apsim-src /opt/apsim /opt/apsim
ENV PATH=/opt/apsim:${PATH}

# Conda env
# Reuse the on-prem spec verbatim so Python / GDAL / LAI behaviour matches.
COPY environment/environment.yaml /tmp/environment.yaml
RUN micromamba create -y -n vercye -f /tmp/environment.yaml \
    && micromamba clean -a -y
ENV PATH=/opt/conda/envs/vercye/bin:${PATH}
ENV CONDA_DEFAULT_ENV=vercye

# Snakemake Azure plugins 
RUN /opt/conda/envs/vercye/bin/pip install --no-cache-dir \
        snakemake-executor-plugin-azure-batch \
        snakemake-storage-plugin-azure

# Project source 
# Copied last to maximise layer cache hits on iterative rebuilds.
WORKDIR /app
COPY . /app
RUN /opt/conda/envs/vercye/bin/pip install --no-cache-dir -e .

# Build-time smoke test
RUN Models --version \
    && /opt/conda/envs/vercye/bin/python -c \
        "import snakemake, rasterio, geopandas, pathvalidate, \
                snakemake_storage_plugin_azure, \
                snakemake_executor_plugin_azure_batch; \
         print('smoke-ok')"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
