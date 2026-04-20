# Azure Batch execution

Alternate execution path for operational runs at scale (admin-3, whole countries,
many regions). The on-prem Snakefile and `profiles/hpc/` stay the canonical
local/cluster path - nothing here touches them.

## What's different from the on-prem pipeline

| Concern | On-prem (`Snakefile`) | Azure (`Snakefile.azure`) |
| --- | --- | --- |
| Executor | local cores | Azure Batch pool |
| Per-region rules | one job each | bundled into two group keys (`region_validate_{region}`, `region_pipe_{region}`) |
| Per-region intermediates | kept on disk | marked `temp()`, reclaimed when the group task finishes |
| Shared filesystem | assumed (NFS/local FS) | none - Blob only, driven by `snakemake-storage-plugin-azure` |
| `kaleido_resources` throttle | active (shared-cluster nproc cap) | removed (each Batch task runs in isolation) |
| `mem_mb` / `runtime` | mostly absent | set on heavy rules so Batch can size nodes |
| Checkpoint `all_regions_validated` | unchanged | unchanged - `_VALID` flags are persisted outputs |

## Storage model (no NFS, no Lustre, no ANF)

Three tiers, all cheap:

1. **Head VM local Premium SSD** - holds the Snakemake working tree while the
   driver runs. DAG stats / checkpoint globbing hit local disk, not Blob.
2. **Batch node local NVMe** (`$AZ_BATCH_NODE_ROOT_DIR`) - per-task scratch.
   APSIM `.db` files, intermediate met CSVs, and the generated `.apsimx`
   file all live here and die with the task (marked `temp()`).
3. **Blob** - cold inputs (CHIRPS, LAI VRTs, cropmasks) and final artifacts.
   Accessed via `/vsiaz/` in rasterio or via `az://` prefixes in the Snakemake
   storage plugin.

## One-time setup

1. **Build and push the container**
   ```bash
   az acr login --name <acr>
   docker build -f vercye_ops/snakemake/azure/Dockerfile \
       --build-arg APSIM_IMAGE=apsiminitiative/apsimng:<tag> \
       -t <acr>.azurecr.io/vercye:<tag> .
   docker push <acr>.azurecr.io/vercye:<tag>
   ```

   `APSIM_IMAGE` defaults to `apsiminitiative/apsimng:latest`. Pin it to
   the exact tag your on-prem runs already use
   (`apsim_execution.docker.image` in run_config) so the APSIM version is
   identical between the two execution paths.

   What the Dockerfile actually builds:
   - **Base**: `mambaorg/micromamba:1.5.8` (Debian bookworm-slim with
     micromamba). *Not* the apsiminitiative image - that one is a minimal
     APSIM-only runtime and layering a geo conda stack on top is fragile.
   - **APSIM**: `/opt/apsim/` copied verbatim from
     `${APSIM_IMAGE}` via a multi-stage `COPY --from`. Same binaries the
     on-prem path runs, just imported into our own controlled image.
   - **.NET 8 runtime**: installed explicitly from Microsoft's Debian 12
     apt repo (`dotnet-runtime-8.0`). APSIM is published
     `--no-self-contained` so it needs the system runtime.
   - **SQLite runtime**: `libsqlite3-0` (APSIM Models links to it at run
     time when writing simulation DBs).
   - **azcopy**: on `PATH`, used by `snakemake-storage-plugin-azure`.
   - **Conda env**: materialised from `environment/environment.yaml` at
     `/opt/conda/envs/vercye/`, on `PATH`.
   - **Snakemake plugins**: `snakemake-executor-plugin-azure-batch` and
     `snakemake-storage-plugin-azure` installed into the conda env.

   A build-time smoke test (`Models --version` + importing the conda deps
   and both Snakemake plugins) fails the build if any of the above is
   broken, so bad images never reach Batch.

   Your Azure `run_config.yaml` should set:
   ```yaml
   apsim_execution:
     use_docker: False     # APSIM lives in the Batch container already
     local:
       executable_fpath: /opt/apsim/Models
       dotnet_root: /usr/share/dotnet  # matches dotnet-runtime-8.0 install
       n_jobs: 8
   ```
2. **Create a Batch pool** backed by spot VMs (e.g. `Standard_F32s_v2`) with
   the container image above and an ephemeral OS disk on local NVMe. Use an
   autoscale formula that tracks pending task count so the pool scales 0 → N
   and back down between runs.
3. **Stage inputs** to a Blob container (`inputs/`): CHIRPS NetCDF, cropmasks,
   LAI rasters, shapefiles. Reference them in `run_config.yaml` using `az://`
   or `/vsiaz/` URLs.

## LAI rasters - STAC pipeline hand-off

LAI production is a separate, pre-pipeline stage that runs on its own Azure
VM via `vercye_ops/lai/lai_creation_STAC/run_stac_dl_pipeline.py`. It is not part of
the Batch DAG; it runs once per country × season and its outputs feed many
downstream Snakemake runs.

## Running a study

On the head VM:

```bash
export AZ_BATCH_ACCOUNT_URL=https://<acct>.<region>.batch.azure.com
export AZ_BATCH_ACCOUNT_KEY=...
export AZ_BLOB_ACCOUNT_URL=https://<stg>.blob.core.windows.net
export AZ_BLOB_ACCOUNT_KEY=...
export AZ_BATCH_POOL_ID=vercye-pool
export AZ_BATCH_CONTAINER_IMAGE=<acr>.azurecr.io/vercye:<tag>

snakemake \
    --snakefile vercye_ops/snakemake/Snakefile.azure \
    --profile   vercye_ops/snakemake/profiles/azure-batch \
    --configfile /path/to/run_config.yaml
```

## Tuning `group-components`

`profiles/azure-batch/config.yaml` ships with `region_validate=100` and
`region_pipe=25`. These are starting points - rebalance if you see either of:

- Batch task startup overhead dominating total runtime → raise the bundle
  size (fewer, longer tasks).
- Spot preemptions wasting large amounts of work → lower the bundle size
  (more, shorter tasks → less rework when a node dies).

## Things that intentionally did not change

- The `all_regions_validated` checkpoint and the `get_valid_regions()` helper
  in the Snakefile are identical between variants. `_VALID` flag files are
  real outputs persisted to Blob, so the checkpoint fires the same way.
- API-rate resource caps (`nasa_power_calls`, `era5_calls`) are still global
  and still enforced - they protect upstream services, not infrastructure.
- The on-prem `Snakefile` and `profiles/hpc/config.yaml` are untouched; HPC
  runs continue to work exactly as before.
