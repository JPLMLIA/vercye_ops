# Uploading Results (rclone)

At the end of a VeRCYe run the pipeline can optionally package the most
relevant study outputs (a curated subset — not every file produced by the
run) into a zip and upload it to a cloud drive via `rclone`. This is useful
when the machine running VeRCYe is not where users want to fetch results
from (e.g. a shared HPC node).

The upload step is **off by default**. It only runs when
`packaging_params.rclone_target` is set in the run config (or equivalently
when `RCLONE_TARGET` is set in your `.env` before running `vercye prep`).

## What gets uploaded

Every file under the study directory whose basename matches a glob in
`vercye_ops/reporting/output_data_patterns.txt` is bundled into a single
zip. The default pattern set contains:

- `final_report_*.pdf`, `lai_report_*.pdf`, `aggregated_met_stats_*.pdf`
- `multiyear_summary*.zip`, `interactive_map_*.zip`
- `agg_yield_estimates_*.csv`, `all_predictions_*.csv`, `referencedata_*.csv`
- `yield_mosaic_projected_*.tif`, `aggregated_LAI_MAX_*.tif`,
  `aggregated_cropmask_*.tif`
- `aggregated_region_boundaries_*.geojson`
- `config.yaml`

Edit that file to change which outputs are shipped.

## Remote folder layout

The zip is uploaded into:

```
<rclone_target>/<rclone_folder_prefix>/<study_id>_<YYYYMMDD>[_N]/
```

`_N` is a numeric suffix appended only when a folder with the same base
name already exists on the target, so re-runs on the same day never
overwrite an existing upload.

`vercye run --validate` checks that `rclone` is on PATH, that
`rclone_target` names a configured remote, and that the target is
reachable. Auth / reachability issues surface at validation time rather
than at the end of a long run.

## Setting up rclone for Google Drive

1. **Install rclone**

   ```bash
   # Debian / Ubuntu
   sudo apt-get install rclone
   # macOS
   brew install rclone
   # Or the upstream one-liner:
   curl https://rclone.org/install.sh | sudo bash
   ```

2. **Create a Google Drive remote**

   ```bash
   rclone config
   ```

   Follow the prompts:

   - `n` → new remote
   - name: `gdrive`
   - storage: `drive`
   - leave `client_id` / `client_secret` empty unless you have your own
   - scope: `1` (full access) or `2` (file scope)
   - leave root_folder_id / service account blank for interactive auth
   - `y` → auto config; a browser window opens — sign in and grant access
   - `n` → not a team drive (unless it is)
   - confirm with `y`

   On a **headless server** pick `n` for auto config and follow the
   printed instructions: run `rclone authorize "drive"` on a machine with
   a browser and paste the resulting token back.

3. **Verify**

   ```bash
   rclone listremotes          # should list 'gdrive:'
   rclone lsd gdrive:          # should list your Drive top-level folders
   ```

4. **Point VeRCYe at it.** Add to `.env` (preferred — `vercye prep` writes
   it into the run config) …

   ```dotenv
   RCLONE_TARGET="gdrive:"
   RCLONE_FOLDER_PREFIX="vercye_kenya"
   ```

   … or set it directly in `config.yaml`:

   ```yaml
   packaging_params:
     rclone_target: "gdrive:"
     rclone_folder_prefix: "vercye_kenya"
   ```

   Each run then uploads to
   `gdrive:vercye_kenya/<study_id>_<YYYYMMDD>[_N]/<zip>`.

## Other backends

`rclone_target` can be any [rclone-supported backend](https://rclone.org/overview/)
— Google Drive, S3, Azure Blob, Dropbox, SFTP, Backblaze B2, OneDrive, etc.
The setup is always the same: `rclone config` to add the remote, then set
`RCLONE_TARGET` to `<remote_name>:` (or `<remote_name>:<path>`).

## Running

No extra CLI flag is needed — the upload step runs automatically as the
last rule of `vercye run` when `rclone_target` is set. Logs are written to
`logs_package_and_upload/<study_id>_package_and_upload.log` under the
snakemake run directory, and a marker file
`.package_and_upload_<study_id>.done` is left in the study directory once
the upload has completed successfully.
