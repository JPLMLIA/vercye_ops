The VeRCYe webapp is an interface to core functionality, wrapping the CLI utility and extending it with more convenient functions. It provides a webbased UI to interactively start yield studies and explore results.


### Setup
1. Ensure you have installed the VeRCYe core library as described in the [setup instruction](../index.md#vercye-library-setup).
2. The webapp requires you to set a number of default folders, for example for the storage of cached outputs, the path to the APSIM installation and others. For this set the environmental variables by copying `vercye_ops/.env_example` to `vercye_ops/.env` and setting the actual values. This file also contains the webapp-specific settings (redis path, log directory, socket path, user permissions, etc.).
3. Navigate to `vercye_ops/vercye_webapp/`: `cd vercye_ops/vercye_webapp`.
4. Install the additional requirements for the webapp: Ensure you have loaded your environment from step 1 and run `pip install -r requirements.txt`.
5. To queue incoming jobs and allow workers to fetch jobs independantly, `redis` is used. Install redis for your system by following the [official instructions](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/).


### Running
1. Ensure you have loaded your environment with all requirements installed as described in the section above.
2. Ensure redis is running.
3. Navigate to `vercye_ops/vercye_webapp/`
4. Run `./run.sh`. You should now be able to connect to the app under your specified socketname.

### Architecture

The webapp is split into a few components and consitutes a wrapper around the core `vercye` library.

- **Backend**: FastAPI API exposing an interface to setup yield studies and run them. Uses the vercye core module (`vercye_ops/vercye_ops/`) in the backend to run specific functions. Triggers snakemake pipeline executions, that are run on celery workers.

- **Frontend**: Simple HTML+JS files served by the FastAPI API.

- **Workers**: Individual Celery workers runs Snakemake pipelines for complete vercye runs. Celery workers fetch a job (=complete vercye pipeline to run) from a queue in a Redis cache.

- **Queing**: Redis Queing System for Pipeline executions - Limits nuber of pipeline to be runnable to a single execution at a time currently.


**Understanding the Flow**: User request new pipeline execution -> Frontend uses JS to send a request with specific paramters to the FastAPI Restful API backend that is exposed on a UNIX Socket.
The FastAPI backend then handles handles this and if a execution is requested it prepares a job that is queued via Redis. Once a Celery worker becomes available, the worker will fetch the job and process it, by starting and monitoring the snakemake pipeline.


### Running on VM startup (systemd)

On a server you typically want the webapp to come up automatically on boot and to be restarted if it ever crashes. The `run.sh` script is the entrypoint - it builds the frontend, starts Redis, the two Celery worker queues, and Uvicorn, and installs traps that clean everything up on exit. That makes it a clean fit for a single systemd service.

#### 1. Install the unit file

Create `/etc/systemd/system/vercye-webapp.service` (adjust `User`, `Group`, and the paths if your install lives elsewhere):

```ini
[Unit]
Description=VeRCYe webapp (FastAPI + Celery workers + Redis)
Documentation=file:///home/raapidadmin/vercye/vercye_ops/docs/docs/Vercye/webapp.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=raapidadmin
Group=raapidadmin
WorkingDirectory=/home/raapidadmin/vercye/vercye_ops/vercye_webapp
ExecStart=/usr/bin/env bash /home/raapidadmin/vercye/vercye_ops/vercye_webapp/run.sh
Restart=always
RestartSec=10
TimeoutStopSec=30
KillMode=mixed
UMask=0002
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 2. Enable and start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now vercye-webapp.service
```

`enable --now` both turns on auto-start at boot and starts the service immediately.

#### 4. Verify

```bash
systemctl status vercye-webapp.service       # should be "active (running)"
journalctl -u vercye-webapp -f               # live logs (Ctrl+C to detach)
ls -l /tmp/vercye-uvicorn.sock               # socket should exist with the perms run.sh sets
```

The first start is slower than subsequent ones because `run.sh` runs `npm ci && npm run build` for the frontend on every launch.

#### 5. Day-to-day operation

```bash
sudo systemctl restart vercye-webapp         # apply code/config changes
sudo systemctl stop vercye-webapp            # stop (won't auto-restart until started)
sudo systemctl disable vercye-webapp         # remove from boot, keep current state
journalctl -u vercye-webapp -n 200 --no-pager
```

The detailed component logs (Celery, Redis, Uvicorn) still go to `LOGS_PATH` from `.env` with a per-start timestamp suffix; systemd's journal only captures `run.sh`'s own stdout/stderr.

### Storage layout (keep studies off the OS disk)

Studies grow large quickly (per-cell APSIM runs, snakemake intermediates, output rasters). On a typical install the OS disk (`/`) is small and a separate, larger volume is mounted at `/data`. To keep new studies from filling the OS disk:

1. Pick the canonical location on `/data` and create it once:

    ```bash
    sudo mkdir -p /data/vercye/studies
    sudo chown raapidadmin:raapidadmin /data/vercye/studies
    chmod 2775 /data/vercye/studies      # 2 = setgid, see "Shared multi-user access"
    ```

2. In `vercye_ops/.env`, point `STUDY_DIR` at it:

    ```bash
    STUDY_DIR=/data/vercye/studies
    ```

   This is the simplest setup: the webapp writes directly to `/data` and there is no symlink involved.

#### Migration from an existing in-`$HOME` install

If `STUDY_DIR` historically pointed somewhere in the install user's home (e.g. `/home/raapidadmin/vercye/data/studies`) and you don't want to change the env var (or scripts that hard-code the path), replace the directory itself with a symlink so the path resolves to `/data`:

```bash
sudo systemctl stop vercye-webapp        # avoid races with the live celery workers

# Move existing studies onto /data (skip if there are none)
mv /home/raapidadmin/vercye/data/studies/* /data/vercye/studies/   2>/dev/null || true

# Replace the dir with a symlink
rmdir /home/raapidadmin/vercye/data/studies
ln -s /data/vercye/studies /home/raapidadmin/vercye/data/studies

sudo systemctl start vercye-webapp
```

After the swap, `os.listdir(STUDY_DIR)` and every other path lookup resolves to `/data/vercye/studies/...` transparently - no code or env change.

### Shared multi-user access to `/data`

When more than one human user works on the VM (e.g. `raapidadmin` runs the service, but a colleague also needs to read and edit study inputs/outputs), set up shared group access on `/data/vercye` so both users can collaborate without `sudo` and without files ending up owned by whoever happened to create them.

The webapp install already runs as `raapidadmin:raapidadmin`, so the simplest setup is to use `raapidadmin` as the shared group. Replace `<colleague>` with each additional username:

```bash
# 1. Add the colleague to the install group.
sudo usermod -aG raapidadmin <colleague>
# The colleague must log out and back in for the new group to take effect.

# 2. Make every existing file/dir under /data/vercye group-readable+writable.
sudo chmod -R g+rwX /data/vercye         # capital X = exec only for dirs, not files

# 3. Set the setgid bit on every directory so newly-created files inherit
#    the raapidadmin group instead of the creator's primary group.
sudo find /data/vercye -xdev -type d -exec chmod g+s {} +

# 4. Make sure both users create files with group-write by default.
#    For interactive shells: add `umask 002` to ~/.bashrc for each user.
#    For the webapp service: the unit above already sets `UMask=0002`.
```

What each piece does:

- **Group membership** gives the colleague the same `g` permission bit as `raapidadmin` on every file in `/data/vercye`.
- **`chmod -R g+rwX`** flips on group read/write everywhere, and group-execute on directories only (so you can `cd` into them) - the capital `X` is the safe form that doesn't accidentally mark plain files as executable.
- **Setgid on directories (`g+s`)** makes new files and subdirectories inherit the parent dir's group (`raapidadmin`) instead of the creator's primary group. Without this, files a colleague creates would land as `<colleague>:<colleague>` and `raapidadmin` would lose group access.
- **`umask 002` / `UMask=0002`** controls the *mode* of new files. With the default `umask 022`, files are created `rw-r--r--` - group can read but not write. With `002` they're `rw-rw-r--`, which is what we want for shared editing. The systemd unit's `UMask=0002` covers everything the running webapp creates; per-user shells need their own `umask 002` (typically in `~/.bashrc`).

Verify the result:

```bash
ls -ld /data/vercye/studies
# drwxrwsr-x   ...   raapidadmin raapidadmin   /data/vercye/studies
#    ^^^^      group has rwx and the trailing 's' shows setgid is set
```

`/data/vercye/keys/` contains things like the Google Earth Engine key - by giving the group read access you are explicitly making those keys readable to everyone in the `raapidadmin` group. If a key should *not* be shared with a colleague, store it elsewhere (e.g. each user's home) and set its path in their own environment, rather than hand-tightening permissions on individual files inside `/data/vercye` (the next `chmod -R g+rwX` for a new colleague would undo it).