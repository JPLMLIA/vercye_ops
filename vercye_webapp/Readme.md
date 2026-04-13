# VeRCYe Webapp

The VeRCYe Webapp is a simpe wrapper around vercye_ops that allows running yield studies from a web based UI.

Documentation will be properly filled in soon.

**NOTICE** this version is not intended for deployment on unsecured environments - it does not yet encompass strong security measures!

## Prerequisites

### 1. Python requirements

Ensure you have followed the setup instructions for the core library and are in the `vercye_ops/vercye_webapp` directory. Then activate the conda environment you previously created and run:

`pip install -r requirements.txt`

Take note of the conda environment's bin directory - you will need it for the `.env` file:

```bash
echo "$CONDA_PREFIX/bin"
# e.g. /home/user/miniconda3/envs/vercye/bin
```

### 2. Node.js / npm (via nvm)

The frontend is built with npm. Install it via [nvm](https://github.com/nvm-sh/nvm):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# Restart your shell or run:
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

nvm install --lts
```

Take note of your `NVM_DIR` (usually `$HOME/.nvm`) - you will need it for the `.env` file:

```bash
echo "$NVM_DIR"
```

### 3. Redis

Install Redis from source or your package manager:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install redis-server -y

# Or from source
# See https://redis.io/docs/getting-started/installation/install-redis-from-source/
```

Take note of the path to the `redis-server` binary - you will need it for the `.env` file:

```bash
which redis-server
# e.g. /usr/bin/redis-server
```

### 4. Install frontend dependancies

Navigate to the frontend folder:
`cd vercye_webapp/frontend`

and run

`npm install`.

## Setup

1. Copy the root `.env_example` to `.env` in the **project root** (one level above `vercye_webapp/`) and fill in the values:

   ```bash
   cp .env_example .env
   ```

2. Fill in the webapp-specific variables using the paths gathered above:

   | Variable | Description | Example |
   |---|---|---|
   | `PYTHON_ENV_PATH` | Conda env bin directory | `/home/user/miniconda3/envs/vercye/bin` |
   | `NVM_DIR` | NVM installation directory | `/home/user/.nvm` |
   | `REDIS_PATH` | Path to `redis-server` binary | `/usr/bin/redis-server` |
   | `SOCKET_PATH` | Unix socket for Uvicorn | `/tmp/vercye-uvicorn.sock` (default) |
   | `LOGS_PATH` | Directory for log files | `./logs` (default) |
   | `USERS` | Comma-separated users granted socket access | `user1,user2` |
   | `XDG_CACHE_HOME` | *(Optional)* Override cache dir (e.g. redirect to GPFS) | `/gpfs/data1/project/.cache` |

3. Start all services (Redis, Celery, Uvicorn) with the run script:

   ```bash
   cd vercye_webapp
   chmod +x run.sh
   ./run.sh
   ```

   The script automatically builds the frontend, starts Redis, launches Celery workers, and starts Uvicorn on a Unix socket.

4. Connect from your client machine via SSH port-forwarding over the Unix socket:

   ```bash
   ssh -L 8000:yoursocketpath yourusername@serveraddress
   ```

5. Open `http://127.0.0.1:8000` in your browser to load the dashboard.

You might want to run this all with `tmux` or `nohup` to keep the process alive.

## Architecture
Coming soon
