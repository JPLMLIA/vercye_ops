# VeRCYe Webapp

The VeRCYe Webapp is a simpe wrapper around vercye_ops that allows running yield studies from a web based UI.

Documentation will be properly filled in soon.

**NOTICE** this version is not intended for deployment on unsecured environments - it doess not yet encompass strong security measures!

## Setup
1. Run `pip install -r requirements.txt`
2. Copy `.env_example`to `.env` and fill in the values according to your system
2. Set env variables in the vercye_ops root as described in vercye_ops docs
3. Ensure you have redis installed or available.
4. Use the run script to run the redis broker, celery worker and fastapi/uvicorn server and to set user permissions:
```bash
chmod +x run.sh
./run.sh
```

You might want to run this with `tmux` or `nohup` to keep the process alive.

## Architecture
Coming soon