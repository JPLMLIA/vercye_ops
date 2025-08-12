The VeRCYe webapp is an interface to core functionality, similar as the CLI utility. It provides a webbased UI to interactively start yield studies and explore results.

A proper in depth documentation will follow soon.

### Running
To run the webapp, ensure you have set the main environmental variables (copy and fill in `vercye_ops/.env_examples` to `.env` and also the environmental variables in `vercye_ops/vercye_webapp/.env_example` to `vercye_ops/vercye_webapp/.env`.)

Install all requirements with `pip install -r vercye_ops/vercye_webapp/requirements.txt` and make sure you have followed the main setup guide for the vercye core module.

Navigate to `vercye_ops/vercye_webapp/` and run `./run.sh`. You should now be able to connect to the app under your specified socketname.

### Architecture
Backend: FastAPI API exposing an interface to setup yield studies and run them. Uses the vercye core module (`vercye_ops/vercye_ops/`) in the backend to run specific functions. Triggers snakemake pipeline executions, that are run on celery workers.

Frontend: Simple HTML+JS files served by the FastAPI API.

Celery: Runs Snakemake Pipelines for complete vercye runs. Celery workers fetch new jobs from a queue in a Redis cache.

Redis: Queing System for Pipeline executions - Limited to a single execution at a time.

Flow: User request new pipeline execution -> Frontend uses JS to send a request with specific paramters to the FastAPI backend that is exposed on a UNIX Socket. 
FastAPI backend handles specific function and prepares a job that is queued via Redis. Once a Celery worker becomes available, be will fetch the job and process it, by starting and monitoring the snakemake pipeline.