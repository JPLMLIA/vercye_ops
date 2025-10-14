The VeRCYe webapp is an interface to core functionality, wrapping the CLI utility and extending it with more convenient functions. It provides a webbased UI to interactively start yield studies and explore results.


### Setup
1. Ensure you have installed the VeRCYe core library as described in the [setup instruction](../index.md#vercye-library-setup).
2. The webapp requires you to set a number of default folders, for example for the storage of cached outputs, the path to the APSIM installation and others. For this set the environmental variables by copying  `vercye_ops/.env_examples` to `vercye_ops/.env` and setting the actual values.
3. Navigate to `vercye_ops/vercye_webapp/`: `cd vercye_ops/vercye_webapp`.
4. Install the additional requirements for the webapp: Ensure you have loaded your environment from step 1 and run `pip install -r requirements.txt`.
5. To queue incoming jobs and allow workers to fetch jobs independantly, `redis` is used. Install redis for your system by following the [official instructions](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/).
4. You will now have to specify a few more environmental variables for the webapp. For this copy `vercye_webapp/.env_example` to `vercye_webapp/.env` and set the values.


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
