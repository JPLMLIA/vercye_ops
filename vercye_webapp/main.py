import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from routers import cropmasks, lai, studies
from utils import clean_running_tasks

##########################################################################################################
# TODO
# show data availability percentage coverage per day of the ROI
# show colored snakemake logs
##########################################################################################################


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for startup and shutdown ensuring all running operations are terminated and the status is updated"""
    clean_running_tasks()
    yield
    clean_running_tasks()


app = FastAPI(lifespan=lifespan)

# Keep your API routers first
app.include_router(lai.router, prefix="/api")
app.include_router(studies.router, prefix="/api")
app.include_router(cropmasks.router, prefix="/api")

# Serve frontend
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")


# Catch-all for React Router to serve frontend for all non /api routes
@app.get("/{full_path:path}")
def serve_react_app(full_path: str):
    index_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(index_path)
