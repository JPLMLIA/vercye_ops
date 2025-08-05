from fastapi import FastAPI

from routers import frontend, lai, studies


app = FastAPI()

##########################################################################################################
# TODO
# On startup set all tasks to failed that have status running, because it means sever crashed/was stopped.
# If a job is still running on startup, kill?
# Add possibility to copy a study to adapt params
# show data availability as a plot for lai data to see actual dates where avail in a region.
##########################################################################################################

# Include routes defined in seperate modules
app.include_router(frontend.router)
app.include_router(lai.router, prefix='/api')
app.include_router(studies.router, prefix='/api')