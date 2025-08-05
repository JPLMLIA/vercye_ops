from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(
    prefix="",
    tags=["frontend"]
)

@router.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open("assets/ui/study_dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())
    
@router.get("/lai", response_class=HTMLResponse)
def serve_dashboard():
    with open("assets/ui/lai_dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())
