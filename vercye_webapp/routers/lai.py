import os
from typing import List

from fastapi import APIRouter

from models import LAIEntry
from vercye_ops.cli import read_lai_dir_from_env

router = APIRouter(
    prefix="/lai",
    tags=["lai"]
)

lai_dir = read_lai_dir_from_env()

@router.get("/")
def get_all_lai_entries() -> List[LAIEntry]:
    if not os.path.exists(lai_dir):
        return []
    
    entries = []
    for f in os.listdir(lai_dir):
        if not os.path.isdir(os.path.join(lai_dir, f)):
            continue

        entry_id = f

        # TODO need to adapt the whole LAI creation logic to do better indexing
        lat = 49.589542
        lon = 34.551273
        status = 'completed'

        vrts_dir = os.path.join(lai_dir, f, 'merged-lai')

        # Dates will have format YYYY-MM-DD
        dates = [f.split('_')[2] for f in os.listdir(vrts_dir)]

        entries.append(LAIEntry(
            id=entry_id,
            lat=lat,
            lon=lon,
            dates=dates,
            status=status
        ))

    return entries