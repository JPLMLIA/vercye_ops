import os
import shutil

import numpy as np
import rasterio as rio
from fastapi import APIRouter, File, HTTPException, UploadFile

from vercye_ops.utils.env_utils import read_cropmasks_dir_from_env

router = APIRouter(
    prefix="/cropmasks",
    tags=["cropmasks"],
)

cropmasks_dir = read_cropmasks_dir_from_env()


@router.get("")
def get_all_cropmasks():
    if not os.path.exists(cropmasks_dir):
        return []

    cropmasks = [{"id": f} for f in os.listdir(cropmasks_dir) if not os.path.isdir(f)]

    return cropmasks


@router.post("/{cropmask_id}")
def upload_cropmask(cropmask_id: str, cropmask_file: UploadFile = File(...)):
    storage_path = os.path.join(cropmasks_dir, f"{cropmask_id}.tif")

    if os.path.exists(storage_path):
        raise HTTPException(status_code=409, detail="A cropmask with this name already exists")

    # Save temporarily for validation
    tmp_path = storage_path + ".tmp"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(cropmask_file.file, f)

    # Validate raster content to be binary (0/1)
    try:
        with rio.open(tmp_path) as src:
            arr = src.read(1, masked=True)  # read first band
            unique_vals = np.unique(arr.compressed())  # ignore nodata

            if not np.all(np.isin(unique_vals, [0, 1])):
                os.remove(tmp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cropmask values found: {unique_vals.tolist()}. Only 0 and 1 allowed.",
                )

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF: {str(e)}")

    os.rename(tmp_path, storage_path)
    return {"message": f"Cropmask {cropmask_id} uploaded successfully"}
