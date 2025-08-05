from pydantic import BaseModel
from pydantic import StringConstraints
from typing import Annotated, List, Tuple

StudyID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_\-]+$")]

class StudyCreateRequest(BaseModel):
    study_id: StudyID  # or just str if simpler

class LAIEntry(BaseModel):
    id: str
    lat: float
    lon: float
    dates: List[str] # Format like 2020-12-31
    status: str