from pydantic import BaseModel
from pydantic import StringConstraints
from typing import Annotated

StudyID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_\-]+$")]

class StudyCreateRequest(BaseModel):
    study_id: StudyID  # or just str if simpler
