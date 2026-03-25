from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, StringConstraints
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

StudyID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9_\-]+$")]


class StudyCreateRequest(BaseModel):
    study_id: StudyID


class DuplicateStudyRequest(BaseModel):
    new_study_id: StudyID


class LAIEntry(BaseModel):
    id: str
    lat: float
    lng: float
    dates: List[str]  # Format like 2020-12-31
    status: str
    resolution: int  # in meters
    geometry: dict

    @classmethod
    def from_shapely(cls, **kwargs):
        if "geometry" in kwargs and isinstance(kwargs["geometry"], BaseGeometry):
            kwargs["geometry"] = mapping(kwargs["geometry"])
        return cls(**kwargs)


class FilterConfig(BaseModel):
    column: str
    allow: List[str]


class RegionExtraction(BaseModel):
    admin_name_column: str
    target_projection: str
    filter: Optional[FilterConfig] = None


class WindowConfig(BaseModel):
    year: str
    timepoint: str
    sim_start_date: str
    sim_end_date: str
    met_start_date: str
    met_end_date: str
    lai_start_date: str
    lai_end_date: str


MappingState = Dict[str, Union[str, List[str]]]


class SetupSubmissionsRequest(BaseModel):
    region_extraction: RegionExtraction
    apsim_column: str
    apsim_mapping: MappingState
    reference_mapping: MappingState
    reference_years_mapping: MappingState
    years: List[str]
    timepoints: List[str]
    simulation_windows: List[WindowConfig]


class RegionExtractionResponse(BaseModel):
    adminNameColumn: str
    targetProjection: str
    filter: Optional[FilterConfig] = None


class WindowNoId(BaseModel):
    year: str
    timepoint: str
    sim_start_date: str
    sim_end_date: str
    met_start_date: str
    met_end_date: str
    lai_start_date: str
    lai_end_date: str


class Feature(BaseModel):
    geometry: Dict[str, Any]
    properties: Dict[str, Any]


class ShapefileData(BaseModel):
    type: str
    features: List[Feature]


class SetupConfigTemplate(BaseModel):
    regionExtraction: RegionExtractionResponse
    apsimColumn: str
    apsimMapping: Dict[str, List[str]]
    apsimFiles: List[str]
    referenceMapping: Dict[str, str]
    referenceYearsMapping: Dict[str, str]
    referenceFiles: List[str]
    years: List[str]
    timepoints: List[str]
    simulationWindows: List[WindowNoId]
    shapefileData: ShapefileData
    shapefileName: str


class RunConfigFormParams(BaseModel):
    laiId: str
    laiResolution: int
    cropmasks: Dict[str, str]


class LAIConfigRunParams(BaseModel):
    lai_source_id: str
    lai_source_resolution: int


class LAIGenerationDateRange(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str


class LAIGenerationConfig(BaseModel):
    resolution: int
    keep_imagery: bool
    name: str
    date_ranges: List[LAIGenerationDateRange]
    imagery_src: Literal["MPC", "ES_S2C1"]
    chunk_days: int


class AddLaiDatesConfig(BaseModel):
    resolution: int
    keep_imagery: bool
    name: str
    date_ranges: List[LAIGenerationDateRange]
    chunk_days: int
