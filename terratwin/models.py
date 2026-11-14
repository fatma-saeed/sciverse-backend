from pydantic import BaseModel, ConfigDict, Field

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    targetId: str
    objective: str
    environments: list[str] = Field(default_factory=list)
    search: str | None = None
    minScore: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    includeExcluded: bool = False

class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    targetId: str
    objective: str
    candidateIds: list[str] = Field(min_length=1)
    includeExcluded: bool = False
