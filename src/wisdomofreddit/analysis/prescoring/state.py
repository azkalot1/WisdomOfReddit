from pydantic import BaseModel, Field

class RelevanceResults(BaseModel):
    is_relevant: bool = Field(description="Whether the discussion contains valid discussions of investment opportunities")
    reasoning: str = Field(description="Explanation of the relevance decision")
    confidence: int = Field(description="Confidence score for the relevance decision")




