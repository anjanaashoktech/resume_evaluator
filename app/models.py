
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Input Model ---
class JobDescriptionInput(BaseModel):
    text: str = Field(..., min_length=30, description="The full text of the job description.")

# --- Detailed Response Model for POST /evaluate ---
class SkillsMatch(BaseModel):
    score: Optional[float] = Field(None, description="Semantic similarity score for skills (0.0 to 1.0)")
    present_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    explanation: Optional[str] = Field(None)

class ExperienceRelevance(BaseModel):
    score: Optional[float] = Field(None, description="Estimated score (placeholder)")
    explanation: Optional[str] = Field(None, description="Explanation (placeholder)")

class EvaluationDetails(BaseModel):
    skills_match: SkillsMatch = SkillsMatch()
    experience_relevance: ExperienceRelevance = ExperienceRelevance()

class DetailedEvaluationResponse(BaseModel):
    overall_score: Optional[float] = Field(None, description="Overall weighted score")
    evaluation: EvaluationDetails = EvaluationDetails()
    recommendations: List[str] = Field(default_factory=list)
    # evaluation_id: Optional[int] = None # Optionally include ID if needed

# --- Simpler Report Model for GET /report/{id} (Matches DB structure better) ---
class EvaluationReport(BaseModel):
    id: int
    job_description_summary: str
    resume_filename: Optional[str] = None
    analysis: Dict[str, Any] # Contains simplified data from DB
    recommendations: Optional[str] = None # Single string from DB
    created_at: datetime

    @classmethod
    def from_db_record(cls, record: Dict[str, Any]) -> 'EvaluationReport':
        # Transforms the flat DB record into this structure
        analysis_details = {
            "overall_similarity_score": record.get("overall_similarity_score"),
            "matched_skills_details": record.get("matched_skills"), # The stored {skill: score} dict
            "missing_skills_list": record.get("missing_skills"), # The stored [skill] list
        }
        return cls(
            id=record["id"],
            job_description_summary=(record.get("job_description") or "")[:200] + "...",
            resume_filename=record.get("resume_filename"),
            analysis=analysis_details,
            recommendations=record.get("recommendations"), # The stored text blob
            created_at=record["created_at"]
        )