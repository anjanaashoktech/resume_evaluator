# FastAPI Application for Resume Evaluation
from fastapi import (
    FastAPI, File, UploadFile, HTTPException, Body, status, Depends, Form
)
from typing import List
import traceback # For detailed error logging
# --- Import your Models ---
from app.models import (
    JobDescriptionInput, DetailedEvaluationResponse, SkillsMatch,
    ExperienceRelevance, EvaluationDetails, EvaluationReport
)
# --- Import your Database Functions ---
from app.database import insert_evaluation, get_supabase_client, get_evaluation_by_id
# --- Import your Evaluation Functions ---
from app.evaluator import (
    extract_text_from_pdf, calculate_semantic_skill_match,
    analyze_experience, generate_skills_explanation,
    generate_recommendations_list,
    get_sbert_model    
)
from app.config import settings # Import settings for threshold access
from sentence_transformers import SentenceTransformer




app = FastAPI()

@app.post("/evaluate/",
          response_model=DetailedEvaluationResponse,
          summary="Evaluate Resume against Job Description",
          status_code=status.HTTP_200_OK)
async def evaluate_resume_detailed(
    job_description_text: str = Form(..., description="Job description text."),
    resume_file: UploadFile = File(..., description="Candidate's resume in PDF format."),
    model: SentenceTransformer = Depends(get_sbert_model)
):
    job_description = JobDescriptionInput(text=job_description_text)
    """
    Upload a PDF resume and provide job description text.
    The API extracts text, performs semantic analysis for skills,
    estimates experience relevance, generates recommendations,
    and returns a detailed JSON evaluation. Core results are saved to the database.
    """
    print(f"[ENDPOINT /evaluate/] Received request for resume: {resume_file.filename}")
    try:
        if not resume_file.filename or not resume_file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file type. Please upload a PDF file.",
            )
        try:
            resume_content = await resume_file.read()
            if not resume_content:
                 raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Resume file is empty.")
        except Exception as e:
            print(f"[ENDPOINT /evaluate/ ERROR] Reading file {resume_file.filename}: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not read uploaded file: {e}")
        finally:
             await resume_file.close() 
        
        resume_text = extract_text_from_pdf(resume_content)
        

        skill_score, present_skills, missing_skills, matched_scores = calculate_semantic_skill_match(
            job_description.text, resume_text, model, settings.similarity_threshold
        )
        skills_explanation = generate_skills_explanation(present_skills, missing_skills, skill_score)

        # print("[ENDPOINT /evaluate/] Performing placeholder experience analysis...") # Debug
        experience_score, experience_explanation = analyze_experience(
            job_description.text, resume_text
        )

        # Calculate Overall Score (Example: Weighted Average)
        overall_score = round(((skill_score or 0) * 0.7) + ((experience_score or 0) * 0.3), 3)

  
        # Generate Recommendations List 
        recommendations_list = generate_recommendations_list(
            present_skills, missing_skills, experience_explanation, skill_score, experience_score
        )

        # Assemble the Detailed Response Object
        response_object = DetailedEvaluationResponse(
            overall_score=overall_score,
            evaluation=EvaluationDetails(
                skills_match=SkillsMatch(
                    score=skill_score, present_skills=present_skills,
                    missing_skills=missing_skills, explanation=skills_explanation
                ),
                experience_relevance=ExperienceRelevance(
                    score=experience_score, explanation=experience_explanation
                )
            ),
            recommendations=recommendations_list
        )

        #Persist Core Results to Database (Asynchronously)
        print("[ENDPOINT /evaluate/] Attempting to save core results to DB...")
        recommendations_string = "\n".join(recommendations_list) # Convert list back to string for DB
        # Decide whether to store full resume text in DB (can make it large)
        text_to_store = resume_text if len(resume_text or "") < 15000 else (resume_text[:15000] + "... (truncated)") # Example truncation
        # text_to_store = None # Option to not store it

        evaluation_id = await insert_evaluation(
            job_description=job_description.text,
            resume_filename=resume_file.filename,
            extracted_resume_text=text_to_store,
            overall_similarity_score=skill_score, # Store semantic skill score
            matched_skills=matched_scores,       # Store {skill: score} dict
            missing_skills=missing_skills,       # Store [skill] list
            recommendations=recommendations_string, # Store recommendations text blob
        )

        if evaluation_id is None:
            # Log warning but still return the analysis to the user
            print(f"[ENDPOINT /evaluate/ WARNING] DB insert failed for {resume_file.filename}. Analysis results still returned.")
        else:
            print(f"[ENDPOINT /evaluate/] DB insert successful. Evaluation ID: {evaluation_id}")
            # Optionally add evaluation_id to the response if needed:
            # response_object.evaluation_id = evaluation_id

        # 8. Return the detailed analysis
        print(f"[ENDPOINT /evaluate/] Successfully processed request for {resume_file.filename}.")
        return response_object

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except RuntimeError as e:
         # Catch the specific "Model not loaded" error if it somehow happens here
         print(f"[ENDPOINT /evaluate/ FATAL RUNTIME ERROR]: {e}")
         traceback.print_exc()
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail=f"Internal server error: {e}. Model might not be loaded."
         )
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"[ENDPOINT /evaluate/ UNEXPECTED ERROR] for {resume_file.filename}: {e}")
        traceback.print_exc() # Log the full traceback for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during evaluation: {e}"
        )


@app.get("/report/{evaluation_id}",
         response_model=EvaluationReport, # Uses the simpler model matching DB structure
         summary="Retrieve Stored Evaluation Report",
         status_code=status.HTTP_200_OK)
async def get_evaluation_report(evaluation_id: int):
    """
    Retrieves a previously stored evaluation report summary by its unique ID.
    The structure reflects the core data saved in the database.
    """
    print(f"[ENDPOINT /report] Request for evaluation ID: {evaluation_id}")
    report_data = await get_evaluation_by_id(evaluation_id)

    if report_data is None:
        print(f"[ENDPOINT /report] Report {evaluation_id} not found in DB.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evaluation report with ID {evaluation_id} not found.",
        )
    try:
        # Use the Pydantic model's helper to format the DB dictionary
        report = EvaluationReport.from_db_record(report_data)
        print(f"[ENDPOINT /report] Successfully retrieved and formatted report {evaluation_id}.")
        return report
    except Exception as e:
         # Handle potential errors during data conversion/validation by the Pydantic model
         print(f"[ENDPOINT /report ERROR] Formatting report data for ID {evaluation_id}: {e}")
         traceback.print_exc()
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail=f"Failed to process report data from database. Error: {e}",
         )


@app.get("/", include_in_schema=False) # Hide from OpenAPI docs if desired
async def read_root():
    """Provides a simple welcome message."""
    return {"message": "Welcome to the Resume Evaluator API! Visit /docs for API documentation."}


@app.get("/health", tags=["Health"], summary="Check API Health Status")
async def health_check():
    """Checks the status of the API, database connection, and ML model."""
    db_status = "unknown"
    try:
        get_supabase_client() # Check if client can be initialized/retrieved
        # Could add a simple DB query here for a deeper check if needed
        db_status = "connected"
    except Exception as e:
        print(f"[HEALTH CHECK WARN] DB connection check failed: {e}")
        db_status = "disconnected"

    model_status = "unknown"
    try:
        if get_sbert_model() is not None:
            model_status = "loaded"
        else:
             model_status = "not loaded" # Should not happen if lifespan worked
    except RuntimeError:
        model_status = "not loaded" # Explicitly catch the specific error
    except Exception as e:
         print(f"[HEALTH CHECK WARN] Model status check failed: {e}")
         model_status = "error"

    return {"api_status": "ok", "database_status": db_status, "model_status": model_status}