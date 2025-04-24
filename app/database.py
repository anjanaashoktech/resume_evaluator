# app/database.py
from supabase import create_client, Client
from app.config import settings # Import our settings
from typing import Optional, Dict, Any, List # For type hinting


supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    global supabase_client
    if supabase_client is None:
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("Supabase URL/Key missing. Check .env file.")
        try:
            print("Initializing Supabase client...")
            supabase_client = create_client(settings.supabase_url, settings.supabase_key)
            print("Supabase client initialized.")
        except Exception as e:
            print(f"Error initializing Supabase client: {e}")
            raise
    return supabase_client

# --- Function to insert data (matches the table schema) ---
async def insert_evaluation(
    job_description: str,
    resume_filename: Optional[str],
    extracted_resume_text: Optional[str],
    overall_similarity_score: Optional[float],
    matched_skills: Optional[Dict[str, float]], 
    missing_skills: Optional[List[str]],      # 
    recommendations: Optional[str]
) -> Optional[int]:
    """Inserts core evaluation results into the Supabase 'evaluations' table."""
    try:
        client = get_supabase_client()
        data_to_insert = {
            "job_description": job_description,
            "resume_filename": resume_filename,
            "extracted_resume_text": extracted_resume_text, 
            "overall_similarity_score": overall_similarity_score,
            "matched_skills": matched_skills or {},
            "missing_skills": missing_skills or [], 
            "recommendations": recommendations,
        }
        response = client.table("evaluations").insert(data_to_insert).execute()

        if response.data and len(response.data) > 0:
            inserted_id = response.data[0].get('id')
            if inserted_id:
                print(f"DB Insert successful. ID: {inserted_id}")
                return inserted_id
            else:
                print("DB Insert Warning: No ID returned in response.", response.data)
                return None
        else:
            error_msg = response.error.message if response.error else "Unknown DB insert error."
            print(f"DB Insert Error: {error_msg}")
            return None
    except Exception as e:
        print(f"DB Insert Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


async def get_evaluation_by_id(evaluation_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a specific evaluation record by its ID."""
    try:
        client = get_supabase_client()
        response = client.table("evaluations").select("*").eq("id", evaluation_id).execute()

        if response.data and len(response.data) > 0:
            return response.data[0] # Return the found record
        elif response.error:
             print(f"DB Fetch Error: {response.error.message}")
             return None
        else:
             return None # Not found
    except Exception as e:
        print(f"DB Fetch Exception: {e}")
        return None