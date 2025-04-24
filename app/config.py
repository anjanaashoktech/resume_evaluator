# app/config.py
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class Settings(BaseSettings):
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    # Add defaults for other settings we'll need later
    sentence_transformer_model: str = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    class Config:
        env_file = '.env'
        extra = 'ignore'

settings = Settings()


if not settings.supabase_url or not settings.supabase_key:
    print("WARNING: Supabase credentials not found in .env file!")
