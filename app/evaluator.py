
import io
import re
from PyPDF2 import PdfReader
from typing import Optional, Tuple, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer, util
import numpy as np
from keybert import KeyBERT
from datetime import datetime
from dateutil.relativedelta import relativedelta
import dateparser
from typing import Tuple
from app.config import settings # Use settings for model name, threshold
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



sentence_transformer_model = "all-MiniLM-L6-v2"
model = SentenceTransformer(sentence_transformer_model)

def get_sbert_model() -> SentenceTransformer:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def extract_text_from_pdf(pdf_content: bytes) -> Optional[str]:
    try:
        pdf_stream = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_stream)
        all_text = []
        # Loop through each page and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
        
        # Join all text and return it
        resume_text = "\n".join(all_text)
        return resume_text
    
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

kw_model = KeyBERT()
COMMON_SKILLS = [
    "aws", "azure", "gcp", "machine learning", "deep learning", "nlp", 
    "python", "java", "pytorch", "tensorflow", "kubernetes", "docker", 
    "sql", "huggingface", "llm", "transformers"
]
def simple_keyword_extraction(text: str, top_n: int = 30, num_keywords: int = 30) -> List[str]:
    bert_keywords = [kw for kw, _ in kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n
    )]

    # Check for known skills via substring match
    found_skills = [skill for skill in COMMON_SKILLS if skill.lower() in text.lower()]

    # Combine and deduplicate
    all_keywords = list(set(bert_keywords + found_skills))
    return all_keywords
'''
   # This function uses regex to extract keywords from the text
    potential_keywords = re.split(r'[\n,;•●*➣➢-]+', text.lower())
    keywords = [kw.strip().lower() for kw in potential_keywords if kw and len(kw.strip()) > 3]
    seen = set()
    unique_keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]
    return unique_keywords[:num_keywords]
'''
def calculate_semantic_skill_match(
    job_description: str, resume_text: str, model: SentenceTransformer, threshold: float
) -> Tuple[Optional[float], List[str], List[str], Dict[str, float]]:
    # (This function calculates score, present skills, missing skills, and the dict of matched scores)
    # ... (Keep the implementation from the previous detailed answer - Step 2) ...
    
    jd_keywords = simple_keyword_extraction(job_description)
    resume_chunks = [chunk.strip() for chunk in re.split(r'\n\s*\n|[\n.]+', resume_text) if chunk and len(chunk.strip()) > 20]
    
    jd_embeddings = model.encode(jd_keywords, convert_to_tensor=True, show_progress_bar=False)
    resume_embeddings = model.encode(resume_chunks, convert_to_tensor=True, show_progress_bar=False)
    
    cosine_scores = util.cos_sim(jd_embeddings, resume_embeddings)
    max_scores_tensor, _ = cosine_scores.max(dim=1)
    max_scores = max_scores_tensor.cpu().numpy()
    present_skills, missing_skills, scores_above_threshold, matched_skill_scores = [], [], [], {}
    for i, keyword in enumerate(jd_keywords):
        score = float(max_scores[i])
        if score >= threshold:
            present_skills.append(keyword)
            scores_above_threshold.append(score)
            matched_skill_scores[keyword] = round(score, 3)
        else:
            missing_skills.append(keyword)
    skill_score = np.mean(scores_above_threshold) if scores_above_threshold else 0.0
    print(f"Skill Match: Score={skill_score:.3f}, Present={len(present_skills)}, Missing={len(missing_skills)}")
    return round(float(skill_score), 3), present_skills, missing_skills, matched_skill_scores

def extract_work_experience_section(text: str) -> str:
    section_headers = [
        "work experience", "professional experience", "experience",
        "employment history", "career summary"
    ]
    pattern = r'(?i)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|' \
          r'Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})' \
          r'\s*(?:to|–|-)\s*(Present|present|' \
          r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|' \
          r'Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?))?\s*(\d{4})?'

    match = re.search(pattern, text, re.DOTALL)
    return match.group(0).strip() if match else ""


def extract_date_ranges(text: str):
    pattern = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{4})\s*(?:to|\-|–)\s*(Present|present|Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)?\s*(\d{4})?'
    matches = re.findall(pattern, text)

    date_ranges = []
    for smonth, syear, emonth, eyear in matches:
        try:
            start = dateparser.parse(f"{smonth} {syear}")
            end = dateparser.parse(f"{emonth} {eyear}") if emonth.lower() != "present" else datetime.today()
            if start and end and start < end:
                date_ranges.append((start, end))
        except:
            continue
    return date_ranges

def calculate_total_experience(date_ranges):
    total_months = 0
    for start, end in date_ranges:
        delta = relativedelta(end, start)
        total_months += delta.years * 12 + delta.months
    return round(total_months / 12, 2)

def analyze_experience(job_description: str, resume_text: str) -> Tuple[float, str]:
    # 1. Extract required years from JD
    required_match = re.search(r"(\d+)\+?\s*years?.*(machine learning|ml)", job_description, re.IGNORECASE)
    required_years = int(required_match.group(1)) if required_match else 0

    # 2. Extract only work experience section
    work_section = extract_work_experience_section(resume_text)

    if not work_section:
        return 0.40, "Could not locate work experience section in resume."

    # 3. Extract date ranges and calculate experience
    date_ranges = extract_date_ranges(work_section)
    total_years = calculate_total_experience(date_ranges)

    # 4. Score based on comparison
    if required_years == 0:
        score = 0.70
        explanation = f"Found {total_years} years (approx.) experience based on dates."
    elif total_years >= required_years:
        score = 0.90
        explanation = f"Matches {required_years}+ years requirement. Found ~{total_years} years."
    elif total_years > 0:
        score = 0.60
        explanation = f"Found ~{total_years} years, less than required {required_years}+."
    else:
        score = 0.40
        explanation = f"Could not extract clear experience from work experience section."

    return score, explanation


'''
def analyze_experience(job_description: str, resume_text: str) -> Tuple[float, str]:
    required_years_match = re.search(r"(\d+)\+?\s*years?.*(machine learning|ml)", job_description, re.IGNORECASE)
    required_years = int(required_years_match.group(1)) if required_years_match else 0
    resume_years_match = re.search(r"(\d+)\s*years?.*(machine learning|ml)", resume_text, re.IGNORECASE)
    resume_years = int(resume_years_match.group(1)) if resume_years_match else 0
    score = 0.70; explanation = "Experience analysis placeholder."
    if required_years > 0:
        if resume_years >= required_years: score, explanation = 0.90, f"Matches {required_years}+ years ML (simple search)."
        elif resume_years > 0: score, explanation = 0.60, f"Found {resume_years} years ML, less than {required_years}+ required (simple search)."
        else: score, explanation = 0.40, f"Cannot confirm {required_years}+ years ML (simple search)."
    return score, explanation
'''

def generate_skills_explanation(present_skills: List[str], missing_skills: List[str], skill_score: Optional[float]) -> str:
    present_count, missing_count = len(present_skills), len(missing_skills); total_keywords = present_count + missing_count
    if total_keywords == 0: return "No keywords extracted from JD."
    score_text = f"{skill_score*100:.0f}%" if skill_score is not None else "N/A"
    if present_count == total_keywords: explanation = f"Excellent skill alignment ({score_text}). All {total_keywords} keywords found."
    elif present_count > missing_count: explanation = f"Strong skill alignment ({score_text}). Found {present_count}/{total_keywords}. Missing: {', '.join(missing_skills[:2])}{'...' if missing_count > 2 else ''}."
    elif present_count > 0: explanation = f"Partial skill alignment ({score_text}). Found {present_count}/{total_keywords}. Missing {missing_count} like: {', '.join(missing_skills[:2])}{'...' if missing_count > 2 else ''}."
    else: explanation = f"Low skill alignment ({score_text}). None of {total_keywords} keywords found. Missing: {', '.join(missing_skills[:2])}{'...' if missing_count > 2 else ''}."
    return explanation

def generate_recommendations_list(
    present_skills: List[str], missing_skills: List[str], experience_explanation: str, skill_score: Optional[float], experience_score: Optional[float]
) -> List[str]:
    recs = []
    overall_impression = "Potential candidate."
    if skill_score is not None and experience_score is not None: avg_score = (skill_score + experience_score) / 2
    else: avg_score = 0.5 # Default if scores missing
    if avg_score >= 0.8: overall_impression = "Strong profile."
    elif avg_score >= 0.6: overall_impression = "Reasonable profile."
    else: overall_impression = "Potential gaps noted."
    recs.append(f"Summary: {overall_impression}")
    if missing_skills: recs.append(f"Investigate missing skills: {', '.join(missing_skills[:3])}{'...' if len(missing_skills) > 3 else ''}.")
    if experience_score is not None and experience_score < 0.7 and "short" in experience_explanation.lower(): recs.append("Verify years of experience.")
    if skill_score is not None and skill_score > 0.8 and present_skills: top_skill = max(present_skills, key=len); recs.append(f"Verify depth in key areas like '{top_skill}'.")
    if not recs or len(recs) == 1: recs.append("Standard interview process recommended.")
    return recs