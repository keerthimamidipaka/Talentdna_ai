from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid

from resume_parser import extract_resume_text, guess_candidate_name
from inference_engine import generate_dna_json
from matching_engine import generate_job_dna, cosine_similarity, weighted_match_score, explain_match, rank_candidates
from evolution_predictor import predict_evolution
from llm_reasoning import refine_dna_with_llm, is_available as llm_available

app = FastAPI(title="TalentDNA AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory "database" for the demo (Module: Candidate/Job/Match tables, simplified)
DB = {"candidates": {}, "jobs": {}}


class JobDescriptionIn(BaseModel):
    title: str
    description: str


class MatchRequest(BaseModel):
    candidate_id: str
    job_id: str


def _process_resume(filename: str, file_bytes: bytes) -> dict:
    """Shared pipeline for single and bulk upload."""
    text = extract_resume_text(filename, file_bytes)
    if not text.strip():
        raise ValueError("Could not extract any text from this file.")

    name = guess_candidate_name(text)
    dna_result = generate_dna_json(text, candidate_name=name)

    llm_result = refine_dna_with_llm(text, dna_result["dna"])
    final_dna = llm_result["dna"]

    evolution = predict_evolution(final_dna)
    candidate_id = str(uuid.uuid4())[:8]
    record = {
        "id": candidate_id,
        "name": name,
        "resume_text": text,
        "dna": final_dna,
        "rule_based_dna": dna_result["dna"],
        "details": dna_result["details"],
        "evolution": evolution,
        "llm_used": llm_result["used_llm"],
        "llm_adjustments": llm_result.get("adjustments", []),
    }
    DB["candidates"][candidate_id] = record
    return record


@app.get("/api/llm-status")
async def llm_status():
    return {"available": llm_available()}


@app.post("/api/candidates/upload")
async def upload_candidate(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        return _process_resume(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/candidates/bulk-upload")
async def bulk_upload_candidates(files: List[UploadFile] = File(...)):
    results = []
    errors = []
    for f in files:
        file_bytes = await f.read()
        try:
            results.append(_process_resume(f.filename, file_bytes))
        except ValueError as e:
            errors.append({"filename": f.filename, "error": str(e)})
    return {"uploaded": results, "errors": errors}


@app.post("/api/jobs")
async def create_job(job: JobDescriptionIn):
    job_dna = generate_job_dna(job.description, title=job.title)
    job_id = str(uuid.uuid4())[:8]
    DB["jobs"][job_id] = {
        "id": job_id,
        "title": job.title,
        "description": job.description,
        "dna": job_dna["dna"],
        "required_capabilities": job_dna["required_capabilities"],
        "details": job_dna["details"],
    }
    return DB["jobs"][job_id]


@app.get("/api/candidates")
async def list_candidates():
    return list(DB["candidates"].values())


@app.get("/api/jobs")
async def list_jobs():
    return list(DB["jobs"].values())


@app.post("/api/match")
async def match(req: MatchRequest):
    candidate = DB["candidates"].get(req.candidate_id)
    job = DB["jobs"].get(req.job_id)
    if not candidate or not job:
        raise HTTPException(404, "Candidate or job not found")

    cos = cosine_similarity(candidate["dna"], job["dna"])
    score = weighted_match_score(candidate["dna"], job["dna"])
    explanation = explain_match(candidate["dna"], job["dna"], candidate["details"])

    return {
        "candidate_name": candidate["name"],
        "job_title": job["title"],
        "match_score": score,
        "cosine_similarity": round(cos, 3),
        "strengths": explanation["strengths"],
        "gaps": explanation["gaps"],
        "candidate_dna": candidate["dna"],
        "job_dna": job["dna"],
        "evolution": candidate["evolution"],
        "llm_used": candidate.get("llm_used", False),
        "llm_adjustments": candidate.get("llm_adjustments", []),
    }


@app.get("/api/jobs/{job_id}/rank")
async def rank_all_candidates(job_id: str):
    job = DB["jobs"].get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    candidates = list(DB["candidates"].values())
    ranked = rank_candidates(candidates, job)
    return ranked


# Serve the simple frontend
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
