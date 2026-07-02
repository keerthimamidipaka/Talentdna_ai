# TalentDNA AI — Working Demo

A capability-based candidate matching platform, built from the project brief.
Backend and frontend both implemented and tested end-to-end (results below).

## What's implemented

| Brief Module | Status | Where |
|---|---|---|
| 1. Resume Intelligence Engine | ✅ | `backend/resume_parser.py` (PDF, DOCX, TXT) |
| 2. Job Description Intelligence | ✅ | `backend/matching_engine.py::generate_job_dna` |
| 3. Capability Inference Engine | ✅ | `backend/inference_engine.py` (rule-based, evidence-weighted) |
| 4. Talent DNA Generator | ✅ | `backend/inference_engine.py::generate_dna_json` |
| 5. Embedding Generator | ⚙️ simplified | DNA is already a numeric vector; used directly for cosine similarity instead of a separate embedding model, to keep the demo dependency-free |
| 6. Semantic Matching Engine | ✅ | `backend/matching_engine.py` (cosine similarity + weighted score) |
| 7. Explainable AI | ✅ | `explain_match()` — evidence-backed strengths & gaps |
| 8. DNA Evolution Predictor | ✅ | `backend/evolution_predictor.py` |
| 9. Recruiter Dashboard | ✅ simplified | `frontend/index.html` — single-page dashboard with radar chart, rankings, explainability panel, bulk upload |
| LLM reasoning pass | ✅ optional | `backend/llm_reasoning.py` — Claude reviews resume + rule-based scores, suggests capped adjustments (±15 pts) with cited evidence. Skips cleanly if no API key. |
| Bulk resume upload | ✅ | `POST /api/candidates/bulk-upload` — upload many resumes at once, then rank them all against a job |

## Two-layer capability inference

1. **Rule-based (always runs)** — transparent keyword/phrase evidence + impact bonuses
   ("led 4 developers", "AWS Certified"). No API key, fully offline, deterministic.
2. **LLM reasoning (optional, layered on top)** — if you set `ANTHROPIC_API_KEY`, Claude
   reviews the resume text plus the rule-based scores and looks for evidence the keyword
   list would miss (e.g. "orchestrated our container fleet across three regions" implies
   devops even without the word "Kubernetes"). Adjustments are capped at ±15 points per
   dimension and each comes with a one-line justification — it can only *refine* the
   rule-based baseline, never override it wholesale. If the API call fails for any reason
   (no key, bad key, network issue, malformed response), upload falls back to rule-based
   scores silently — it never breaks the request.

To enable it:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Check `GET /api/llm-status` (also shown as a badge in the dashboard header) to confirm
it's active.


## How to run it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8420
```

Then open **http://localhost:8420/** — the backend serves the frontend directly, so
there's nothing else to start.

## What I tested (all passing)

1. **Inference engine unit test** — fed a sample senior backend engineer resume directly
   into `generate_dna_json()`. Correctly scored backend=90, cloud=72, devops=71,
   leadership=56 (picked up "led 4 developers"), learning_agility=71 (picked up
   "AWS Certified" + "open-source contributor"), with evidence quoted for each.

2. **API integration test** (full server, real HTTP calls):
   - `POST /api/candidates/upload` with a resume file → 200, correct DNA JSON returned.
   - `POST /api/jobs` with a JD → 200, correct Job DNA + required-capabilities list.
   - `POST /api/match` → **91.8/100** match score, cosine similarity 0.86, 5 evidenced
     strengths, 0 gaps — correct for a strong-fit candidate.
   - Uploaded a second, clearly weaker (junior frontend) candidate against the same
     backend job → correctly scored **18.3/100**, with explicit gap explanations
     ("Cloud: JD wants 44/100, candidate shows 0/100", etc).
   - `GET /api/jobs/{id}/rank` → correctly ranked the strong candidate above the weak
     one (91.8 vs 18.3).

3. **Bug caught & fixed during testing**: the initial weighted-score formula let
   irrelevant dimensions (things the JD never asked for) dilute the score, so a clearly
   excellent match was scoring only 47/100 while cosine similarity showed 0.86. Rewrote
   `weighted_match_score()` to only weight JD-required dimensions, redistributing weight
   proportionally — the strong candidate now correctly scores 91.8/100.

4. **Frontend integration** — confirmed the FastAPI server serves `index.html` at `/`
   and the API endpoints respond correctly from the same origin (no CORS issues).

5. **Bulk upload** — uploaded 3 resumes at once (`POST /api/candidates/bulk-upload`)
   with distinctly different profiles (senior backend, junior frontend, mid DevOps).
   All 3 parsed correctly with sensible, differentiated DNA scores. Created a DevOps
   job and confirmed `GET /api/jobs/{id}/rank` correctly ranked the two devops-relevant
   candidates (93.2, 91.8) well above the frontend-only candidate (2.0).

6. **LLM reasoning layer** — tested three scenarios directly against `llm_reasoning.py`:
   - No `ANTHROPIC_API_KEY` set → `is_available()` returns `False`, `refine_dna_with_llm()`
     returns the rule-based DNA unchanged, `used_llm: False`. Confirmed via a live server
     call to `/api/llm-status` too.
   - Invalid API key set → the module correctly caught the resulting `401` from a real
     call to `api.anthropic.com` and fell back to rule-based scores rather than crashing
     the upload request (this also confirms outbound connectivity to the Anthropic API
     works from wherever this backend runs).
   - JSON parsing robustness — verified `_extract_json()` correctly handles a clean JSON
     response, one wrapped in markdown code fences, and one with preamble text before
     the JSON — all common real-world LLM response shapes.
   - **Not yet tested**: an actual successful LLM adjustment end-to-end, since that
     requires a real, valid `ANTHROPIC_API_KEY`, which I don't have in this environment.
     Worth a quick smoke test on your end once you add a key.

## Project structure

```
talentdna/
├── backend/
│   ├── main.py                 # FastAPI app, all routes
│   ├── resume_parser.py        # Module 1
│   ├── inference_engine.py     # Module 3 + 4 (rule-based core)
│   ├── llm_reasoning.py        # Optional Claude reasoning pass on top
│   ├── matching_engine.py      # Module 2 + 6 + 7
│   ├── evolution_predictor.py  # Module 8
│   └── requirements.txt
└── frontend/
    └── index.html              # Module 9 — single-page dashboard (vanilla JS + Chart.js)
```

## Known simplifications (flagged, not hidden)

- **No persistent database** — candidates/jobs are stored in-memory (`DB` dict in
  `main.py`). Fine for a demo/hackathon; swap in PostgreSQL for production per the brief.
- **No dedicated vector DB** (FAISS/ChromaDB) — with only a handful of numeric DNA
  dimensions, a plain cosine-similarity function is equivalent and avoids extra
  dependencies. Worth adding FAISS once you're ranking thousands of candidates.
- **Capability signal list is a starting point** — `CAPABILITY_SIGNALS` in
  `inference_engine.py` is easy to extend with more keywords/phrases per dimension.

## Suggested next steps

1. Swap/augment rule-based inference with an LLM reasoning pass for resumes that
   use unusual phrasing the keyword list misses.
2. Add PostgreSQL persistence so candidates/jobs survive a restart.
3. Wire in the bulk-ranking view (upload many resumes → rank against one job) — the
   backend endpoint already supports it (`GET /api/jobs/{id}/rank`), just needs a
   "bulk upload" UI.
# Talentdna_ai
