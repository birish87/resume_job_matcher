# JobResumeMatcher

**JobResumeMatcher** is a production-ready NLP engine that matches resumes to jobs and jobs to resumes using a hybrid approach that combines **semantic embeddings** with **explicit skill matching**.

It supports:
- Resume → Job search
- Job → Resume search
- Direct Resume ↔ Job comparison

The system is designed to be **fast**, **explainable**, and **deployment-ready**, with a Flask API and a simple front-end UI.

---

## Why This Project Exists

Traditional keyword-based resume matching fails when:
- Different terminology is used (`ML` vs `Machine Learning`)
- Relevant experience is phrased differently
- Semantic similarity matters more than exact wording

This project addresses those limitations by combining:
- **Sentence embeddings** for semantic similarity
- **Explicit skill extraction** for explainability
- **Weighted scoring + confidence filtering** for meaningful ranking

The result is a system that is both **accurate** and **interpretable**.

---

## High-Level Architecture

```text
User Input
│
▼
Text Normalization
│
├─► Synonym Expansion
│
├─► Skill Extraction (Regex)
│
└─► Embedding Encoding
│
▼
Similarity + Skill Scoring
│
▼
Filtering & Ranking
│
▼
API Response
```
---

## How Matching Works (Detailed)

### 1. Text Normalization
- Lowercases text
- Expands skill synonyms (`ml → machine learning`)
- Removes formatting inconsistencies

### 2. Skill Extraction
- Uses a predefined skill vocabulary (`skills.txt`)
- Extracts skills using a **compiled regex** (fast)
- Produces:
  - `matching_skills`
  - `missing_skills`

### 3. Semantic Similarity
- Uses `SentenceTransformer (paraphrase-MiniLM-L6-v2)`
- Computes cosine similarity between resume and job text
- Stored resumes and jobs have **precomputed embeddings** for performance

### 4. Scoring Formula

skill_score = |matching_skills| / |job_skills|
final_score =
(skill_score × skill_weight) +
(semantic_similarity × text_weight)

```yaml
Default weights:
- Skills: 60%
- Semantic similarity: 40%
```

### 5. Confidence & Ranking
- Confidence is derived from cosine similarity
- Results are filtered by:
  - Minimum confidence threshold
  - Minimum match score
- Final results are sorted by:
  1. Match score (descending)
  2. Confidence (descending)

---

## Features

- Resume ↔ Job matching
- Resume → Job search
- Job → Resume search
- Matching skills and missing skills
- Match score (0–100)
- Confidence score (semantic similarity)
- Explainable recommendations:
  - `Good`
  - `Weak`
  - `Poor`
- Fast execution via precomputed embeddings
- REST API
- Simple front-end UI
- Railway-ready deployment

---

## API Endpoints

### `/api/match` — POST

Accepts:
- Resume only → returns top matching jobs
- Job only → returns top matching resumes
- Resume + Job → returns a single match result

---

## API Examples

### Resume + Job (Single Match)

```bash
curl -X POST http://127.0.0.1:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{"resume":"Python developer with ML and SQL","job":"Machine learning engineer with Python and SQL"}'
```

### Resume → Jobs
```bash
curl -X POST http://127.0.0.1:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{"resume":"Senior Python engineer with ML and SQL"}'
```
### Job → Resumes
```bash
curl -X POST http://127.0.0.1:5000/api/match \
  -H "Content-Type: application/json" \
  -d '{"job":"Looking for a machine learning engineer"}'
```

## Project Structure
```csharp
resume_job_matcher/
├── app.py                 # Flask app & matcher logic
├── resumes.csv            # Resume dataset
├── jobs.csv               # Job dataset
├── skills.txt             # Canonical skills list
├── synonyms.json          # Skill synonym mappings
├── templates/
│   └── index.html         # Front-end UI
├── static/
│   └── app.js             # Front-end logic
├── requirements.txt       # Dependencies
├── .gitignore
└── README.md
```
---


## Configuration Files
skills.txt
Canonical skills list (one per line):

sql
python
machine learning
sql
javascript
deep learning
project management
synonyms.json

Canonical skill → aliases mapping:

```json
{
  "machine learning": ["ml", "deep learning", "dl"],
  "python": ["py"],
  "javascript": ["js"],
  "sql": ["structured query language"],
  "project management": ["pm"],
  "user experience": ["ux"],
  "user interface": ["ui"],
  "software as a service": ["saas"]
}
```
---
## Running Locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Then open:

http://127.0.0.1:5000
Deployment (Railway)
Uses requirements.txt

No training step required

Embeddings are computed at startup

Stateless API suitable for scaling

Recommended Railway Settings

Python 3.10+

≥ 1GB RAM

Single worker per instance (embedding model is memory heavy)

## Performance Optimizations
Skill regex compiled once at startup

Resume and job embeddings precomputed

Early skill-based filtering prevents unnecessary cosine similarity calls

Sorting happens only after filtering

No repeated embedding computation inside loops

---

## Design Tradeoffs
| Decision                | Reason                                |
| ----------------------- | ------------------------------------- |
| Use MiniLM              | Fast inference, good semantic quality |
| Regex skills            | Deterministic + explainable           |
| Precompute embeddings   | Major latency reduction               |
| Weighted hybrid scoring | Balances precision & recall           |

---

## Future Enhancements
Experience-based scoring bonus
Education-level weighting
Skill importance weighting per job
Approximate nearest-neighbor search (FAISS)
Vector database integration
Async API execution
Authentication & user profiles
---

## License
MIT License