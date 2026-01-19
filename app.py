import re
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util


# ============================================================
# JobResumeMatcher
# ============================================================
# This class encapsulates ALL logic related to:
#   - Loading resumes and jobs
#   - Extracting skills
#   - Computing semantic similarity
#   - Ranking matches efficiently
#
# IMPORTANT DESIGN GOAL:
#   Expensive operations (NLP, embeddings, parsing) are done
#   ONCE at startup, not repeatedly per request.
# ============================================================
class JobResumeMatcher:
    def __init__(self, resume_file, job_file, skills_file, synonyms_file=None):
        """
        Constructor is intentionally heavy.
        We do expensive preprocessing here so that
        API requests stay fast.
        """

        # ----------------------------------------------------
        # Load resumes and jobs into DataFrames
        # ----------------------------------------------------
        # Pandas is used for:
        #   - Fast iteration
        #   - Vectorized preprocessing
        #   - Convenient column storage (skills, embeddings)
        self.resumes = pd.read_csv(resume_file)
        self.jobs = pd.read_csv(job_file)

        # ----------------------------------------------------
        # Load skills list
        # ----------------------------------------------------
        # This is the authoritative list of skills we match on.
        # Everything is normalized to lowercase to avoid
        # case-sensitivity bugs.
        with open(skills_file) as f:
            self.skills = [s.strip().lower() for s in f if s.strip()]

        # ----------------------------------------------------
        # Load synonyms (optional)
        # ----------------------------------------------------
        # Currently not applied in this optimized version,
        # but loaded so the design supports future expansion
        # (e.g., synonym normalization or alias replacement).
        self.synonyms = {}
        if synonyms_file:
            with open(synonyms_file) as f:
                self.synonyms = json.load(f)

        # ----------------------------------------------------
        # Build a compiled REGEX for skill extraction
        # ----------------------------------------------------
        # WHY regex?
        #   - Much faster than spaCy / token loops
        #   - Exact-word matching avoids false positives
        #
        # Example pattern:
        #   \b(python|sql|machine\ learning)\b
        self.skill_pattern = re.compile(
            r"\b(" + "|".join(map(re.escape, self.skills)) + r")\b"
        )

        # ----------------------------------------------------
        # Initialize sentence embedding model
        # ----------------------------------------------------
        # MiniLM is chosen for:
        #   - Good semantic quality
        #   - Very fast inference
        #   - Small memory footprint
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

        # ====================================================
        # PREPROCESS EVERYTHING ONCE (CRITICAL FOR SPEED)
        # ====================================================

        # ----------------------------------------------------
        # Normalize all resume and job text
        # ----------------------------------------------------
        # - Fill NaNs to avoid crashes
        # - Lowercase for consistent matching
        self.resumes["normalized"] = (
            self.resumes["resume_text"].fillna("").str.lower()
        )
        self.jobs["normalized"] = (
            self.jobs["job_description"].fillna("").str.lower()
        )

        # ----------------------------------------------------
        # Extract skills ONCE per document
        # ----------------------------------------------------
        # These become cached sets used during matching.
        self.resumes["skills"] = self.resumes["normalized"].apply(self.extract_skills)
        self.jobs["skills"] = self.jobs["normalized"].apply(self.extract_skills)

        # ----------------------------------------------------
        # Precompute embeddings ONCE
        # ----------------------------------------------------
        # THIS IS THE MOST IMPORTANT PERFORMANCE OPTIMIZATION.
        #
        # Without this:
        #   - Each API call recomputes embeddings
        #   - Complexity explodes to O(N × model_inference)
        #
        # With this:
        #   - Matching becomes O(N × cosine_similarity)
        self.resumes["embedding"] = list(
            self.model.encode(
                self.resumes["normalized"].tolist(),
                convert_to_tensor=True
            )
        )
        self.jobs["embedding"] = list(
            self.model.encode(
                self.jobs["normalized"].tolist(),
                convert_to_tensor=True
            )
        )

    # ========================================================
    # Synonym normalization (not currently used)
    # =======================================================
    def normalize_synonyms(self, text: str) -> str:
        """
        Replace known abbreviations with canonical skill names.
        This runs ONCE per input, not per comparison.
        """
        for alias, canonical in self.synonyms.items():
            text = re.sub(
                rf"\b{re.escape(alias)}\b",
                canonical,
                text
            )
        return text

    # ========================================================
    # Skill extraction
    # ========================================================
    def extract_skills(self, text: str) -> set:
        """
        Extract skills using a compiled regex.

        WHY return a set?
        - Fast intersection operations
        - No duplicates
        """
        return set(self.skill_pattern.findall(text))

    # ========================================================
    # Match ONE resume against ONE job
    # ========================================================
    def match_resume_to_job(self, resume_text, job_text, weights=None):
        """
        Used when BOTH resume and job are provided.
        This is a single-pair comparison (no loops).
        """

        # Default weighting:
        # Skills matter more than semantic similarity
        if not weights:
            weights = {"skills": 0.6, "text": 0.4}

        # Normalize inputs
        resume_text = self.normalize_synonyms(resume_text.lower())
        job_text = job_text.lower()

        # Extract skills from inputs
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_text)

        # Skill overlap
        matching = resume_skills & job_skills
        missing = job_skills - resume_skills

        # ----------------------------------------------------
        # Encode each input ONCE
        # ----------------------------------------------------
        resume_vec = self.model.encode(resume_text, convert_to_tensor=True)
        job_vec = self.model.encode(job_text, convert_to_tensor=True)

        # Semantic similarity (confidence proxy)
        confidence = float(util.cos_sim(resume_vec, job_vec))

        # Skill score is a recall-style metric:
        #   "How many required skills are present?"
        skill_score = len(matching) / max(len(job_skills), 1)

        # Final weighted score
        match_score = (
            skill_score * weights["skills"] +
            confidence * weights["text"]
        ) * 100

        return {
            "matching_skills": sorted(matching),
            "missing_skills": sorted(missing),
            "match_score": round(match_score, 2),
            "confidence": round(confidence, 3),
            "recommendation": (
                "Good" if match_score > 75 else
                "Weak" if match_score > 50 else
                "Poor"
            )
        }

    # ========================================================
    # Find BEST jobs for a given resume
    # ========================================================
    def search_jobs_for_resume(self, resume_text, top_k=5):
        """
        Resume → many jobs
        Optimized for speed using:
        - Cached embeddings
        - Early skill filtering
        """

        resume_text = self.normalize_synonyms(resume_text.lower())
        resume_vec = self.model.encode(resume_text, convert_to_tensor=True)
        resume_skills = self.extract_skills(resume_text)

        results = []

        for _, job in self.jobs.iterrows():
            job_skills = job["skills"]

            # ------------------------------------------------
            # EARLY EXIT: no shared skills
            # ------------------------------------------------
            # This avoids unnecessary cosine similarity calls
            if not resume_skills & job_skills:
                continue

            # Use cached job embedding
            confidence = float(util.cos_sim(resume_vec, job["embedding"]))
            if confidence < 0.45:
                continue

            skill_score = len(resume_skills & job_skills) / max(len(job_skills), 1)
            match_score = (skill_score * 0.6 + confidence * 0.4) * 100

            if match_score < 55:
                continue

            results.append({
                "job_title": job["title"],
                "matching_skills": sorted(resume_skills & job_skills),
                "missing_skills": sorted(job_skills - resume_skills),
                "match_score": round(match_score, 2),
                "confidence": round(confidence, 3)
            })

        # Sort best matches first
        results.sort(
            key=lambda x: (x["match_score"], x["confidence"]),
            reverse=True
        )

        return results[:top_k]

    # ========================================================
    # Find BEST resumes for a given job
    # ========================================================
    def search_resumes_for_job(self, job_text, top_k=5):
        """
        Job → many resumes
        Same optimizations as resume → jobs
        """

        job_text = self.normalize_synonyms(job_text.lower())
        job_vec = self.model.encode(job_text, convert_to_tensor=True)
        job_skills = self.extract_skills(job_text)

        results = []

        for _, resume in self.resumes.iterrows():
            resume_skills = resume["skills"]

            if not job_skills & resume_skills:
                continue

            confidence = float(util.cos_sim(job_vec, resume["embedding"]))
            if confidence < 0.5:
                continue

            skill_score = len(job_skills & resume_skills) / max(len(job_skills), 1)
            match_score = (skill_score * 0.6 + confidence * 0.4) * 100

            if match_score < 30:
                continue

            results.append({
                "resume_name": resume["name"],
                "matching_skills": sorted(job_skills & resume_skills),
                "missing_skills": sorted(job_skills - resume_skills),
                "match_score": round(match_score, 2),
                "confidence": round(confidence, 3)
            })

        results.sort(
            key=lambda x: (x["match_score"], x["confidence"]),
            reverse=True
        )

        return results[:top_k]


# ============================================================
# Flask API
# ============================================================
app = Flask(__name__)
CORS(app)

# Heavy initialization happens ONCE at startup
matcher = JobResumeMatcher(
    "resumes.csv",
    "jobs.csv",
    "skills.txt",
    "synonyms.json"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/match", methods=["POST"])
def match():
    """
    Unified endpoint:
      - resume + job → single comparison
      - resume only → job search
      - job only → resume search
    """

    data = request.json or {}
    resume = data.get("resume", "")
    job = data.get("job", "")

    if resume and job:
        return jsonify(matcher.match_resume_to_job(resume, job))
    if resume:
        return jsonify(matcher.search_jobs_for_resume(resume))
    if job:
        return jsonify(matcher.search_resumes_for_job(job))

    return jsonify({"error": "Provide resume and/or job"}), 400


if __name__ == "__main__":
    # threaded=True allows concurrent requests
    app.run(host="0.0.0.0", port=5000, threaded=True)
