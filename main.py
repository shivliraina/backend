import os
import json
import logging
import traceback
from io import BytesIO
from datetime import datetime

import requests
import google.generativeai as genai
import PyPDF2
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client

# Load .env in local/development
load_dotenv()

# Flask config
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
app = Flask(__name__)
app.config["DEBUG"] = DEBUG
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

MAX_CONTENT_LENGTH = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "txt"}
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_pdf(pdf_content):
    try:
        pdf_file = BytesIO(pdf_content) if isinstance(pdf_content, bytes) else pdf_content
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        logger.info(f"PDF has {len(reader.pages)} pages")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                logger.info(f"Page {i+1}: {len(page_text)} chars")
            except Exception as e:
                logger.error(f"Error on page {i+1}: {e}")
        if not text.strip():
            return None, "No text could be extracted; PDF may be image-based."
        return text.strip(), None
    except PyPDF2.errors.PdfReadError as e:
        return None, f"Invalid PDF file: {e}"
    except Exception as e:
        return None, f"PDF extraction error: {e}"


def extract_text_from_file(file):
    try:
        fname = file.filename.lower()
        if fname.endswith(".pdf"):
            return extract_text_from_pdf(file)
        elif fname.endswith(".txt"):
            try:
                content = file.read().decode("utf-8")
                logger.info(f"Extracted {len(content)} chars from TXT")
                return content, None
            except UnicodeDecodeError:
                return None, "Text file encoding error; please use UTF-8."
        else:
            return None, f"Unsupported file format: {fname}"
    except Exception as e:
        return None, f"File extraction error: {e}"


def analyze_resume_match(job_description, resume_text, candidate_name):
    try:
        if len(resume_text) < 50:
            return {"error": "Resume text too short", "candidate_name": candidate_name, "match_score": 0}
        if len(job_description) < 50:
            return {"error": "Job description too short", "candidate_name": candidate_name, "match_score": 0}

        # Truncate to token limits
        resume_text = (resume_text[:15000] + "...") if len(resume_text) > 15000 else resume_text
        job_description = (job_description[:8000] + "...") if len(job_description) > 8000 else job_description

        prompt = f"""
You are an expert HR recruiter. Analyze the following resume against the job description and provide a comprehensive evaluation.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Please provide a detailed analysis in the following JSON format:
{{
    "candidate_name": "{candidate_name}",
    "match_score": 75,
    "experience_years": 5,
    "matching_skills": ["Python", "JavaScript", "React"],
    "missing_skills": ["AWS", "Docker"],
    "strengths": ["Strong technical background", "Good communication skills"],
    "weaknesses": ["Limited cloud experience", "No DevOps experience"],
    "recommendation": "review",
    "summary": "Good technical candidate with room for growth"
}}

Important rules:
1. match_score must be a number between 0-100
2. experience_years must be a number (estimate if not clear)
3. recommendation must be exactly one of: "qualified", "review", "not_qualified"
4. All arrays must contain strings
5. Return ONLY the JSON object, no additional text or formatting
"""
        response = model.generate_content(prompt)
        text = response.text.strip()

        # strip markdown fences if any
        for fence in ["```json", "```"]:
            text = text.strip().lstrip(fence).rstrip(fence).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.error("JSON parsing failed; fallback.")
            return {
                "candidate_name": candidate_name,
                "match_score": 50,
                "experience_years": 0,
                "matching_skills": [],
                "missing_skills": [],
                "strengths": ["Analysis completed"],
                "weaknesses": ["Unable to parse detailed analysis"],
                "recommendation": "review",
                "summary": "Analysis completed but parsing failed",
                "error": "JSON parsing failed, using fallback analysis",
            }

        # validate fields
        score = result.get("match_score", 50)
        exp = result.get("experience_years", 0)
        rec = result.get("recommendation", "review")

        result["match_score"] = score if isinstance(score, (int, float)) and 0 <= score <= 100 else 50
        result["experience_years"] = exp if isinstance(exp, (int, float)) and exp >= 0 else 0
        result["recommendation"] = rec if rec in ["qualified", "review", "not_qualified"] else "review"

        # ensure lists/strings
        for arr in ["matching_skills", "missing_skills", "strengths", "weaknesses"]:
            result[arr] = result.get(arr) if isinstance(result.get(arr), list) else []
        result["summary"] = result.get("summary", "Analysis completed")

        return result

    except Exception as e:
        logger.error(f"analyze_resume_match error: {e}\n{traceback.format_exc()}")
        return {"error": f"Analysis failed: {e}", "candidate_name": candidate_name, "match_score": 0}


@app.route("/analyze-resumes", methods=["POST"])
def analyze_resumes():
    try:
        job_title = request.form.get("jobTitle")
        job_description = request.form.get("jobDescription")
        if not job_title or not job_description:
            return jsonify({"error": "Missing job title or description"}), 400

        # insert job record
        job_data = {
            "title": job_title,
            "description": job_description,
            "required_skills": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        job_result = supabase.table("jobs").insert(job_data).execute()
        if not job_result.data:
            return jsonify({"error": "DB error creating job"}), 500
        job_id = job_result.data[0]["id"]

        files = request.files.getlist("resumes")
        if not files:
            return jsonify({"error": "No resume files uploaded"}), 400

        results = []
        for file in files:
            if not (file and file.filename and allowed_file(file.filename)):
                continue

            candidate_name = (
                os.path.splitext(file.filename)[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )
            resume_text, err = extract_text_from_file(file)
            if err:
                results.append({"candidate_name": candidate_name, "error": err, "match_score": 0})
                continue

            # save candidate
            cand_data = {
                "job_id": job_id,
                "name": candidate_name,
                "filename": file.filename,
                "resume_text": resume_text[:50000],
                "created_at": datetime.utcnow().isoformat(),
            }
            cand_res = supabase.table("candidates").insert(cand_data).execute()
            candidate_id = cand_res.data[0]["id"] if cand_res.data else None

            analysis = analyze_resume_match(job_description, resume_text, candidate_name)

            if not analysis.get("error") and candidate_id:
                rec = {
                    "job_id": job_id,
                    "candidate_id": candidate_id,
                    "match_score": int(analysis["match_score"]),
                    "experience_years": int(analysis["experience_years"]),
                    "matching_skills": analysis["matching_skills"],
                    "missing_skills": analysis["missing_skills"],
                    "strengths": analysis["strengths"],
                    "weaknesses": analysis["weaknesses"],
                    "recommendation": analysis["recommendation"],
                    "summary": analysis["summary"],
                    "created_at": datetime.utcnow().isoformat(),
                }
                supabase.table("analysis_results").insert(rec).execute()

            results.append(analysis)

        results.sort(key=lambda r: r.get("match_score", 0), reverse=True)
        return (
            jsonify(
                {
                    "job_id": job_id,
                    "job_title": job_title,
                    "total_candidates": len(results),
                    "results": results,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"analyze_resumes error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/test-ai", methods=["POST"])
def test_ai():
    try:
        data = request.get_json() or {}
        txt = data.get("text", "Hello, test.")
        resp = model.generate_content(
            f"Please respond with JSON {{'message':'AI is working','received_text':'{txt}'}}"
        )
        return jsonify({"status": "success", "response": resp.text}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    try:
        db_test = supabase.table("jobs").select("count").execute()
        ai_test = model.generate_content("Hi")
        return jsonify(
            {
                "status": "healthy",
                "database": "connected" if db_test else "error",
                "ai_service": "connected" if ai_test else "error",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=DEBUG)