import os
import json
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import tempfile
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content"""
    try:
        if isinstance(pdf_content, bytes):
            pdf_file = BytesIO(pdf_content)
        else:
            pdf_file = pdf_content
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        if not text.strip():
            return None, "No text could be extracted from the PDF"
        
        return text.strip(), None
        
    except Exception as e:
        return None, f"Error extracting text from PDF: {str(e)}"

def extract_text_from_file(file):
    """Extract text from uploaded file"""
    try:
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            return content, None
        elif filename.endswith(('.doc', '.docx')):
            # For now, return error for Word docs (you can add python-docx later)
            return None, "Word document processing not yet supported. Please convert to PDF."
        else:
            return None, "Unsupported file format"
            
    except Exception as e:
        return None, f"Error extracting text from file: {str(e)}"

def analyze_resume_match(job_description, resume_text, candidate_name):
    """Analyze resume match against job description using Gemini AI"""
    prompt = f"""
    You are an expert HR recruiter. Analyze the following resume against the job description and provide a comprehensive evaluation.

    JOB DESCRIPTION:
    {job_description}

    RESUME:
    {resume_text}

    Please provide a detailed analysis in the following JSON format:
    {{
        "candidate_name": "{candidate_name}",
        "match_score": <number between 0-100>,
        "experience_years": <estimated years of experience>,
        "matching_skills": [<array of skills that match job requirements>],
        "missing_skills": [<array of important skills missing from resume>],
        "strengths": [<array of candidate's key strengths>],
        "weaknesses": [<array of areas for improvement>],
        "recommendation": "<qualified/review/not_qualified>",
        "summary": "<brief summary of why this candidate is/isn't a good fit>"
    }}

    Base your analysis on:
    1. Technical skills alignment
    2. Experience level match
    3. Relevant background
    4. Education fit
    5. Overall suitability

    Only return the JSON object, no additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {
            "error": "Failed to parse AI response",
            "candidate_name": candidate_name,
            "match_score": 0
        }
    except Exception as e:
        print(f"AI generation error: {e}")
        return {
            "error": "Failed to analyze resume",
            "candidate_name": candidate_name,
            "match_score": 0
        }

# Your existing endpoints
@app.route('/extract-skills', methods=['POST'])
def extract_skills_endpoint():
    """Extract technical skills from job description"""
    try:
        data = request.get_json()
        
        if not data or 'job_description' not in data:
            return jsonify({"error": "Missing 'job_description' in request body"}), 400
        
        job_description = data['job_description']
        
        if not job_description.strip():
            return jsonify({"error": "Job description cannot be empty"}), 400
        
        # Extract skills using your existing function
        result = extract_technical_skills(job_description)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def extract_technical_skills(job_description):
    """Extract technical skills from job description using Gemini AI"""
    prompt = f"""
    Analyze the following job description and extract ALL technical skills mentioned. 
    This includes programming languages, frameworks, tools, technologies, databases, 
    cloud platforms, methodologies, certifications, and any other technical requirements.
    
    Job Description:
    {job_description}
    
    Please return the result as a valid JSON object with the following structure:
    {{
        "technical_skills": [
            "skill1",
            "skill2",
            "skill3"
        ]
    }}
    
    Only return the JSON object, no additional text or formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        result = json.loads(response_text)
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {"error": "Failed to parse AI response as JSON"}
    except Exception as e:
        print(f"AI generation error: {e}")
        return {"error": "Failed to generate response from AI"}

# NEW ENDPOINTS FOR RESUME MATCHING

@app.route('/analyze-resumes', methods=['POST'])
def analyze_resumes():
    """Main endpoint to analyze multiple resumes against a job description"""
    try:
        # Get job description from form data
        job_title = request.form.get('jobTitle')
        job_description = request.form.get('jobDescription')
        
        if not job_title or not job_description:
            return jsonify({"error": "Missing job title or description"}), 400
        
        # Get uploaded files
        files = request.files.getlist('resumes')
        
        if not files or len(files) == 0:
            return jsonify({"error": "No resume files uploaded"}), 400
        
        results = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                # Extract candidate name from filename
                candidate_name = os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ').title()
                
                # Extract text from resume
                resume_text, error = extract_text_from_file(file)
                
                if error:
                    results.append({
                        "candidate_name": candidate_name,
                        "error": error,
                        "match_score": 0
                    })
                    continue
                
                # Analyze resume against job description
                analysis = analyze_resume_match(job_description, resume_text, candidate_name)
                results.append(analysis)
        
        # Sort results by match score (highest first)
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return jsonify({
            "job_title": job_title,
            "total_candidates": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "message": "Resume Matching API",
        "endpoints": {
            "/analyze-resumes": {
                "method": "POST",
                "description": "Analyze multiple resumes against job description",
                "content-type": "multipart/form-data",
                "body": {
                    "jobTitle": "string - Job title",
                    "jobDescription": "string - Job description text",
                    "resumes": "files[] - Array of resume files (PDF, TXT)"
                }
            },
            "/extract-skills": {
                "method": "POST",
                "description": "Extract technical skills from job description",
                "body": {"job_description": "string"}
            },
            "/health": {
                "method": "GET",
                "description": "Health check"
            }
        }
    })

if __name__ == '__main__':
    if not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY not found in environment variables")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)