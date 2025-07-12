import os
import json
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from supabase import create_client, Client
from datetime import datetime
import uuid
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Supabase
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Gemini AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash')

# File upload configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content with better error handling"""
    try:
        if isinstance(pdf_content, bytes):
            pdf_file = BytesIO(pdf_content)
        else:
            pdf_file = pdf_content
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        logger.info(f"PDF has {len(pdf_reader.pages)} pages")
        
        for page_num in range(len(pdf_reader.pages)):
            try:
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
                logger.info(f"Extracted {len(page_text)} characters from page {page_num + 1}")
            except Exception as e:
                logger.error(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        if not text.strip():
            return None, "No text could be extracted from the PDF. The PDF might be image-based or corrupted."
        
        logger.info(f"Total extracted text length: {len(text)} characters")
        return text.strip(), None
        
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"PDF read error: {str(e)}")
        return None, f"Invalid PDF file: {str(e)}"
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return None, f"Error extracting text from PDF: {str(e)}"

def extract_text_from_file(file):
    """Extract text from uploaded file with better error handling"""
    try:
        filename = file.filename.lower()
        logger.info(f"Processing file: {filename}")
        
        if filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif filename.endswith('.txt'):
            try:
                content = file.read().decode('utf-8')
                logger.info(f"Extracted {len(content)} characters from TXT file")
                return content, None
            except UnicodeDecodeError as e:
                logger.error(f"Text file encoding error: {str(e)}")
                return None, f"Text file encoding error. Please ensure the file is UTF-8 encoded."
        else:
            return None, f"Unsupported file format: {filename}"
            
    except Exception as e:
        logger.error(f"File extraction error: {str(e)}")
        return None, f"Error extracting text from file: {str(e)}"

def analyze_resume_match(job_description, resume_text, candidate_name):
    """Analyze resume match with comprehensive error handling"""
    try:
        logger.info(f"Starting analysis for candidate: {candidate_name}")
        
        # Check text lengths
        if len(resume_text) < 50:
            logger.warning(f"Resume text too short: {len(resume_text)} characters")
            return {
                "error": "Resume text too short to analyze",
                "candidate_name": candidate_name,
                "match_score": 0
            }
        
        if len(job_description) < 50:
            logger.warning(f"Job description too short: {len(job_description)} characters")
            return {
                "error": "Job description too short to analyze",
                "candidate_name": candidate_name,
                "match_score": 0
            }
        
        # Truncate if too long (Gemini has token limits)
        max_resume_length = 15000
        max_job_length = 8000
        
        if len(resume_text) > max_resume_length:
            logger.warning(f"Truncating resume text from {len(resume_text)} to {max_resume_length} characters")
            resume_text = resume_text[:max_resume_length] + "..."
        
        if len(job_description) > max_job_length:
            logger.warning(f"Truncating job description from {len(job_description)} to {max_job_length} characters")
            job_description = job_description[:max_job_length] + "..."

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
        
        logger.info("Sending request to Gemini AI...")
        
        # Make API call with timeout and retry logic
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            logger.info(f"Received response from Gemini: {len(response_text)} characters")
            logger.info(f"Response preview: {response_text[:200]}...")
            
        except Exception as api_error:
            logger.error(f"Gemini API error: {str(api_error)}")
            return {
                "error": f"AI service error: {str(api_error)}",
                "candidate_name": candidate_name,
                "match_score": 0
            }
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        
        response_text = response_text.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
            logger.info(f"Successfully parsed JSON response for {candidate_name}")
            
            # Validate required fields
            required_fields = ['match_score', 'experience_years', 'recommendation']
            for field in required_fields:
                if field not in result:
                    logger.error(f"Missing required field: {field}")
                    return {
                        "error": f"AI response missing required field: {field}",
                        "candidate_name": candidate_name,
                        "match_score": 0
                    }
            
            # Validate data types and ranges
            if not isinstance(result['match_score'], (int, float)) or not (0 <= result['match_score'] <= 100):
                logger.error(f"Invalid match_score: {result['match_score']}")
                result['match_score'] = 50  # Default fallback
            
            if not isinstance(result['experience_years'], (int, float)) or result['experience_years'] < 0:
                logger.error(f"Invalid experience_years: {result['experience_years']}")
                result['experience_years'] = 0  # Default fallback
            
            if result['recommendation'] not in ['qualified', 'review', 'not_qualified']:
                logger.error(f"Invalid recommendation: {result['recommendation']}")
                result['recommendation'] = 'review'  # Default fallback
            
            # Ensure arrays exist
            for array_field in ['matching_skills', 'missing_skills', 'strengths', 'weaknesses']:
                if array_field not in result or not isinstance(result[array_field], list):
                    result[array_field] = []
            
            # Ensure strings exist
            for string_field in ['summary']:
                if string_field not in result or not isinstance(result[string_field], str):
                    result[string_field] = "Analysis completed"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Response text: {response_text}")
            
            # Return a fallback result
            return {
                "candidate_name": candidate_name,
                "match_score": 50,
                "experience_years": 0,
                "matching_skills": [],
                "missing_skills": [],
                "strengths": ["Analysis completed"],
                "weaknesses": ["Unable to parse detailed analysis"],
                "recommendation": "review",
                "summary": "Analysis completed but detailed parsing failed",
                "error": "JSON parsing failed, using fallback analysis"
            }
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_resume_match: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "candidate_name": candidate_name,
            "match_score": 0
        }

@app.route('/analyze-resumes', methods=['POST'])
def analyze_resumes():
    """Main endpoint with comprehensive error handling"""
    try:
        logger.info("Starting resume analysis request...")
        
        # Get job description from form data
        job_title = request.form.get('jobTitle')
        job_description = request.form.get('jobDescription')
        
        logger.info(f"Job title: {job_title}")
        logger.info(f"Job description length: {len(job_description) if job_description else 0}")
        
        if not job_title or not job_description:
            return jsonify({"error": "Missing job title or description"}), 400
        
        # Test database connection first
        try:
            # Create job record in database
            job_data = {
                "title": job_title,
                "description": job_description,
                "required_skills": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            job_result = supabase.table('jobs').insert(job_data).execute()
            
            if not job_result.data:
                logger.error("Failed to create job record in database")
                return jsonify({"error": "Database error: Failed to create job record"}), 500
            
            job_id = job_result.data[0]['id']
            logger.info(f"Created job record with ID: {job_id}")
            
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            return jsonify({"error": f"Database connection failed: {str(db_error)}"}), 500
        
        # Get uploaded files
        files = request.files.getlist('resumes')
        logger.info(f"Received {len(files)} files")
        
        if not files or len(files) == 0:
            return jsonify({"error": "No resume files uploaded"}), 400
        
        results = []
        
        for i, file in enumerate(files):
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            if file and file.filename and allowed_file(file.filename):
                # Extract candidate name from filename
                candidate_name = os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ').title()
                
                # Extract text from resume
                resume_text, error = extract_text_from_file(file)
                
                if error:
                    logger.error(f"File extraction error for {candidate_name}: {error}")
                    results.append({
                        "candidate_name": candidate_name,
                        "error": error,
                        "match_score": 0
                    })
                    continue
                
                # Save candidate to database
                try:
                    candidate_data = {
                        "job_id": job_id,
                        "name": candidate_name,
                        "filename": file.filename,
                        "resume_text": resume_text[:50000],  # Limit text length for database
                        "created_at": datetime.now().isoformat()
                    }
                    
                    candidate_result = supabase.table('candidates').insert(candidate_data).execute()
                    
                    if not candidate_result.data:
                        logger.error(f"Failed to save candidate {candidate_name} to database")
                        results.append({
                            "candidate_name": candidate_name,
                            "error": "Failed to save candidate data",
                            "match_score": 0
                        })
                        continue
                    
                    candidate_id = candidate_result.data[0]['id']
                    logger.info(f"Saved candidate {candidate_name} with ID: {candidate_id}")
                    
                except Exception as db_error:
                    logger.error(f"Database error saving candidate {candidate_name}: {str(db_error)}")
                    results.append({
                        "candidate_name": candidate_name,
                        "error": f"Database error: {str(db_error)}",
                        "match_score": 0
                    })
                    continue
                
                # Analyze resume against job description
                logger.info(f"Starting AI analysis for {candidate_name}")
                analysis = analyze_resume_match(job_description, resume_text, candidate_name)
                
                # Save analysis results if successful
                if "error" not in analysis:
                    try:
                        analysis_record = {
                            "job_id": job_id,
                            "candidate_id": candidate_id,
                            "match_score": int(analysis.get("match_score", 0)),
                            "experience_years": int(analysis.get("experience_years", 0)),
                            "matching_skills": analysis.get("matching_skills", []),
                            "missing_skills": analysis.get("missing_skills", []),
                            "strengths": analysis.get("strengths", []),
                            "weaknesses": analysis.get("weaknesses", []),
                            "recommendation": analysis.get("recommendation", "review"),
                            "summary": analysis.get("summary", ""),
                            "created_at": datetime.now().isoformat()
                        }
                        
                        supabase.table('analysis_results').insert(analysis_record).execute()
                        logger.info(f"Saved analysis results for {candidate_name}")
                        
                    except Exception as db_error:
                        logger.error(f"Database error saving analysis for {candidate_name}: {str(db_error)}")
                        # Continue anyway, we have the analysis
                
                results.append(analysis)
                
            else:
                logger.warning(f"Skipping invalid file: {file.filename}")
        
        # Sort results by match score (highest first)
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        logger.info(f"Analysis complete. Processed {len(results)} candidates")
        
        return jsonify({
            "job_id": job_id,
            "job_title": job_title,
            "total_candidates": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Test endpoint to check AI connection
@app.route('/test-ai', methods=['POST'])
def test_ai():
    """Test endpoint to verify AI connection"""
    try:
        data = request.get_json()
        test_text = data.get('text', 'Hello, this is a test.')
        
        response = model.generate_content(f"Please respond with a simple JSON object containing 'message': 'AI is working' and 'received_text': '{test_text}'")
        
        return jsonify({
            "status": "success",
            "response": response.text
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    try:
        # Test database connection
        db_test = supabase.table('jobs').select('count').execute()
        db_status = "connected" if db_test else "error"
        
        # Test AI connection
        ai_test = model.generate_content("Hello")
        ai_status = "connected" if ai_test else "error"
        
        return jsonify({
            "status": "healthy",
            "database": db_status,
            "ai_service": ai_status,
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Check environment variables
    required_vars = ['GOOGLE_API_KEY', 'SUPABASE_URL', 'SUPABASE_ANON_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing environment variables: {', '.join(missing_vars)}")
        exit(1)
    
    print("Starting server with enhanced error handling...")
    app.run(debug=True, host='0.0.0.0', port=5000)