import os
import json
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import tempfile

load_dotenv()

app = Flask(__name__)

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

model = genai.GenerativeModel('gemini-2.5-flash')

def download_pdf_from_url(url):
    """
    Download PDF from URL and return the content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            if not response.content.startswith(b'%PDF'):
                return None, "The URL does not point to a valid PDF file"
        
        return response.content, None
        
    except requests.exceptions.Timeout:
        return None, "Request timeout - PDF download took too long"
    except requests.exceptions.ConnectionError:
        return None, "Connection error - Unable to reach the URL"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error {e.response.status_code} - {e.response.reason}"
    except Exception as e:
        return None, f"Error downloading PDF: {str(e)}"

def extract_text_from_pdf(pdf_content):
    """
    Extract text from PDF content
    """
    try:
        # Create a BytesIO object from the PDF content
        pdf_file = BytesIO(pdf_content)
        
        # Create PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        if not text.strip():
            return None, "No text could be extracted from the PDF"
        
        return text.strip(), None
        
    except PyPDF2.errors.PdfReadError:
        return None, "Invalid PDF file or corrupted PDF"
    except Exception as e:
        return None, f"Error extracting text from PDF: {str(e)}"

def extract_technical_skills(job_description):
    """
    Extract technical skills from job description using Gemini AI
    """
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
        
        # Extract the text response
        response_text = response.text.strip()
        
        # Remove any markdown formatting if present
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
        print(f"Response text: {response_text}")
        return {"error": "Failed to parse AI response as JSON"}
    except Exception as e:
        print(f"AI generation error: {e}")
        return {"error": "Failed to generate response from AI"}

@app.route('/extract-skills', methods=['POST'])
def extract_skills_endpoint():
    """
    API endpoint to extract technical skills from job description
    """
    try:
        # Get job description from request
        data = request.get_json()
        
        if not data or 'job_description' not in data:
            return jsonify({
                "error": "Missing 'job_description' in request body"
            }), 400
        
        job_description = data['job_description']
        
        if not job_description.strip():
            return jsonify({
                "error": "Job description cannot be empty"
            }), 400
        
        # Extract skills using AI
        result = extract_technical_skills(job_description)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/extract-skills-from-pdf', methods=['POST'])
def extract_skills_from_pdf_endpoint():
    """
    API endpoint to extract technical skills from PDF URL
    """
    try:
        # Get PDF URL from request
        data = request.get_json()
        
        if not data or 'pdf_url' not in data:
            return jsonify({
                "error": "Missing 'pdf_url' in request body"
            }), 400
        
        pdf_url = data['pdf_url'].strip()
        
        if not pdf_url:
            return jsonify({
                "error": "PDF URL cannot be empty"
            }), 400
        
        # Validate URL format
        if not (pdf_url.startswith('http://') or pdf_url.startswith('https://')):
            return jsonify({
                "error": "Invalid URL format. URL must start with http:// or https://"
            }), 400
        
        # Download PDF from URL
        pdf_content, download_error = download_pdf_from_url(pdf_url)
        if download_error:
            return jsonify({
                "error": f"Failed to download PDF: {download_error}"
            }), 400
        
        # Extract text from PDF
        pdf_text, extraction_error = extract_text_from_pdf(pdf_content)
        if extraction_error:
            return jsonify({
                "error": f"Failed to extract text from PDF: {extraction_error}"
            }), 400
        
        # Check if extracted text is too long (Gemini has token limits)
        if len(pdf_text) > 30000:  # Approximate token limit consideration
            pdf_text = pdf_text[:30000] + "..."
            print("Warning: PDF text was truncated due to length")
        
        # Extract skills using AI
        result = extract_technical_skills(pdf_text)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Add metadata to response
        result["metadata"] = {
            "pdf_url": pdf_url,
            "text_length": len(pdf_text),
            "truncated": len(pdf_text) > 30000
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"status": "healthy"}), 200

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint with API documentation
    """
    return jsonify({
        "message": "Technical Skills Extractor API",
        "endpoints": {
            "/extract-skills": {
                "method": "POST",
                "description": "Extract technical skills from job description text",
                "body": {
                    "job_description": "string - The job description text"
                },
                "response": {
                    "technical_skills": ["array of extracted skills"]
                }
            },
            "/extract-skills-from-pdf": {
                "method": "POST",
                "description": "Extract technical skills from PDF URL",
                "body": {
                    "pdf_url": "string - URL to the PDF file"
                },
                "response": {
                    "technical_skills": ["array of extracted skills"],
                    "metadata": {
                        "pdf_url": "original URL",
                        "text_length": "length of extracted text",
                        "truncated": "boolean indicating if text was truncated"
                    }
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

if __name__ == '__main__':
    # Check if API key is configured
    if not os.getenv('GOOGLE_API_KEY'):
        print("Error: GOOGLE_API_KEY not found in environment variables")
        exit(1)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
