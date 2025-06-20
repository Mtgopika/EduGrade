from flask import Flask, request, jsonify
import os
import pdfplumber
import json
from tensorflow.keras.models import load_model
from lastocr import ( extract_characters, predict_text_from_images,
    CTCLayer)
from PIL import Image
import numpy as np
import cv2
from flask_cors import CORS
import traceback
import re
import nltk
from spellchecker import SpellChecker
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, storage, firestore
from grading import match_questions, calculate_scores  # Assuming these functions are imported from grading.py

# Ensure required NLTK data is available
nltk.download('punkt')


# Initialize Flask app
app = Flask(__name__)

# CORS setup
CORS(app, resources={r"/grade": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"  # Allow the specific origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"  # Allow credentials (cookies, etc.)
    return response


# Firebase Initialization
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "edugrade-e75e0.firebasestorage.app"
})
bucket = storage.bucket()
db = firestore.client()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load OCR model with custom layer
model = load_model('handwritingg.h5', custom_objects={'CTCLayer': CTCLayer})

# Initialize spell checker
spell = SpellChecker()

import requests
from io import BytesIO
from pdf2image import convert_from_bytes

def process_pdf_from_stream(pdf_stream):
    
    try:
        images = convert_from_bytes(pdf_stream.read())
        
        # Debug: Check image dimensions and handle invalid images
        valid_images = []
        for i, image in enumerate(images):
            width, height = image.size
            print(f"Image {i+1}: width={width}, height={height}")

            if width <= 0 or height <= 0:
                print(f"Skipping image {i+1} due to invalid dimensions: {width}x{height}")
                continue  # Skip invalid image

            # Check if image is completely white (empty)
            image_np = np.array(image)
            if np.all(image_np == 255):  # If image is all white
                print(f"Skipping image {i+1} as it is completely white (blank page)")
                continue  # Skip blank image

            valid_images.append(image)
        
        if not valid_images:
            print("No valid images found in the PDF.")
            # Optionally, return a default image (e.g., a blank image) if no valid pages
            default_image = Image.new("RGB", (100, 100), (255, 255, 255))  # White 100x100 image
            valid_images.append(default_image)
        
        return valid_images
    except Exception as e:
        print(f"Error processing PDF from stream: {e}")
        return []  # Return an empty list in case of error



def extract_text_from_pdf_from_url(pdf_url):
    
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check if the request was successful
        pdf_data = BytesIO(response.content)

        with pdfplumber.open(pdf_data) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n" if page.extract_text() else ""
        return text.strip()

    except Exception as e:
        print(f"Error extracting text from PDF URL: {e}")
        return ""
        
def process_answer_sheet_with_ocr_from_url(pdf_url):
    
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check if the request was successful
        pdf_data = BytesIO(response.content)

        # Convert PDF pages to images
        images = process_pdf_from_stream(pdf_data)
        
        # If no valid images are found, log it
        if not images:
            print("No valid images were extracted from the PDF.")
            return ""  # Return empty string if no valid images
        
        student_answers_texts = []
        
        for i, image in enumerate(images):
            try:
                # Check image dimensions
                width, height = image.size
                print(f"Image {i+1}: width={width}, height={height}")

                if width <= 0 or height <= 0:
                    print(f"Skipping image {i+1} due to invalid dimensions: {width}x{height}")
                    continue  # Skip invalid image

                # Convert image to numpy array (OpenCV format) for processing
                image_np = np.array(image.convert('RGB'))
 

                # Extract characters from the image
                extracted_characters = extract_characters(image_np)

                # Check if extracted characters are empty
                if not extracted_characters:
                    print(f"No characters extracted from image {i+1}. This could be due to empty or unreadable text.")
                    continue  # Skip image if no characters were extracted

                # Predict text from the extracted characters using OCR model
                page_predicted_texts = predict_text_from_images(extracted_characters)

                # If OCR produces empty or invalid text, skip it
                if not page_predicted_texts:
                    print(f"OCR returned no text for image {i+1}. Skipping this image.")
                    continue  # Skip this image if OCR returns empty text

                student_answers_texts.extend(page_predicted_texts)
                
            except Exception as e:
                print(f"Error processing image {i+1}: {e}")
                continue  # Skip this image if an error occurs

        return " ".join(student_answers_texts)

    except Exception as e:
        print(f"Error processing answer sheet PDF URL: {e}")
        return ""  # Return empty string if something goes wrong




# Function to correct spelling errors
def correct_spelling(text):
    words = text.split()
    misspelled = spell.unknown(words)
    corrections = {word: spell.correction(word) or word for word in misspelled}
    corrected_words = [corrections.get(word, word) for word in words]
    return ' '.join(corrected_words)

# Function to fix punctuation spacing
def fix_punctuation(text):
    text = re.sub(r'(?<!\w)([.,!?;:])', r' \1', text)  
    text = re.sub(r'([.,!?;:])(?!\s|$)', r'\1 ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text


def correct_grammar(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    corrected_sentences = [str(TextBlob(sentence).correct()) for sentence in sentences]
    return ' '.join(corrected_sentences)

# Function to post-process the text
def postprocess_text(text):
    print("\nPerforming post-processing...\n")
    text = text.lower() 
    text = correct_spelling(text)  
    text = fix_punctuation(text)  
    text = correct_grammar(text) 
    text = text.capitalize()  
    return text

# Firebase file upload function
def upload_to_firebase(file_path, file_name):
    """Uploads a file to Firebase Storage and returns its URL."""
    blob = bucket.blob(file_name)
    blob.upload_from_filename(file_path)
    blob.make_public()
    return blob.public_url

# Main route to handle file uploads and OCR processing
@app.route('/grade', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract URLs and other data from the JSON
        answer_key_url = data.get('answerKeyUrl')
        answer_sheet_url = data.get('answerSheetUrl')
        exam_id = data.get('examId')
        subject_code = data.get('subjectCode')
        student_id = data.get('studentId')

        # Check if the necessary data is provided
        if not all([answer_key_url, answer_sheet_url, exam_id, subject_code, student_id]):
            return jsonify({'error': 'Missing required fields: answerKeyUrl, answerSheetUrl, examId, subjectCode, studentId'}), 400

        # Extract Answer Key Text using pdfplumber from the provided URL
        answer_key_text = extract_text_from_pdf_from_url(answer_key_url)
       # print(student_answers_text)
        if not answer_key_text:
            return jsonify({'error': 'Failed to extract text from answer key PDF'}), 400

        # Process Student Answer Sheet using OCR (from the provided URL)
        student_answers_text = process_answer_sheet_with_ocr_from_url(answer_sheet_url)
        if not student_answers_text:
            return jsonify({'error': 'Failed to extract text from answer sheet PDF'}), 400
        

        # Post-process the OCR text (spell check, grammar, punctuation)
        postprocessed_text = postprocess_text(student_answers_text)

        # Match questions and calculate scores
        matched_answers = match_questions(answer_key_text, postprocessed_text)
        score_details_str, final_percentage = calculate_scores(matched_answers)

        # Save grading results to Firestore
        grading_results = {
            'final_percentage': final_percentage,
            'detailed_scores': score_details_str
        }

        grading_results_ref = db.collection("grades").document(student_id).collection("exams").document(f"{exam_id}_{subject_code}")
        grading_results_ref.set(grading_results)

        # Include file URLs in the response
        grading_results['answer_key_url'] = answer_key_url
        grading_results['answer_sheet_url'] = answer_sheet_url

        return jsonify(grading_results)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during grading:\n{error_trace}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


@app.route('/get_grades', methods=['GET'])
def get_grades():
    try:
        student_id = request.args.get('student_id')

        if student_id:
            doc_ref = db.collection("grades").document(student_id)
            doc = doc_ref.get()
            if doc.exists:
                return jsonify(doc.to_dict())
            else:
                return jsonify({'error': 'No grades found for this student'}), 404

        grades_ref = db.collection("grades").stream()
        grades = [doc.to_dict() for doc in grades_ref]
        return jsonify(grades)

    except Exception as e:
        print(f"❌ Error fetching grades: {e}")
        return jsonify({'error': f'Failed to fetch grades: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)