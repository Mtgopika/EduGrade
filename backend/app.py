import firebase_admin
from firebase_admin import credentials, storage, firestore
from flask import Flask, request, jsonify
import os
import pdfplumber
import requests
import traceback
from grading import match_questions, calculate_scores
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "edugrade-e75e0.firebasestorage.app"
})
bucket = storage.bucket()
db = firestore.client()

def download_file(url, save_path):
    """Downloads a file from a given URL and saves it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"✅ Downloaded: {save_path}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text.strip()

@app.route('/grade', methods=['OPTIONS', 'POST'])
def grade():
    if request.method == 'OPTIONS':
        return jsonify({'message': 'CORS Preflight Passed'}), 200

    try:
        data = request.json
        print(f"Received Data: {data}")

        answer_key_url = data.get("answerKeyUrl")
        answer_sheet_url = data.get("answerSheetUrl")
        student_id = data.get("studentId")
        exam_id = data.get("examId")
        subject_code = data.get("subjectCode")

        if not all([answer_key_url, answer_sheet_url, student_id, exam_id, subject_code]):
            return jsonify({'error': 'Missing required fields'}), 400

        print(f"📌 Grading Request for Student: {student_id}")

        # Define file paths
        answer_key_path = os.path.join(UPLOAD_FOLDER, f"answer_key_{exam_id}.pdf")
        answer_sheet_path = os.path.join(UPLOAD_FOLDER, f"answer_sheet_{student_id}.pdf")

        # Download PDFs
        download_file(answer_key_url, answer_key_path)
        download_file(answer_sheet_url, answer_sheet_path)

        # Extract text from the PDFs
        answer_key_text = extract_text_from_pdf(answer_key_path)
        if not answer_key_text:
            return jsonify({'error': 'Failed to extract text from answer key PDF'}), 500
        print(f"📄 Extracted Answer Key:\n{answer_key_text[:500]}")

        student_answer_text = extract_text_from_pdf(answer_sheet_path)
        if not student_answer_text:
            return jsonify({'error': 'Failed to extract text from answer sheet PDF'}), 500
        print(f"📄 Extracted Student Answer Sheet:\n{student_answer_text[:500]}")

        # Match questions and calculate scores
        matched_answers = match_questions(answer_key_text, student_answer_text)
        score_details, final_percentage = calculate_scores(matched_answers)
        print(f"Matched Answers: {matched_answers}")

        if not score_details or final_percentage is None:
            print("⚠️ No valid answers found. Returning 0 score.")
            return jsonify({'message': 'No valid answers found', 'final_percentage': 0, 'detailed_scores': {}}), 200

        grading_results = {
            'student_id': student_id,
            'exam_id': exam_id,
            'subject_code': subject_code,
            'final_percentage': final_percentage,
            'detailed_scores': score_details,
        }

        print(f"✅ Computed grading results: {grading_results}")

        # Store the grading results in Firebase Firestore (under the exams sub-collection)
        try:
            doc_ref = db.collection("grades").document(student_id).collection("exams").document(f"{exam_id}_{subject_code}")
            doc_ref.set(grading_results)

            stored_doc = doc_ref.get()
            if stored_doc.exists:
                print(f"🔥 Successfully stored in Firestore: {stored_doc.to_dict()}")
            else:
                print("⚠️ Firestore write failed!")

        except Exception as firestore_error:
            print(f"🔥 Firestore Error: {traceback.format_exc()}")

        return jsonify(grading_results)

    except Exception as e:
        print(f"❌ Error during processing: {traceback.format_exc()}")
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

@app.route('/get-grades', methods=['GET'])
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

