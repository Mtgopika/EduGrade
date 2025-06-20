import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { storage, db } from "../firebase";
import { ref, uploadBytesResumable, getDownloadURL } from "firebase/storage";
import { collection, addDoc, Timestamp } from "firebase/firestore";
import axios from "axios";


const UploadComponent = () => {
    const [formData, setFormData] = useState({
        StudentName: "", Semester: "", RegNo: "", ExamID: "", SubjectCode: ""
    });
    const [answerSheet, setAnswerSheet] = useState(null);
    const [answerKey, setAnswerKey] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const navigate = useNavigate();

    const handleInputChange = useCallback((e) => {
        setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    }, []);

    const handleFileChange = useCallback((event, type) => {
        const file = event.target.files[0];
        if (!file) return;
        if (file.type !== "application/pdf") {
            setError("Only PDF files are allowed.");
            return;
        }
        setError(null);
        type === "answerSheet" ? setAnswerSheet(file) : setAnswerKey(file);
    }, []);

    const uploadFileToFirebase = async (file, path) => {
        const storageRef = ref(storage, path);
        const uploadTask = uploadBytesResumable(storageRef, file);
        return new Promise((resolve, reject) => {
            uploadTask.on("state_changed", null, reject, async () => {
                resolve(await getDownloadURL(uploadTask.snapshot.ref));
            });
        });
    };

    const handleUpload = async () => {
        if (!answerSheet || !answerKey) {
            setError("Please upload both files.");
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const { RegNo, ExamID, SubjectCode } = formData;
            const timestamp = Date.now();
            const answerSheetUrl = await uploadFileToFirebase(answerSheet, `answer_sheets/${RegNo}_${ExamID}_${timestamp}.pdf`);
            const answerKeyUrl = await uploadFileToFirebase(answerKey, `answer_keys/${ExamID}_${SubjectCode}_${timestamp}.pdf`);
            const docRef = await addDoc(collection(db, "graded_papers"), {
                ...formData, answerSheetUrl, answerKeyUrl, timestamp: Timestamp.now()
            });
            console.log("📩 Sending data to backend:", {
                answerSheetUrl, 
                answerKeyUrl, 
                ExamID, 
                SubjectCode, 
                studentId: RegNo 
            });
            
            const response = await axios.post("http://127.0.0.1:5000/grade", 
                { 
                    answerSheetUrl, 
                    answerKeyUrl, 
                    examId: ExamID, 
                    subjectCode: SubjectCode, 
                    studentId: RegNo // Change `RegNo` to `studentId`
                },
                { headers: { "Content-Type": "application/json" }, withCredentials: true }
            );
            
        
            
            navigate("/student-results", { state: { studentDetails: formData, documentId: docRef.id, gradingResult: response.data } });
            console.log("📩 Sending data to backend:", JSON.stringify({
                answerSheetUrl, 
                answerKeyUrl, 
                ExamID, 
                SubjectCode, 
                studentId: RegNo 
            }, null, 2)); // Pretty-print JSON
            
            console.log("📩 Sending student details:", JSON.stringify(formData, null, 2));
            
           
            
        } catch (err) {
            setError("Upload & Grading Failed. Try Again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dashboard-container">
            <h2>Upload Answer Sheet</h2>
            <div className="input-container">
                {Object.entries(formData).map(([key, value]) => (
                    <input key={key} type="text" name={key} value={value} onChange={handleInputChange} placeholder={key} />
                ))}
    
                <div className="file-upload-container">
                    <label>Upload Answer Sheet:</label>
                    <input type="file" accept=".pdf" onChange={(e) => handleFileChange(e, "answerSheet")} />
                </div>
    
                <div className="file-upload-container">
                    <label>Upload Answer Key:</label>
                    <input type="file" accept=".pdf" onChange={(e) => handleFileChange(e, "answerKey")} />
                </div>
    
                <button className="upload-button" onClick={handleUpload} disabled={loading}>
                    {loading ? "Uploading..." : "Upload & Grade"}
                </button>
                {error && <p style={{ color: "red" }}>{error}</p>}
            </div>
        </div>
    );
    
};

export default UploadComponent;
