import React, { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import "./StudentResults.css";

const StudentResults = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const gradingResult = location.state?.gradingResult;
    const studentDetails = location.state?.studentDetails;

    // Debugging logs
    useEffect(() => {
        console.log("Location State:", location.state);
        if (!gradingResult) console.error("Error: Grading result is missing!");
        if (!studentDetails) console.error("Error: Student details are missing!");
    }, [location.state]);

    if (!gradingResult || !studentDetails) {
        return <p className="error-message">No results available. Please upload answer sheets first.</p>;
    }

    return (
        <div className="results-container">
            <h2>📄 Grading Results</h2>

            {/* Display Student Details */}
            <h3>👨‍🎓 Student Information</h3>
            <p><strong>Name:</strong> {studentDetails.StudentName || "N/A"}</p>
            <p><strong>Semester:</strong> {studentDetails.Semester || "N/A"}</p>
            <p><strong>Registration No:</strong> {studentDetails.RegNo || "N/A"}</p>
            <p><strong>Exam ID:</strong> {studentDetails.ExamID || "N/A"}</p>
            <p><strong>Subject Code:</strong> {studentDetails.SubjectCode || "N/A"}</p>

            {/* Display Grading Result */}
            <h3>📊 Results</h3>
            <p><strong>Final Percentage:</strong> {gradingResult.final_percentage ? `${gradingResult.final_percentage}%` : "N/A"}</p>
            
            <h3>📝 Marks Breakdown:</h3>
            {gradingResult.detailed_scores ? (
                <ul className="simple-scores-list">
                    {gradingResult.detailed_scores.map((q, idx) => (
                        <li key={idx}>
                            <strong>Question {q["Question Number"]}:</strong> {q["Marks Obtained"]}
                            {q.sub_questions && Object.keys(q.sub_questions).length > 0 && (
                                <ul className="sub-scores-list">
                                    {Object.entries(q.sub_questions).map(([subQ, subData], subIdx) => (
                                        <li key={subIdx}>
                                            ( {subQ} ) – {subData["Marks Obtained"]}
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </li>
                    ))}
    </ul>
) : (
    <p>No detailed scores available.</p>
)}



            <button className="back-button" onClick={() => navigate("/")}>
                ⬅ Go Back
            </button>
        </div>
    );
};

export default StudentResults;
