import React, { useState, useEffect } from "react";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { useNavigate, useLocation } from "react-router-dom";
import UploadComponent from "../components/UploadComponent";
import "./Dashboard.css";

const Dashboard = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const user = location.state?.user || {};
  const [gradingResult, setGradingResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSignOut = async () => {
    try {
      await signOut(auth);
      alert("You have been logged out!");
      navigate("/");
    } catch (error) {
      console.error("Error signing out:", error.message);
    }
  };

  const fetchGradingResults = async () => {
    const studentId = user.regNo || user.admissionNumber; // Ensure correct field is used

    if (!studentId) {
      setError("Student registration number not found.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://127.0.0.1:5000/get-grades?student_id=${studentId}`);
      if (!response.ok) throw new Error(`Server Error: ${response.status}`);
      
      const data = await response.json();
      setGradingResult(data);
    } catch (error) {
      setError(error.message || "Failed to fetch grading results.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <h1>Welcome, {user.name || "Guest"}!</h1>

      {user.role === "teacher" ? (
        <div className="teacher-section">
          <p><strong>Teacher ID:</strong> {user.id}</p>
          <UploadComponent />
        </div>
      ) : user.role === "student" ? (
        <div className="student-section">
          <p><strong>Registration Number:</strong> {user.regNo || user.admissionNumber || "N/A"}</p>
          
          <button 
            className="view-marks-button" 
            onClick={fetchGradingResults} 
            disabled={loading}
          >
            {loading ? "Fetching..." : "View Marks"}
          </button>

          {error && <p style={{ color: "red" }}>{error}</p>}

          {gradingResult && (
            <div className="grading-results">
              <h3>📊 Grading Results</h3>
              <p><strong>Final Percentage:</strong> {gradingResult.final_percentage ? `${gradingResult.final_percentage}%` : "N/A"}</p>
              
              <h4>📝 Detailed Scores:</h4>
              {gradingResult.detailed_scores ? (
  <pre style={{
    background: "#f4f4f4",
    padding: "10px",
    borderRadius: "5px",
    overflowX: "auto",
    whiteSpace: "pre-wrap"
  }}>
    {gradingResult.detailed_scores.split("\n").map((line, index) => (
      <React.Fragment key={index}>
        {line}
        <br />
      </React.Fragment>
    ))}
  </pre>
) : (
  <p>No detailed scores available.</p>
)}

            </div>
          )}
        </div>
      ) : (
        <p style={{ color: "red" }}>Invalid user role.</p>
      )}

      <button className="signout-button" onClick={handleSignOut}>🚪 Sign Out</button>
    </div>
  );
};

export default Dashboard;
