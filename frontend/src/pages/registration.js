import React, { useState } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth, db } from "../firebase"; 
import { doc, setDoc } from "firebase/firestore";
import { useNavigate } from "react-router-dom";
import "./registration.css";

const Register = () => {
  const [name, setName] = useState("");
  const [role, setRole] = useState(""); // Dropdown for role
  const [regNo, setRegNo] = useState(""); // Registration Number (Only for students)
  const [teacherId, setTeacherId] = useState(""); // Teacher ID (Only for teachers)
  const [semester, setSemester] = useState(""); // Semester (Only for students)
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      // Create user in Firebase Authentication
      const result = await createUserWithEmailAndPassword(auth, email, password);

      // Store user details in Firestore
      const userDoc = doc(db, "users", result.user.uid);
      await setDoc(userDoc, {
        name,
        role,
        regNo: role === "student" ? regNo : null, // Store only if student
        teacherId: role === "teacher" ? teacherId : null, // Store only if teacher
        semester: role === "student" ? semester : null, // Store only if student
        email,
      });

      alert("Registration successful!");
      navigate("/"); // Redirect to login page
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="register-container">
      <h2>Register</h2>
      <form onSubmit={handleRegister}>
        {/* Name Field */}
        <label>Name:</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
        />

        {/* Dropdown for Role Selection */}
        <label>Role:</label>
        <select value={role} onChange={(e) => setRole(e.target.value)} required>
          <option value="">Select Role</option>
          <option value="student">Student</option>
          <option value="teacher">Teacher</option>
        </select>

        {/* Registration Number Field (Only for students) */}
        {role === "student" && (
          <div>
            <label>Registration Number:</label>
            <input
              type="text"
              value={regNo}
              onChange={(e) => setRegNo(e.target.value)}
              required
            />
          </div>
        )}

        {/* Semester Field (Only for students) */}
        {role === "student" && (
          <div>
            <label>Semester:</label>
            <input
              type="text"
              value={semester}
              onChange={(e) => setSemester(e.target.value)}
              required
            />
          </div>
        )}

        {/* Teacher ID Field (Only for teachers) */}
        {role === "teacher" && (
          <div>
            <label>Teacher ID:</label>
            <input
              type="text"
              value={teacherId}
              onChange={(e) => setTeacherId(e.target.value)}
              required
            />
          </div>
        )}

        {/* Email Field */}
        <label>Email:</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        {/* Password Field */}
        <label>Password:</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {/* Error Display */}
        {error && <div className="error">{error}</div>}

        <button type="submit">Register</button>
      </form>
    </div>
  );
};

export default Register;
