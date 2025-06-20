import React, { useState } from "react";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth ,db } from "../firebase";
import { doc, getDoc } from "firebase/firestore";
// Import the auth instance
import { useNavigate } from "react-router-dom";
import "./login.css";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const backgroundImageUrl = 'https://in.pinterest.com/pin/744853225904925976/';

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      // Log in the user
      const result = await signInWithEmailAndPassword(auth, email, password);
  
      // Fetch user details from Firestore
      const userDoc = doc(db, "users", result.user.uid);
      const userData = await getDoc(userDoc);
  
      if (userData.exists()) {
        const data = userData.data();
        navigate("/dashboard", { state: { user: data } }); // Pass user data to Dashboard
      } else {
        console.error("No such user found!");
      }
    } catch (err) {
      setError(err.message);
    }
  };



  return (
    <div className="login-container" style={{ backgroundImage: `url(${backgroundImageUrl})` }}>
      <h2>Login</h2>
      <form onSubmit={handleLogin}>
        <label>Email:</label>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />

        <label>Password:</label>
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && <div className="error">{error}</div>}

        <button type="submit">Login</button>
      </form>

      <div className="register-link">
        <p>Don't have an account?</p>
        <button onClick={() => navigate("/registration")}>Register</button>
      </div>
    </div>
  );
};

export default Login;
