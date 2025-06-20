import React from "react";
import { Navigate } from "react-router-dom";
import { useAuthState } from "react-firebase-hooks/auth";
import { auth } from "../firebase"; // Import the Firebase auth instance

const PrivateRoute = ({ children }) => {
  const [user, loading] = useAuthState(auth); // Hook to check authentication state

  if (loading) {
    return <h1>Loading...</h1>; // Optional: Add a loading spinner
  }

  return user ? children : <Navigate to="/" />;
};

export default PrivateRoute;
