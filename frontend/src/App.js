import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Login from "./pages/login";
import Register from "./pages/registration";
import PrivateRoute from "./components/PrivateRoute";
import Dashboard from "./pages/dashboard";
import UploadComponent from "./components/UploadComponent";
import StudentResults from "./components/StudentResults";
import "./App.css";

const App = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Login />} />
          <Route path="/registration" element={<Register />} />

          {/* Private Routes */}
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/upload"
            element={
              <PrivateRoute>
                <UploadComponent />
              </PrivateRoute>
            }
          />
          <Route
            path="/student-results"
            element={
              <PrivateRoute>
                <StudentResults />
              </PrivateRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
