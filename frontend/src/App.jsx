import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import CheckerPage from "./pages/CheckerPage";
import PlagiarismCheckerPage from "./pages/PlagiarismCheckerPage";
import AiCheckerPage from "./pages/AiCheckerPage";
import ResultsPage from "./pages/ResultsPage";
import HistoryPage from "./pages/HistoryPage";
import ReportPage from "./pages/ReportPage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import RequireAuth from "./lib/RequireAuth";
import "./index.css";

function App() {
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <RequireAuth>
              <HomePage />
            </RequireAuth>
          }
        />
        <Route path="/checker" element={<CheckerPage />} />
        <Route path="/plagiarism" element={<PlagiarismCheckerPage />} />
        <Route path="/ai" element={<AiCheckerPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<ReportPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
      </Routes>
    </Router>
  );
}

export default App;
