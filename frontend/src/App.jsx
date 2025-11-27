import React from "react";
import UploadForm from "./components/UploadForm";
import "./index.css";

export default function App() {
  return (
    <div className="app-root">
      <header className="site-header">
        <div className="site-brand">
          <h1>AI Skin Disease Detection</h1>
          <p className="tagline">Explainable predictions — upload a dermal image to analyze</p>
        </div>
      </header>

      <main className="main-container">
        <UploadForm />
      </main>

      <footer className="site-footer">
        <small>Major Project — AI-Based Skin Disease Detection</small>
      </footer>
    </div>
  );
}
