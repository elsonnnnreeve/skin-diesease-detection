// frontend/src/components/UploadForm.jsx
import React, { useState } from "react";
import axios from "axios";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Choose an image first.");
    setLoading(true);
    const fd = new FormData();
    fd.append("image", file);
    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert("Error: " + (err.response?.data || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: "1rem auto", fontFamily: "Arial, sans-serif" }}>
      <h2>Upload skin image</h2>
      <p style={{ color: "#555" }}>Take a photo in good light, focus on the lesion, avoid blurriness.</p>
      <form onSubmit={submit}>
        <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files[0])} />
        <div style={{ marginTop: 10 }}>
          <button type="submit" disabled={loading}>{loading ? "Analyzing..." : "Upload & Analyze"}</button>
        </div>
      </form>

      {result && (
        <div style={{ marginTop: 20 }}>
          <h3>Prediction</h3>
          <pre>{JSON.stringify(result.predictions || result, null, 2)}</pre>
          {result.heatmap_b64 && (
            <div>
              <h4>Heatmap</h4>
              <img src={`data:image/png;base64,${result.heatmap_b64}`} alt="heatmap" style={{ maxWidth: "100%" }} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
