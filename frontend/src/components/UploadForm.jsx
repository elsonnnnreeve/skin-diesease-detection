import React, { useState, useRef } from "react";

const API_URL = "/api/predict";

export default function UploadForm() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef();

  const handleFileChange = (e) => {
    setError("");
    setPrediction(null);
    const f = e.target.files?.[0];
    if (!f) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const onDropClick = () => fileInputRef.current?.click();

  const handleSubmit = async (ev) => {
    ev.preventDefault();
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const fd = new FormData();
      fd.append("image", file);

      const resp = await fetch(API_URL, { method: "POST", body: fd });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail || "Request failed");
      }
      const data = await resp.json();

      const probsArray = Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]);
      data.probsSorted = probsArray;
      data.top3 = probsArray.slice(0, 3);
      setPrediction(data);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="card">
      <form className="upload-grid" onSubmit={handleSubmit}>
        {/* LEFT: Image area */}
        <section className="panel left-panel">
          <label className="panel-title">Uploaded</label>

          <div className="image-box" onClick={onDropClick} role="button" aria-label="Select image">
            {!previewUrl && (
              <div className="empty-drop">
                <svg width="44" height="44" viewBox="0 0 24 24" aria-hidden>
                  <path fill="currentColor" d="M12 2L12 14M5 9L12 2 19 9" />
                </svg>
                <div className="empty-text">Click to select or drop an image</div>
                <div className="helper">Supported: JPG, PNG — keep closeup of lesion</div>
              </div>
            )}

            {previewUrl && <img className="preview-img" src={previewUrl} alt="selected preview" />}

            {/* overlay heatmap (if available) */}
            {prediction?.heatmap && previewUrl && (
              <img
                className="heatmap-overlay"
                src={`data:image/png;base64,${prediction.heatmap}`}
                alt="heatmap overlay"
                aria-hidden
              />
            )}
          </div>

          <div className="file-controls">
            <input
              ref={fileInputRef}
              id="file"
              name="file"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="visually-hidden"
            />
            <button type="button" className="btn secondary" onClick={onDropClick}>
              Choose File
            </button>

            <div className="selected-filename">{file ? file.name : "No file chosen"}</div>

            <div className="actions">
              <button type="submit" className="btn primary" disabled={loading}>
                {loading ? <span className="spinner" /> : "Upload & Predict"}
              </button>
              <button type="button" className="btn ghost" onClick={clearSelection} disabled={loading && !file}>
                Clear
              </button>
            </div>

            {error && <div className="form-error">{error}</div>}
          </div>
        </section>

        {/* RIGHT: Prediction area */}
        <section className="panel right-panel">
          <label className="panel-title">Prediction</label>

          {!prediction && !error && (
            <div className="placeholder">Upload an image to see predictions and explanation.</div>
          )}

          {prediction && (
            <div className="result-card">
              <div className="result-top">
                <div>
                  <div className="pred-class">{prediction.predicted_class}</div>
                  <div className="pred-confidence">
                    {(prediction.probabilities[prediction.predicted_class] * 100).toFixed(2)}%
                  </div>
                </div>
                <div className="small-meta">
                  {prediction.log_entry && (
                    <div className="log">Logged: {new Date(prediction.log_entry.timestamp).toLocaleString()}</div>
                  )}
                </div>
              </div>

              {prediction.description && (
                <div className="explain">
                  <strong>About {prediction.predicted_class}:</strong>
                  <p>{prediction.description}</p>
                </div>
              )}

              <div className="probs">
                <h4>Probabilities</h4>
                <div className="prob-list">
                  {prediction.probsSorted.map(([cls, p]) => (
                    <div className="prob-row" key={cls}>
                      <div className="prob-label">{cls}</div>
                      <div className="prob-bar-bg">
                        <div
                          className="prob-bar-fill"
                          style={{ width: `${Math.round(p * 100)}%` }}
                          aria-valuenow={Math.round(p * 100)}
                        />
                      </div>
                      <div className="prob-value">{(p * 100).toFixed(2)}%</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="top3">
                <h4>Top 3</h4>
                <ol>
                  {prediction.top3.map(([cls, p]) => (
                    <li key={cls}>
                      {cls} — {(p * 100).toFixed(2)}%
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          )}
        </section>
      </form>
    </div>
  );
}
