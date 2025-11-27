import UploadForm from "./components/UploadForm";

function App() {
  return (
    <div style={{ padding: 20, background: "#f6f7fb", minHeight: "100vh" }}>
      <div style={{ maxWidth: 900, margin: "0 auto", background: "#fff", padding: 24, borderRadius: 8, boxShadow: "0 4px 12px rgba(0,0,0,0.06)" }}>
        <h1 style={{ marginTop: 0 }}>Skin Disease Detection (Demo)</h1>
        <UploadForm />
      </div>
    </div>
  );
}

export default App;
