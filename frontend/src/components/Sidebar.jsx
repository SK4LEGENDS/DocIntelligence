import React, { useRef, useState } from "react";
import { FileText, Upload, Trash2, Loader2, AlertTriangle, CheckCircle2, Sun, Moon } from "lucide-react";

export default function Sidebar({ pdfs, activePdfId, onSelectPdf, onDeletePdf, onUploadSuccess, theme, onToggleTheme }) {
  const fileInputRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }

    setUploading(true);
    setUploadError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data = await response.json();
      onUploadSuccess(data);
    } catch (err) {
      console.error(err);
      setUploadError(err.message || "Failed to upload file.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset input
      }
    }
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>DocIntelligence</h2>
        <p>Hierarchical Agentic PDF Chatbot</p>
      </div>

      <div className="upload-section">
        <label className="upload-label">
          <Upload className={`upload-icon ${uploading ? "pulse" : ""}`} size={28} />
          <span className="upload-text">
            {uploading ? "Uploading PDF..." : "Upload Document"}
          </span>
          <span className="upload-subtext">Click to choose PDF</span>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf"
            disabled={uploading}
          />
        </label>
        {uploadError && (
          <div style={{ color: "var(--accent-red)", fontSize: "0.75rem", marginTop: "0.5rem", textAlign: "center" }}>
            {uploadError}
          </div>
        )}
      </div>

      <div className="pdf-list-container">
        <h3 className="pdf-list-title">My Documents</h3>
        {pdfs.length === 0 ? (
          <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: "0.8rem", marginTop: "2rem" }}>
            No documents uploaded yet.
          </div>
        ) : (
          pdfs.map((pdf) => {
            const isActive = pdf.id === activePdfId;
            return (
              <div
                key={pdf.id}
                className={`pdf-item ${isActive ? "active" : ""}`}
                onClick={() => pdf.status === "ready" && onSelectPdf(pdf.id)}
                style={{ cursor: pdf.status === "ready" ? "pointer" : "default" }}
              >
                <div className="pdf-item-left">
                  <FileText className="pdf-icon" size={18} />
                  <div className="pdf-info">
                    <div className="pdf-name" title={pdf.filename}>
                      {pdf.filename}
                    </div>
                    <div>
                      {pdf.status && pdf.status.startsWith("parsing") && (
                        <span className="pdf-status-badge parsing">
                          <Loader2 size={10} className="spin" style={{ display: "inline", marginRight: "3px", verticalAlign: "middle" }} />
                          {pdf.status.includes(":") ? pdf.status.split(":")[1].trim() : "parsing"}
                        </span>
                      )}
                      {pdf.status === "ready" && (
                        <span className="pdf-status-badge ready">
                          <CheckCircle2 size={10} style={{ display: "inline", marginRight: "3px", verticalAlign: "middle" }} />
                          ready
                        </span>
                      )}
                      {pdf.status === "error" && (
                        <span className="pdf-status-badge error" title={pdf.error_message}>
                          <AlertTriangle size={10} style={{ display: "inline", marginRight: "3px", verticalAlign: "middle" }} />
                          error
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  className="pdf-delete-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeletePdf(pdf.id);
                  }}
                  title="Delete Document"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            );
          })
        )}
      </div>
      <div className="sidebar-footer">
        <button className="theme-toggle-btn" onClick={onToggleTheme} title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}>
          {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
          <span>{theme === "light" ? "Dark Mode" : "Light Mode"}</span>
        </button>
      </div>
    </aside>
  );
}

