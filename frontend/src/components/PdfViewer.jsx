import React, { useEffect, useRef, useState } from "react";
import { Eye, HelpCircle, FileSpreadsheet, Copy, Check } from "lucide-react";

export default function PdfViewer({ activePdf, currentPageInfo }) {
  const iframeRef = useRef(null);
  const [activeTab, setActiveTab] = useState("pdf");
  const [tables, setTables] = useState([]);
  const [loadingTables, setLoadingTables] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  // Auto-switch back to PDF tab when a new citation is clicked
  useEffect(() => {
    if (currentPageInfo?.page) {
      setActiveTab("pdf");
    }
  }, [currentPageInfo]);

  // Handle PDF iframe src updates on page info changes
  useEffect(() => {
    if (activePdf && currentPageInfo && iframeRef.current && activeTab === "pdf") {
      const page = currentPageInfo.page || 1;
      const snippet = currentPageInfo.snippet || "";
      const baseUrl = `/api/pdfs/${activePdf.id}/file`;
      const searchParam = snippet ? `&search=${encodeURIComponent(snippet)}` : "";
      iframeRef.current.src = `${baseUrl}#page=${page}${searchParam}&toolbar=0&navpanes=0`;
    }
  }, [activePdf, currentPageInfo, activeTab]);

  // Fetch tables when active PDF changes
  useEffect(() => {
    if (!activePdf) {
      setTables([]);
      return;
    }
    const fetchTables = async () => {
      setLoadingTables(true);
      try {
        const response = await fetch(`/api/pdfs/${activePdf.id}/tables`);
        if (response.ok) {
          const data = await response.json();
          setTables(data);
        }
      } catch (err) {
        console.error("Failed to fetch tables:", err);
      } finally {
        setLoadingTables(false);
      }
    };
    fetchTables();
  }, [activePdf?.id]);

  const handleCopyMarkdown = (tableId, markdown) => {
    navigator.clipboard.writeText(markdown);
    setCopiedId(tableId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const renderMarkdownTable = (markdown) => {
    if (!markdown) return null;
    const lines = markdown.trim().split("\n");
    if (lines.length === 0) return null;

    // Filter out the separator line (e.g. |---|---|)
    const filteredLines = lines.filter(
      (line) => !line.match(/^\s*\|?\s*:?-+:?\s*(\|?\s*:?-+:?\s*)*$/)
    );

    const rows = filteredLines.map((line) => {
      // Split by | but ignore leading and trailing |
      const parts = line.split("|").map((p) => p.trim());
      if (line.startsWith("|")) parts.shift();
      if (line.endsWith("|")) parts.pop();
      return parts;
    });

    if (rows.length === 0) return null;

    const headers = rows[0];
    const bodyRows = rows.slice(1);

    return (
      <div className="parsed-table-wrapper">
        <table className="parsed-table">
          <thead>
            <tr>
              {headers.map((h, i) => (
                <th key={i}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {bodyRows.map((row, rIdx) => (
              <tr key={rIdx}>
                {row.map((cell, cIdx) => (
                  <td key={cIdx}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  if (!activePdf) {
    return (
      <div className="pdf-viewer-container">
        <div className="pdf-viewer-empty">
          <HelpCircle className="pdf-viewer-empty-icon" size={48} />
          <h3>No PDF Selected</h3>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", maxWidth: "240px", textAlign: "center" }}>
            Select a document from the sidebar to view it side-by-side with the chat.
          </p>
        </div>
      </div>
    );
  }

  // Construct default iframe url for initial rendering
  const pdfUrl = `/api/pdfs/${activePdf.id}/file#toolbar=0&navpanes=0`;

  return (
    <div className="pdf-viewer-container">
      {/* Tabs / Header */}
      <div className="pdf-viewer-header-tabs">
        <button
          className={`pdf-tab-btn ${activeTab === "pdf" ? "active" : ""}`}
          onClick={() => setActiveTab("pdf")}
        >
          <Eye size={15} />
          <span>Document View</span>
        </button>
        <button
          className={`pdf-tab-btn ${activeTab === "tables" ? "active" : ""}`}
          onClick={() => setActiveTab("tables")}
        >
          <FileSpreadsheet size={15} />
          <span>Extracted Tables ({tables.length})</span>
        </button>
      </div>

      {/* Tab Contents */}
      {activeTab === "pdf" ? (
        <div className="pdf-tab-content-pdf">
          <div className="pdf-viewer-subheader">
            <span className="pdf-filename-txt" title={activePdf.filename}>
              {activePdf.filename}
            </span>
            {currentPageInfo?.page && (
              <span className="pdf-page-badge">
                Page {currentPageInfo.page} {activePdf.page_count ? `of ${activePdf.page_count}` : ""}
              </span>
            )}
          </div>
          <iframe
            ref={iframeRef}
            key={activePdf.id}
            src={pdfUrl}
            className="pdf-viewer-frame"
            title="PDF Document Viewer"
          />
        </div>
      ) : (
        <div className="pdf-tab-content-tables">
          {loadingTables ? (
            <div className="tables-loading-state">
              <span className="spin-loader" />
              <p>Fetching extracted tables...</p>
            </div>
          ) : tables.length === 0 ? (
            <div className="tables-empty-state">
              <FileSpreadsheet size={32} style={{ color: "var(--text-muted)", opacity: 0.4 }} />
              <p className="empty-title">No tables found</p>
              <p className="empty-sub">
                Table extraction is active for Advanced Mode files (50+ pages processed via Docling).
              </p>
            </div>
          ) : (
            <div className="tables-list">
              {tables.map((table) => (
                <div key={table.id} className="table-card">
                  <div className="table-card-header">
                    <span className="table-card-page">Page {table.page_number}</span>
                    <button
                      className="table-copy-btn"
                      onClick={() => handleCopyMarkdown(table.id, table.markdown)}
                      title="Copy Markdown Table"
                    >
                      {copiedId === table.id ? (
                        <>
                          <Check size={12} style={{ color: "var(--accent-mint)" }} />
                          <span style={{ color: "var(--accent-mint)" }}>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy size={12} />
                          <span>Copy Markdown</span>
                        </>
                      )}
                    </button>
                  </div>
                  {renderMarkdownTable(table.markdown)}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

