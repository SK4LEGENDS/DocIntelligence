import React, { useEffect, useRef, useState } from "react";
import { Send, Bot, User, Sparkles, ChevronDown, ChevronUp, FileSpreadsheet } from "lucide-react";

export default function ChatArea({
  activePdf,
  messages,
  onSendMessage,
  models = [],
  selectedModel,
  setSelectedModel,
  onCitationClick,
  streamingMessage,
  loading
}) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const [thoughtsExpanded, setThoughtsExpanded] = useState(true);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    onSendMessage(input);
    setInput("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Custom JSX renderer for simple markdown (paragraphs, code blocks, bold, citations)
  const formatMessageText = (text) => {
    if (!text) return null;

    // Split text by code blocks first
    const parts = text.split(/(```[\s\S]*?```)/g);
    return parts.map((part, idx) => {
      if (part.startsWith("```") && part.endsWith("```")) {
        const code = part.slice(3, -3).trim();
        return (
          <pre key={idx}>
            <code>{code}</code>
          </pre>
        );
      }

      // Format line breaks, bold elements, and citation pills
      return (
        <React.Fragment key={idx}>
          {part.split("\n").map((line, lIdx) => {
            if (!line.trim()) {
              return <div key={lIdx} style={{ height: "0.4rem" }} />;
            }

            // Identify page citations: [Page N]
            const segments = line.split(/(\[Page\s+\d+\])/g);
            return (
              <p key={lIdx}>
                {segments.map((seg, sIdx) => {
                  const match = seg.match(/\[Page\s+(\d+)\]/);
                  if (match) {
                    const pageNum = parseInt(match[1], 10);
                    let snippet = "";
                    if (sIdx > 0) {
                      const precedingText = segments[sIdx - 1];
                      const sentences = precedingText.split(/[.!?]/);
                      const lastSentence = sentences[sentences.length - 1].trim();
                      snippet = lastSentence.replace(/\*\*/g, "").trim();
                    }
                    return (
                      <button
                        key={sIdx}
                        type="button"
                        className="citation-pill"
                        onClick={() => onCitationClick(pageNum, snippet)}
                      >
                        [Page {pageNum}]
                      </button>
                    );
                  }

                  // Handle bold segments **text**
                  const boldSegments = seg.split(/(\*\*.*?\*\*)/g);
                  return boldSegments.map((bSeg, bIdx) => {
                    if (bSeg.startsWith("**") && bSeg.endsWith("**")) {
                      return <strong key={bIdx}>{bSeg.slice(2, -2)}</strong>;
                    }
                    return bSeg;
                  });
                })}
              </p>
            );
          })}
        </React.Fragment>
      );
    });
  };

  if (!activePdf) {
    return (
      <div className="chat-area">
        <div className="empty-chat">
          <Bot size={48} style={{ color: "var(--accent-purple)", opacity: 0.3 }} />
          <h4>Document Assistant</h4>
          <p>Please select a document from the sidebar to start a conversation.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-area">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-info">
          <h3>Chat with Document</h3>
          <p>Powered by local Ollama Agent</p>
        </div>
        <div className="model-selector">
          <span>Model:</span>
          <select
            className="model-select"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {models.length > 0 ? (
              models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))
            ) : (
              <>
                <option value="qwen2.5:7b">qwen2.5:7b</option>
                <option value="qwen3:8b">qwen3:8b</option>
              </>
            )}
          </select>
        </div>
      </header>

      {/* Messages */}
      <div className="chat-messages-container">
        {messages.length === 0 && !streamingMessage && (
          <div className="empty-chat">
            <Sparkles size={32} style={{ color: "var(--accent-purple)" }} />
            <h4>Ask anything!</h4>
            <p>Ask questions about compliance chapters, layouts, tables, or request summaries.</p>
          </div>
        )}

        {/* Render Historical Messages */}
        {messages.map((msg) => (
          <div key={msg.id} className={`message-row ${msg.role}`}>
            <div className="message-bubble">
              {formatMessageText(msg.content)}
            </div>
          </div>
        ))}

        {/* Render Active Streaming Message */}
        {streamingMessage && (
          <div className="message-row assistant">
            <div className="message-bubble" style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {/* Thought Process Accordion */}
              {streamingMessage.thoughts && streamingMessage.thoughts.length > 0 && (
                <div className="thought-container">
                  <div
                    className="thought-header"
                    onClick={() => setThoughtsExpanded(!thoughtsExpanded)}
                  >
                    <span>Agent Thought Process</span>
                    {thoughtsExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  </div>
                  {thoughtsExpanded && (
                    <div className="thought-list">
                      {streamingMessage.thoughts.map((thought, idx) => {
                        const isLast = idx === streamingMessage.thoughts.length - 1;
                        return (
                          <div
                            key={idx}
                            className={`thought-item ${isLast && loading ? "active" : ""}`}
                          >
                            <span>➔</span>
                            <span>{thought}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* Streaming Content */}
              {streamingMessage.content && (
                <div className="streaming-content-text">
                  {formatMessageText(streamingMessage.content)}
                </div>
              )}

              {/* Source Highlights */}
              {streamingMessage.sources && streamingMessage.sources.length > 0 && (
                <div className="sources-section">
                  <h4 className="sources-title">
                    <FileSpreadsheet size={12} />
                    Grounded Source References
                  </h4>
                  <div className="sources-list">
                    {streamingMessage.sources.map((source, idx) => (
                      <div
                        key={idx}
                        className="source-item"
                        onClick={() => onCitationClick(source.page_number, source.content.slice(0, 50))}
                      >
                        <div className="source-item-header">
                          <span>Page {source.page_number}</span>
                          <span>{source.heading}</span>
                        </div>
                        <div className="source-item-content" title={source.content}>
                          {source.content}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <textarea
            className="chat-textarea"
            placeholder="Ask a question about this document..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={1}
          />
          <button
            type="submit"
            className="chat-send-btn"
            disabled={!input.trim() || loading}
          >
            <Send size={16} />
          </button>
        </form>
      </div>
    </div>
  );
}
