import React, { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import PdfViewer from "./components/PdfViewer";
import "./App.css";

export default function App() {
  const [pdfs, setPdfs] = useState([]);
  const [activePdfId, setActivePdfId] = useState(null);
  const [activeChatId, setActiveChatId] = useState(null);
  
  const [messages, setMessages] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("qwen2.5:7b");
  const [currentPageInfo, setCurrentPageInfo] = useState({ page: 1, snippet: "" });
  const [theme, setTheme] = useState(localStorage.getItem("theme") || "light");
  const [loading, setLoading] = useState(false);
  
  // Streaming message buffer state
  const [streamingMessage, setStreamingMessage] = useState(null);

  // Sync theme to document body
  useEffect(() => {
    if (theme === "dark") {
      document.body.classList.add("dark");
    } else {
      document.body.classList.remove("dark");
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  // Fetch available models from Ollama
  const fetchModels = async () => {
    try {
      const response = await fetch("/api/models");
      if (response.ok) {
        const data = await response.json();
        setModels(data);
        if (data.length > 0) {
          // Default to qwen2.5:7b if present, else qwen3:8b, else first available
          if (data.includes("qwen2.5:7b")) {
            setSelectedModel("qwen2.5:7b");
          } else if (data.includes("qwen3:8b")) {
            setSelectedModel("qwen3:8b");
          } else {
            setSelectedModel(data[0]);
          }
        }
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  };

  // Fetch PDFs with automatic retry (resilient to backend startup latency)
  const fetchPdfs = async (retries = 8, delay = 2500) => {
    try {
      const response = await fetch("/api/pdfs");
      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      const data = await response.json();
      setPdfs(data);
    } catch (err) {
      console.warn(`Fetch PDFs failed. Retrying in ${delay}ms... (${retries} retries left). Error: ${err.message}`);
      if (retries > 0) {
        setTimeout(() => fetchPdfs(retries - 1, delay), delay);
      }
    }
  };

  useEffect(() => {
    fetchPdfs();
    fetchModels();
  }, []);

  // Poll PDF list if any document is currently parsing
  useEffect(() => {
    const hasParsing = pdfs.some((pdf) => pdf.status && (pdf.status === "parsing" || pdf.status.startsWith("parsing")));
    if (!hasParsing) return;

    const interval = setInterval(() => {
      fetchPdfs();
    }, 3000);

    return () => clearInterval(interval);
  }, [pdfs]);

  // React hook to load or create a chat session as soon as the active PDF is ready
  useEffect(() => {
    if (!activePdfId) {
      setActiveChatId(null);
      setMessages([]);
      return;
    }
    
    const activePdfObj = pdfs.find((p) => p.id === activePdfId);
    if (!activePdfObj || activePdfObj.status !== "ready") {
      // PDF is not ready yet (e.g. parsing)
      return;
    }

    const loadOrCreateChat = async () => {
      try {
        // 1. Check if there are any chats in the system
        const chatsRes = await fetch("/api/chats");
        let chatId = null;
        
        if (chatsRes.ok) {
          const chatsList = await chatsRes.json();
          // Look for chat linked to this pdf
          const existingChat = chatsList.find((c) => c.pdf_id === activePdfId);
          if (existingChat) {
            chatId = existingChat.id;
          }
        }
        
        // 2. If no chat exists, create one
        if (!chatId) {
          const title = `Chat about ${activePdfObj.filename || "Document"}`;
          const createForm = new FormData();
          createForm.append("pdf_id", activePdfId);
          createForm.append("title", title);
          
          const createRes = await fetch("/api/chats", {
            method: "POST",
            body: createForm
          });
          
          if (createRes.ok) {
            const chatData = await createRes.json();
            chatId = chatData.id;
          }
        }
        
        setActiveChatId(chatId);
        
        // 3. Load chat history
        if (chatId) {
          const messagesRes = await fetch(`/api/chats/${chatId}/messages`);
          if (messagesRes.ok) {
            const msgs = await messagesRes.json();
            setMessages(msgs);
          }
        }
      } catch (err) {
        console.error("Error setting active chat:", err);
      }
    };

    loadOrCreateChat();
  }, [activePdfId, pdfs]);

  const activePdf = pdfs.find((pdf) => pdf.id === activePdfId);

  // Select a PDF and clear active messages/page (chat loading is handled reactively by useEffect)
  const handleSelectPdf = (pdfId) => {
    setActivePdfId(pdfId);
    setMessages([]);
    setStreamingMessage(null);
    setCurrentPageInfo({ page: 1, snippet: "" });
  };

  const handleUploadSuccess = (data) => {
    fetchPdfs();
    // Auto-select PDF once it uploads (will show "parsing" state first)
    setActivePdfId(data.id);
  };

  const handleDeletePdf = async (pdfId) => {
    if (!window.confirm("Are you sure you want to delete this document? All chats and vector embeddings will be permanently removed.")) {
      return;
    }
    
    try {
      const res = await fetch(`/api/pdfs/${pdfId}`, {
        method: "DELETE"
      });
      if (res.ok) {
        if (activePdfId === pdfId) {
          setActivePdfId(null);
          setActiveChatId(null);
          setMessages([]);
          setStreamingMessage(null);
          setCurrentPageInfo({ page: 1, snippet: "" });
        }
        fetchPdfs();
      }
    } catch (err) {
      console.error("Error deleting PDF:", err);
    }
  };

  // SSE Stream Message Handler
  const handleSendMessage = async (text) => {
    if (!activeChatId) return;

    setLoading(true);
    
    // Add user message locally
    const userMsg = { id: Date.now(), role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Initialize streaming message structure
    setStreamingMessage({
      role: "assistant",
      content: "",
      thoughts: [],
      sources: []
    });

    try {
      const form = new FormData();
      form.append("content", text);
      form.append("model", selectedModel);

      const response = await fetch(`/api/chats/${activeChatId}/message`, {
        method: "POST",
        body: form
      });

      if (!response.ok) {
        throw new Error(`API returned error: ${response.statusText}`);
      }

      // Read SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        // Save back incomplete line
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "").trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);
              
              if (data.type === "thought") {
                setStreamingMessage((prev) => ({
                  ...prev,
                  thoughts: [...prev.thoughts, data.content]
                }));
              } else if (data.type === "token") {
                setStreamingMessage((prev) => ({
                  ...prev,
                  content: prev.content + data.content
                }));
              } else if (data.type === "sources") {
                setStreamingMessage((prev) => ({
                  ...prev,
                  sources: data.content
                }));
              }
            } catch (err) {
              console.warn("Could not parse stream JSON:", err, line);
            }
          }
        }
      }
      
      // Trigger database sync by fetching history again to get the final assistant message ID
      const messagesRes = await fetch(`/api/chats/${activeChatId}/messages`);
      if (messagesRes.ok) {
        const msgs = await messagesRes.json();
        setMessages(msgs);
      }
      setStreamingMessage(null);

    } catch (err) {
      console.error("Streaming error:", err);
      setStreamingMessage((prev) => ({
        ...prev,
        content: prev.content + `\n\n[System Connection Error: ${err.message}]`
      }));
    } finally {
      setLoading(false);
    }
  };

  const handleCitationClick = (pageNum, snippet = "") => {
    setCurrentPageInfo({ page: pageNum, snippet });
  };

  return (
    <div className="app-container">
      <Sidebar
        pdfs={pdfs}
        activePdfId={activePdfId}
        onSelectPdf={handleSelectPdf}
        onDeletePdf={handleDeletePdf}
        onUploadSuccess={handleUploadSuccess}
        theme={theme}
        onToggleTheme={() => setTheme((prev) => (prev === "light" ? "dark" : "light"))}
      />
      <ChatArea
        activePdf={activePdf}
        messages={messages}
        onSendMessage={handleSendMessage}
        models={models}
        selectedModel={selectedModel}
        setSelectedModel={setSelectedModel}
        onCitationClick={handleCitationClick}
        streamingMessage={streamingMessage}
        loading={loading}
      />
      <PdfViewer
        activePdf={activePdf}
        currentPageInfo={currentPageInfo}
      />
    </div>
  );
}
