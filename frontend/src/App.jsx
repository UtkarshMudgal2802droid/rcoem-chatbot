import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

function App() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const generateResponse = async () => {
    if (!prompt.trim()) return;
    
    setIsLoading(true);
    setResponse(''); 
    
    try {
      const res = await axios.post(`${API_BASE_URL}/generate`, {
        prompt: prompt,
      });
      
      const newResponse = res.data.response;
      setResponse(newResponse);
      
      setHistory(prev => [{ prompt, response: newResponse }, ...prev]);
      
    } catch (error) {
      console.error("Error generating response:", error);
      setResponse("Error: Could not connect to the RBU AI backend.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="university-shell">
      <header className="header">
        <div className="logo-area">
          <div className="logo-box">RBU</div>
          <div className="logo-text">
            <h1>RAMDEOBABA UNIVERSITY, NAGPUR</h1>
            <p>Learn | Innovate | Accomplish</p>
          </div>
        </div>
        <div className="status-badge">
          <span className="dot"></span> Online
        </div>
      </header>

      <div className="sub-nav">
        <span>Institute ▾</span>
        <span>Personal ▾</span>
        <span>Academic Schedules ▾</span>
        <span>Academic Functions ▾</span>
        <span>Events ▾</span>
        <span>Facilities ▾</span>
      </div>

      <main className="main-content">
        <div className="chat-container">
          <div className="chips">
            <button className="chip-btn chip-red" onClick={() => setPrompt("What are the latest announcements?")}>
              <div className="chip-value">0</div>
              <div className="chip-label">Announcements</div>
            </button>
            <button className="chip-btn chip-gold" onClick={() => setPrompt("What is the attendance policy?")}>
              <div className="chip-value">84%</div>
              <div className="chip-label">Attendance</div>
            </button>
            <button className="chip-btn chip-purple" onClick={() => setPrompt("How are assessments graded?")}>
              <div className="chip-value">0</div>
              <div className="chip-label">Assessment</div>
            </button>
            <button className="chip-btn chip-green" onClick={() => setPrompt("Tell me about placement statistics.")}>
              <div className="chip-value">0</div>
              <div className="chip-label">Placement</div>
            </button>
          </div>

          <div className="input-group">
            <textarea
              placeholder="Ask a question about academics, placements, or campus..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button 
              className="ask-btn" 
              onClick={generateResponse}
              disabled={isLoading || !prompt.trim()}
            >
              {isLoading ? "Processing..." : "Ask AI"}
            </button>
          </div>

          <div className="response-box">
            <h3>AI Assistant Response</h3>
            {isLoading ? (
              <div className="loader"></div>
            ) : response ? (
              <p>{response}</p>
            ) : (
              <p className="placeholder-text">(No response generated yet)</p>
            )}
          </div>
        </div>

        <div className="history-sidebar">
          <h3>Recent Queries</h3>
          {history.length === 0 ? (
            <p className="placeholder-text">No queries today.</p>
          ) : (
            history.map((item, i) => (
              <div key={i} className="history-card">
                <strong>Q: {item.prompt}</strong>
                <p>A: {item.response.substring(0, 80)}...</p>
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
