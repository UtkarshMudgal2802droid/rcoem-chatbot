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
      setResponse("Error: Could not connect to the RCOEM AI backend.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="university-shell">
      <header className="header">
        <div className="logo-area">
          <div className="logo-box">R</div>
          <div>
            <h1>RCOEM Student Assistant</h1>
            <p>Shri Ramdeobaba College of Engineering and Management</p>
          </div>
        </div>
        <div className="status-badge">
          <span className="dot"></span> Online
        </div>
      </header>

      <main className="main-content">
        <div className="chat-container">
          <div className="examples-section">
            <p>Ask me about RCOEM:</p>
            <div className="chips">
              <button onClick={() => setPrompt("What courses are offered at RCOEM?")}>Courses</button>
              <button onClick={() => setPrompt("How are the placements at RCOEM?")}>Placements</button>
              <button onClick={() => setPrompt("Where is RCOEM located?")}>Location</button>
            </div>
          </div>

          <div className="input-group">
            <textarea
              placeholder="Ask a question about admissions, campus, or placements..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button 
              className="ask-btn" 
              onClick={generateResponse}
              disabled={isLoading || !prompt.trim()}
            >
              {isLoading ? "Thinking..." : "Ask AI"}
            </button>
          </div>

          <div className="response-box">
            <h3>AI Response:</h3>
            {isLoading ? (
              <div className="loader"></div>
            ) : response ? (
              <p>{response}</p>
            ) : (
              <p className="placeholder-text">Your answer will appear here.</p>
            )}
          </div>
        </div>

        <div className="history-sidebar">
          <h3>Recent Questions</h3>
          {history.length === 0 ? (
            <p className="empty">No questions yet.</p>
          ) : (
            history.map((item, i) => (
              <div key={i} className="history-card">
                <strong>Q: {item.prompt}</strong>
                <p>A: {item.response.substring(0, 60)}...</p>
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
