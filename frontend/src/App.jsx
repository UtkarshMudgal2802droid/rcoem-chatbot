import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'https://centaurian-tashia-dextrorse.ngrok-free.dev';

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
      const res = await axios.post(`${API_BASE_URL}/rcoem-chatbot/generate`, {
        prompt: prompt,
      });

      const newResponse = res.data.response;
      setResponse(newResponse);

      setHistory((prev) => [{ prompt, response: newResponse }, ...prev]);
    } catch (error) {
      console.error('Error generating response:', error);
      setResponse('Error: Could not connect to the RCOEM AI backend.');
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
          <span className="dot"></span> System Online
        </div>
      </header>

      <main className="main-content">
        <div className="chat-container">
          <div className="glass-panel">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <textarea
                placeholder="Ask about RCOEM canteens, buildings, placements, or academic policies..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <button
                className="ask-btn"
                onClick={generateResponse}
                disabled={isLoading || !prompt.trim()}
              >
                {isLoading ? 'Processing...' : 'Ask RBU AI'}
              </button>
            </div>
          </div>

          <div className="glass-panel response-box">
            <h3>✦ AI Response</h3>
            {isLoading ? (
              <div className="loader"></div>
            ) : response ? (
              <p>{response}</p>
            ) : (
              <p className="placeholder-text">Enter a query above to see the AI response.</p>
            )}
          </div>
        </div>

        <div className="glass-panel history-sidebar">
          <h3>Recent Queries</h3>
          {history.length === 0 ? (
            <p className="placeholder-text">No queries today.</p>
          ) : (
            history.map((item, i) => (
              <div key={i} className="history-card">
                <strong>Q: {item.prompt}</strong>
                <p>{item.response.substring(0, 80)}...</p>
              </div>
            ))
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
