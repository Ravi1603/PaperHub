import React, { useState, useEffect } from 'react';
import './ChatBox.css';

const ChatWidget = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Hello! How can I help you?' }
  ]);
  const [input, setInput] = useState('');

  useEffect(() => {
    if (!isChatOpen) {
      const timer = setTimeout(() => {
        setShowNotification(true);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [isChatOpen]);

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
    if (!isChatOpen) setShowNotification(false);
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    // Add user's message
    const newMessages = [...messages, { sender: 'user', text: input }];
    setMessages(newMessages);
    setInput('');

    try {
      const response = await fetch('http://localhost:5000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: input })
      });

      const data = await response.json();

      if (data.answer) {
        setMessages((prev) => [...prev, { sender: 'bot', text: data.answer }]);
      } else {
        setMessages((prev) => [
          ...prev,
          { sender: 'bot', text: 'Sorry, I could not understand that.' }
        ]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { sender: 'bot', text: 'There was an error contacting the assistant.' }
      ]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') handleSend();
  };

  return (
    <div className="chat-container">
      {!isChatOpen ? (
        <button className="chat-button" onClick={toggleChat}>
          Chat
          {showNotification && <span className="dot-badge"></span>}
        </button>
      ) : (
        <div className="chat-window">
          <div className="chat-header">
            <h2>Support Bot</h2>
            <button className="close-btn" onClick={toggleChat}>
              &times;
            </button>
          </div>
          <div className="chat-body">
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`chat-message ${msg.sender === 'user' ? 'user' : 'bot'}`}
              >
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chat-footer">
            <input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button onClick={handleSend}>Send</button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;
