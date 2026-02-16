import React, { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'

function MessageList({ messages }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="message-list">
      {messages.length === 0 && (
        <div className="welcome-message">
          <h2>👋 Welcome!</h2>
          <p>Upload documents and ask questions. I can:</p>
          <ul>
            <li>Answer questions from your documents</li>
            <li>Search the web for current info</li>
            <li>Perform calculations</li>
            <li>Help with various tasks</li>
          </ul>
        </div>
      )}
      
      {messages.map((message, index) => (
        <div key={index} className={`message message-${message.role}`}>
          <div className="message-content">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
          
          {message.sources && message.sources.length > 0 && (
            <div className="message-sources">
              <strong>Sources:</strong>
              {message.sources.map((source, idx) => (
                <div key={idx} className="source-item">
                  📄 {source.source}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

export default MessageList