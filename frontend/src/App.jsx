import React, { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import DocumentUpload from './components/DocumentUpload'

function App() {
  const [collection, setCollection] = useState('default')
  const [documents, setDocuments] = useState([])
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1)
  }

  return (
    <div className="app">
      <div className="sidebar">
        <h1>🤖 PatGPT</h1>
        
        <div className="collection-section">
          <label>Collection:</label>
          <input
            type="text"
            value={collection}
            onChange={(e) => setCollection(e.target.value)}
            placeholder="default"
          />
        </div>

        <DocumentUpload 
          collection={collection}
          onUploadSuccess={handleUploadSuccess}
        />

        <div className="documents-section">
          <h3>Documents</h3>
          <DocumentList 
            collection={collection} 
            refreshTrigger={refreshTrigger}
            onDocumentsChange={setDocuments}
          />
        </div>

        <div className="info-section">
          <h4>Status</h4>
          <StatusIndicator />
        </div>
      </div>

      <div className="main-content">
        <ChatInterface collection={collection} />
      </div>
    </div>
  )
}

function DocumentList({ collection, refreshTrigger, onDocumentsChange }) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)

  React.useEffect(() => {
    fetchDocuments()
  }, [collection, refreshTrigger])

  const fetchDocuments = async () => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:8000/documents/${collection}`)
      const data = await response.json()
      setDocs(data.documents || [])
      onDocumentsChange(data.documents || [])
    } catch (error) {
      console.error('Error fetching documents:', error)
    }
    setLoading(false)
  }

  if (loading) return <div className="loading">Loading...</div>

  if (docs.length === 0) {
    return <div className="no-documents">No documents uploaded yet</div>
  }

  return (
    <ul className="document-list">
      {docs.map((doc, idx) => (
        <li key={idx} className="document-item">
          📄 {doc}
        </li>
      ))}
    </ul>
  )
}

function StatusIndicator() {
  const [status, setStatus] = useState('checking')

  React.useEffect(() => {
    checkStatus()
  }, [])

  const checkStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/')
      if (response.ok) {
        setStatus('online')
      } else {
        setStatus('offline')
      }
    } catch (error) {
      setStatus('offline')
    }
  }

  return (
    <div className={`status status-${status}`}>
      <span className="status-dot"></span>
      {status === 'online' ? 'Connected' : 'Disconnected'}
    </div>
  )
}

export default App