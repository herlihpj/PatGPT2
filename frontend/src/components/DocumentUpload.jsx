import React, { useState } from 'react'

function DocumentUpload({ collection, onUploadSuccess }) {
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    setUploading(true)
    setMessage('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(
        `http://localhost:8000/upload?collection=${collection}`,
        {
          method: 'POST',
          body: formData
        }
      )

      const data = await response.json()

      if (data.success) {
        setMessage(`✓ Uploaded ${data.filename} (${data.chunks} chunks)`)
        onUploadSuccess()
      } else {
        setMessage(`✗ Error: ${data.error}`)
      }
    } catch (error) {
      setMessage(`✗ Upload failed: ${error.message}`)
    }

    setUploading(false)
    event.target.value = '' // Reset input
  }

  return (
    <div className="upload-section">
      <h3>Upload Document</h3>
      <label className="upload-button">
        {uploading ? 'Uploading...' : 'Choose File'}
        <input
          type="file"
          onChange={handleFileUpload}
          disabled={uploading}
          accept=".pdf,.txt,.docx,.md"
        />
      </label>
      {message && (
        <div className={`upload-message ${message.startsWith('✓') ? 'success' : 'error'}`}>
          {message}
        </div>
      )}
      <div className="supported-formats">
        Supports: PDF, TXT, DOCX, MD
      </div>
    </div>
  )
}

export default DocumentUpload