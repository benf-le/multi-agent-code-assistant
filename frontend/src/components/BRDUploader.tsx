import React, { useState } from 'react'
import { api } from '../api'

interface BRDUploaderProps {
  onSuccess: (workflowId: number) => void
}

export const BRDUploader: React.FC<BRDUploaderProps> = ({ onSuccess }) => {
  const [title, setTitle] = useState('')
  const [maxRetry, setMaxRetry] = useState(2)
  const [brdContent, setBrdContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [fileName, setFileName] = useState('')

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setFileName(file.name)
      const reader = new FileReader()
      reader.onload = (event) => {
        const content = event.target?.result as string
        setBrdContent(content)
      }
      reader.readAsText(file)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!title || !brdContent) {
      alert('Please provide a title and upload a BRD file.')
      return
    }

    setLoading(true)
    try {
      const response = await api.createWorkflow({
        title,
        brd_content: brdContent,
        max_retry: maxRetry,
      })
      alert(response.message)
      setTitle('')
      setBrdContent('')
      setFileName('')
      onSuccess(response.workflow_id)
    } catch (error) {
      console.error(error)
      alert('Failed to create workflow.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel brd-uploader">
      <div className="panel-header">
        <h2>Create New Workflow from BRD</h2>
      </div>
      <form onSubmit={handleSubmit} className="brd-form">
        <div className="form-group">
          <label htmlFor="brd-title">Workflow Title</label>
          <input
            id="brd-title"
            type="text"
            placeholder="e.g. E-commerce Checkout System"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="brd-file">BRD File (.md)</label>
          <div className="file-input-wrapper">
            <button type="button" className="btn-secondary" onClick={() => document.getElementById('brd-file')?.click()}>
              {fileName ? `File: ${fileName}` : 'Choose Markdown File'}
            </button>
            <input
              id="brd-file"
              type="file"
              accept=".md"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="brd-retry">Max Retries</label>
          <input
            id="brd-retry"
            type="number"
            min="0"
            max="10"
            value={maxRetry}
            onChange={(e) => setMaxRetry(Number(e.target.value))}
          />
        </div>

        <button type="submit" className="btn-primary" disabled={loading || !brdContent}>
          {loading ? 'Creating...' : 'Create Workflow'}
        </button>
      </form>
    </div>
  )
}
