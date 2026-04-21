import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import { WorkflowList } from './components/WorkflowList'
import { BRDUploader } from './components/BRDUploader'
import type { Task, Workflow, WorkflowDetail, EventLog } from './types'

const POLL_MS = Number(import.meta.env.VITE_POLL_INTERVAL_MS) || 3000

export default function App() {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [selectedWorkflowId, setSelectedWorkflowId] = useState<number | null>(null)
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowDetail | null>(null)

  async function loadWorkflows() {
    try {
      const data = await api.listWorkflows()
      setWorkflows(data || [])
      if (!selectedWorkflowId && data && data.length > 0) setSelectedWorkflowId(data[0].id)
    } catch (e) {
      console.error('Failed to load workflows:', e)
    }
  }

  async function handleWorkflowCreated(id: number) {
    await loadWorkflows()
    setSelectedWorkflowId(id)
  }

  async function loadWorkflow(id: number) {
    try {
      const detail = await api.getWorkflow(id)
      setSelectedWorkflow(detail)
    } catch (e) {
      console.error('Failed to load workflow detail:', e)
    }
  }

  async function handleDeleteWorkflow(id: number) {
    try {
      await api.deleteWorkflow(id)
      await loadWorkflows()
      if (selectedWorkflowId === id) {
        setSelectedWorkflowId(null)
        setSelectedWorkflow(null)
      }
    } catch (error) {
      console.error('Failed to delete workflow:', error)
      alert('Failed to delete workflow')
    }
  }

  useEffect(() => { loadWorkflows() }, [])
  useEffect(() => { if (selectedWorkflowId) loadWorkflow(selectedWorkflowId) }, [selectedWorkflowId])

  useEffect(() => {
    const timer = setInterval(() => {
      loadWorkflows()
      if (selectedWorkflowId) loadWorkflow(selectedWorkflowId)
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [selectedWorkflowId, POLL_MS])

  const tasks = useMemo(() => selectedWorkflow?.tasks || [], [selectedWorkflow])
  const events = useMemo(() => selectedWorkflow?.events || [], [selectedWorkflow])

  const stats = useMemo(() => {
    let running = 0, done = 0, pending = 0, blocked = 0
    tasks.forEach(t => {
      const s = (t.status || 'NEW').toLowerCase()
      if (s.includes('running') || s.includes('progress') || s.includes('started') || s === 'analyzing') running++
      else if (s === 'done' || s.includes('passed') || s === 'finished' || s === 'completed') done++
      else if (s.includes('failed') || s.includes('blocked') || s.includes('error') || s.includes('exceeded')) blocked++
      else pending++
    })
    return { total: tasks.length, running, done, pending, blocked }
  }, [tasks])

  const activeAgents = useMemo(() => {
    const agents = new Set<string>()
    tasks.forEach(t => {
      const s = (t.status || '').toLowerCase()
      if ((s.includes('in_progress') || s === 'running') && t.current_agent) {
        agents.add(t.current_agent)
      }
    })
    return Array.from(agents)
  }, [tasks])

  const mailboxEvents = useMemo(() => events.filter(e => (e.message || '').includes('[MAIL]') || e.event_type === 'mail' || (e.message || '').includes('->')), [events])
  const resultEvents = useMemo(() => events.filter(e => (e.message || '').toLowerCase().includes('pass') || (e.message || '').toLowerCase().includes('fail') || e.event_type.includes('result') || e.event_type.includes('qc')), [events])
  const feedEvents = events

  function formatTime(iso: string | null | undefined) {
    if (!iso) return '--:--:--'
    try {
      return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    } catch {
      return '--:--:--'
    }
  }

  return (
    <div className="dark-app">
      <div className="control-header">
        <WorkflowList
          workflows={workflows}
          selectedWorkflowId={selectedWorkflowId}
          onSelect={setSelectedWorkflowId}
          onRun={(workflowId) => api.triggerWorkflow(workflowId).then(() => loadWorkflow(workflowId))}
          onStop={(workflowId) => api.stopWorkflow(workflowId).then(() => loadWorkflow(workflowId))}
          onDelete={handleDeleteWorkflow}
        />
        <div style={{ flex: 1, minWidth: '300px' }}>
          <BRDUploader onSuccess={handleWorkflowCreated} />
        </div>
      </div>

      <div className="dashboard">
        <div className="dashboard-topbar">
          <div className="stats-badges">
            <span className="stat-badge"><span className="count">{stats.total}</span> Total</span>
            <span className="stat-badge running"><span className="count">{stats.running}</span> Running</span>
            <span className="stat-badge done"><span className="count">{stats.done}</span> Done</span>
            <span className="stat-badge pending"><span className="count">{stats.pending}</span> Pending</span>
            <span className="stat-badge blocked"><span className="count">{stats.blocked}</span> Blocked</span>
          </div>
          <div className="status-indicator">
            {activeAgents.length > 0 ? <span className="running">Processing...</span> : <span className="idle">Idle</span>}
          </div>
        </div>

        <div className="dashboard-grid">
          <div className="col task-board">
            <h2 className="col-header">TASK BOARD</h2>
            <div className="col-content">
              {tasks.map(t => (
                <div key={t.id} className={`card ${t.status}`}>
                  <div className="card-header">
                    <span className="task-id">t-{(t.task_number || 0).toString().padStart(3, '0')}</span>
                    <span className="task-status">{t.status.replace(/_/g, ' ')}</span>
                  </div>
                  <div className="card-title">{t.title}</div>
                  <div className="task-tags">
                    {t.assignee_team && <span className={`tag ${t.assignee_team.toLowerCase()}`}>{t.assignee_team}</span>}
                  </div>
                </div>
              ))}
              {tasks.length === 0 && <div style={{ color: '#64748b', fontSize: 13 }}>No tasks yet.</div>}
            </div>
          </div>

          <div className="col activity-feed">
            <h2 className="col-header">ACTIVITY FEED</h2>
            <div className="col-content">
              {feedEvents.map(e => (
                <div key={e.id} className="feed-item">
                  <span className="time">{formatTime(e.created_at)}</span>
                  <span className={`agent-badge ${e.agent_name?.toLowerCase()}`}>{e.agent_name || 'system'}</span>
                  <span className="message">{e.message}</span>
                </div>
              ))}
              {feedEvents.length === 0 && <div style={{ color: '#64748b', fontSize: 13 }}>No events yet.</div>}
            </div>
          </div>

          <div className="col middle-right">
            <div className="sub-section">
              <h2 className="col-header">ACTIVE AGENTS</h2>
              <div className="col-content centered">
                {activeAgents.length === 0 ? (
                  <div className="idle-state">
                    <div className="zzz">zZz</div>
                    <div>All Idle</div>
                  </div>
                ) : (
                  activeAgents.map(a => <div key={a} className="agent-active">{a}</div>)
                )}
              </div>
            </div>
            <div className="sub-section">
              <h2 className="col-header">RESULTS</h2>
              <div className="col-content">
                {resultEvents.map(e => (
                  <div key={`res-${e.id}`} className="card result-card">
                    <div className="card-header">
                      <span className={`agent-badge ${e.agent_name?.toLowerCase()}`}>{e.agent_name}</span>
                    </div>
                    <div className="card-body">{e.message}</div>
                  </div>
                ))}
                {resultEvents.length === 0 && <div style={{ color: '#64748b', fontSize: 13 }}>No results yet.</div>}
              </div>
            </div>
          </div>

          <div className="col mailbox-sec">
            <h2 className="col-header">MAILBOX</h2>
            <div className="col-content">
              {mailboxEvents.map(e => (
                <div key={`mail-${e.id}`} className="card mail-card">
                  <div className="card-header">
                    <span className={`agent-badge ${e.agent_name?.toLowerCase()}`}>{e.agent_name}</span>
                  </div>
                  <div className="card-body">{e.message}</div>
                </div>
              ))}
              {mailboxEvents.length === 0 && <div style={{ color: '#64748b', fontSize: 13 }}>No messages yet.</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
