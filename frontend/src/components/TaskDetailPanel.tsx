import type { TaskDetail } from '../types'
import { StatusBadge } from './StatusBadge'

export function TaskDetailPanel({ detail }: { detail: TaskDetail | null }) {
  if (!detail) {
    return <div className="panel"><div className="panel-header"><h2>Task Detail</h2></div><p>Select a task to inspect details.</p></div>
  }

  return (
    <div className="panel">
      <div className="panel-header"><h2>Task Detail</h2></div>
      <div className="detail-grid">
        <div><strong>Task ID:</strong> t-{String(detail.task.task_number).padStart(3, '0')}</div>
        <div><strong>Title:</strong> {detail.task.title}</div>
        <div><strong>Status:</strong> <StatusBadge status={detail.task.status} /></div>
        <div><strong>Current Agent:</strong> {detail.task.current_agent || '-'}</div>
        <div><strong>Retry:</strong> {detail.task.retry_count}/{detail.task.max_retry}</div>
        <div><strong>Team:</strong> {detail.task.assignee_team}</div>
      </div>
      <h3>Acceptance Criteria</h3>
      <ul>{detail.acceptance_criteria.map((item, idx) => <li key={idx}>{item}</li>)}</ul>
      <h3>Bug Reports</h3>
      <ul>{detail.bugs.length === 0 ? <li>No bugs</li> : detail.bugs.map((bug) => <li key={String(bug.id)}>{String(bug.title)} - {String(bug.severity)}</li>)}</ul>
      <h3>Output</h3>
      <pre>{JSON.stringify(detail.task.output_context, null, 2)}</pre>
    </div>
  )
}
