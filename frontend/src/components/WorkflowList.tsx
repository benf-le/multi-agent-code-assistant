import type { Workflow } from '../types'
import { StatusBadge } from './StatusBadge'

export function WorkflowList({ workflows, selectedWorkflowId, onSelect, onRun, onResume, onStop, onDelete }: { workflows: Workflow[]; selectedWorkflowId: number | null; onSelect: (id: number) => void; onRun: (id: number) => void; onResume: (id: number) => void; onStop: (id: number) => void; onDelete: (id: number) => void }) {
  return (
    <div className="panel">
      <div className="panel-header"><h2>Workflow List</h2></div>
      <table className="table">
        <thead>
          <tr><th>ID</th><th>Title</th><th>Status</th><th>Current Agent</th><th>Action</th></tr>
        </thead>
        <tbody>
          {workflows.map((workflow) => (
            <tr key={workflow.id} className={selectedWorkflowId === workflow.id ? 'selected-row' : ''} onClick={() => onSelect(workflow.id)}>
              <td>{workflow.id}</td>
              <td>{String(workflow.metadata_json?.title || `Workflow ${workflow.id}`)}</td>
              <td><StatusBadge status={workflow.status} /></td>
              <td>{workflow.current_agent || '-'}</td>
              <td>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button onClick={(e) => { e.stopPropagation(); onRun(workflow.id) }} disabled={workflow.status !== 'NEW'}>Run</button>
                  {['CANCELLED', 'BLOCKED', 'MAX_RETRY_EXCEEDED', 'FAILED', 'PASSED'].includes(workflow.status) && (
                    <button
                      style={{ background: '#10b981' }}
                      onClick={(e) => { e.stopPropagation(); onResume(workflow.id) }}
                    >
                      Resume
                    </button>
                  )}
                  <button
                    className="danger-btn"
                    style={{ background: '#f59e0b' }}
                    onClick={(e) => { e.stopPropagation(); onStop(workflow.id) }}
                    disabled={!['PO_ANALYZING', 'BACKLOG_CREATED', 'TASK_READY_FOR_DEV', 'DEV_IN_PROGRESS', 'DEV_DONE', 'QC_IN_PROGRESS', 'QC_FAILED', 'BUG_CREATED', 'REOPENED_FOR_DEV'].includes(workflow.status)}
                  >
                    Stop
                  </button>
                  <button className="danger-btn" onClick={(e) => { e.stopPropagation(); if (window.confirm('Are you sure you want to delete this workflow?')) onDelete(workflow.id) }}>Delete</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
