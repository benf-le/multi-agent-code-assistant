import type { Task } from '../types'
import { StatusBadge } from './StatusBadge'

export function TaskTable({ tasks, onSelect }: { tasks: Task[]; onSelect: (taskId: number) => void }) {
  return (
    <div className="panel">
      <div className="panel-header"><h2>Tasks</h2></div>
      <table className="table">
        <thead>
          <tr><th>ID</th><th>Title</th><th>Status</th><th>Team</th><th>Current Agent</th><th>Retry</th></tr>
        </thead>
        <tbody>
          {tasks.map((task) => (
            <tr key={task.id} onClick={() => onSelect(task.id)}>
              <td>t-{String(task.task_number).padStart(3, '0')}</td>
              <td>{task.title}</td>
              <td><StatusBadge status={task.status} /></td>
              <td>{task.assignee_team}</td>
              <td>{task.current_agent || '-'}</td>
              <td>{task.retry_count}/{task.max_retry}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
