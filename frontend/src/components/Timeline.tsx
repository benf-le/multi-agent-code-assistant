import type { EventLog } from '../types'

export function Timeline({ events }: { events: EventLog[] }) {
  return (
    <div className="panel">
      <div className="panel-header"><h2>Timeline / Event Log</h2></div>
      <div className="timeline">
        {events.map((event) => (
          <div key={event.id} className="timeline-item">
            <div className="timeline-time">{new Date(event.created_at).toLocaleString()}</div>
            <div className="timeline-content">
              <strong>{event.event_type}</strong>
              <div>{event.message}</div>
              <small>{event.agent_name || 'SYSTEM'}</small>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
