import type { StateTransition } from '../types'

export function StepFlow({ transitions }: { transitions: StateTransition[] }) {
  return (
    <div className="panel">
      <div className="panel-header"><h2>Step Flow</h2></div>
      <div className="step-flow">
        {transitions.map((transition) => (
          <div key={transition.id} className="step-card">
            <div className="step-title">{transition.to_status}</div>
            <div className="step-meta">{transition.agent_name || 'SYSTEM'}</div>
            <div className="step-reason">{transition.reason || '-'}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
