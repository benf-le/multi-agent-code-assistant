export type Workflow = {
  id: number
  brd_id: number
  status: string
  current_agent: string | null
  max_retry: number
  metadata_json: Record<string, unknown>
  started_at: string | null
  ended_at: string | null
  created_at: string
  updated_at: string
}

export type Task = {
  id: number
  workflow_id: number
  task_number: number
  title: string
  description: string
  assignee_team: string
  status: string
  retry_count: number
  max_retry: number
  current_agent: string | null
  required_markers: string[]
  input_context: Record<string, unknown>
  output_context: Record<string, unknown>
  latest_bug_id: number | null
  created_at: string
  updated_at: string
}

export type EventLog = {
  id: number
  workflow_id: number
  task_id: number | null
  event_type: string
  agent_name: string | null
  message: string
  payload: Record<string, unknown>
  created_at: string
}

export type StateTransition = {
  id: number
  workflow_id: number
  task_id: number | null
  from_status: string | null
  to_status: string
  agent_name: string | null
  reason: string | null
  metadata_json: Record<string, unknown>
  created_at: string
}

export type WorkflowDetail = {
  workflow: Workflow
  tasks: Task[]
  transitions: StateTransition[]
  events: EventLog[]
}

export type TaskDetail = {
  task: Task
  acceptance_criteria: string[]
  bugs: Array<Record<string, unknown>>
  transitions: StateTransition[]
  events: EventLog[]
}

export type WorkflowCreateRequest = {
  title: string
  brd_content: string
  max_retry: number
}

export type WorkflowCreateResponse = {
  workflow_id: number
  message: string
}
