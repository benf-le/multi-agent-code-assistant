import type { Task, TaskDetail, Workflow, WorkflowDetail, WorkflowCreateRequest, WorkflowCreateResponse } from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001/api/v1'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!response.ok) {
    let msg = `API error: ${response.status}`
    try {
      const errorData = await response.json()
      if (errorData && errorData.detail) {
        msg = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail)
      } else if (errorData && errorData.message) {
        msg = errorData.message
      }
    } catch {
      // ignore parse error
    }
    throw new Error(msg)
  }
  return response.json() as Promise<T>
}


export const api = {
  listWorkflows: () => request<Workflow[]>('/workflows'),
  getWorkflow: (workflowId: number) => request<WorkflowDetail>(`/workflows/${workflowId}`),
  deleteWorkflow: (workflowId: number) => request(`/workflows/${workflowId}`, { method: 'DELETE' }),
  triggerWorkflow: (workflowId: number) => request(`/workflows/${workflowId}/run`, { method: 'POST' }),
  resumeWorkflow: (workflowId: number) => request(`/workflows/${workflowId}/resume`, { method: 'POST' }),
  stopWorkflow: (workflowId: number) => request(`/workflows/${workflowId}/stop`, { method: 'POST' }),
  listTasks: (params?: { status?: string; assignee_team?: string; retry_count_gte?: number }) => {
    const search = new URLSearchParams()
    if (params?.status) search.set('status', params.status)
    if (params?.assignee_team) search.set('assignee_team', params.assignee_team)
    if (params?.retry_count_gte !== undefined) search.set('retry_count_gte', String(params.retry_count_gte))
    const query = search.toString()
    return request<Task[]>(`/tasks${query ? `?${query}` : ''}`)
  },
  getTask: (taskId: number) => request<TaskDetail>(`/tasks/${taskId}`),
  createWorkflow: (data: WorkflowCreateRequest) => request<WorkflowCreateResponse>('/workflows', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
}
