from langgraph.graph import END, START, StateGraph

from app.graph.nodes import WorkflowNodes
from app.graph.router import route_after_task_resolution, route_by_qc_result, route_start
from app.graph.state import WorkflowState


class WorkflowGraphFactory:
    def __init__(self, nodes: WorkflowNodes):
        self.nodes = nodes

    def build(self):
        graph = StateGraph(WorkflowState)
        graph.add_node('ingest_brd', self.nodes.ingest_brd)
        graph.add_node('orchestrator_init', self.nodes.orchestrator_init)
        graph.add_node('po_analyze_brd', self.nodes.po_analyze_brd)
        graph.add_node('po_create_user_stories', self.nodes.po_create_user_stories)
        graph.add_node('po_create_backlog_and_tasks', self.nodes.po_create_backlog_and_tasks)
        graph.add_node('dispatch_to_dev', self.nodes.dispatch_to_dev)
        graph.add_node('dev_implement', self.nodes.dev_implement)
        graph.add_node('dispatch_to_qc', self.nodes.dispatch_to_qc)
        graph.add_node('qc_validate', self.nodes.qc_validate)
        graph.add_node('create_bug', self.nodes.create_bug)
        graph.add_node('mark_task_done', self.nodes.mark_task_done)
        graph.add_node('max_retry_exceeded', self.nodes.max_retry_exceeded)
        graph.add_node('finalize_workflow', self.nodes.finalize_workflow)

        # Route from START based on current status (for resumption)
        graph.add_conditional_edges(
            START, 
            route_start, 
            {
                'ingest_brd': 'ingest_brd',
                'po_analyze_brd': 'po_analyze_brd',
                'dispatch_to_dev': 'dispatch_to_dev',
                'dev_implement': 'dev_implement',
                'dispatch_to_qc': 'dispatch_to_qc',
                'qc_validate': 'qc_validate'
            }
        )

        graph.add_edge('ingest_brd', 'orchestrator_init')
        graph.add_edge('orchestrator_init', 'po_analyze_brd')
        graph.add_edge('po_analyze_brd', 'po_create_user_stories')
        graph.add_edge('po_create_user_stories', 'po_create_backlog_and_tasks')
        graph.add_edge('po_create_backlog_and_tasks', 'dispatch_to_dev')
        graph.add_edge('dispatch_to_dev', 'dev_implement')
        graph.add_edge('dev_implement', 'dispatch_to_qc')
        graph.add_edge('dispatch_to_qc', 'qc_validate')
        graph.add_conditional_edges('qc_validate', route_by_qc_result, {'pass': 'mark_task_done', 'retry': 'create_bug', 'max_retry': 'max_retry_exceeded'})
        graph.add_edge('create_bug', 'dispatch_to_dev')
        graph.add_conditional_edges('mark_task_done', route_after_task_resolution, {'has_more': 'dispatch_to_dev', 'finished': 'finalize_workflow'})
        graph.add_conditional_edges('max_retry_exceeded', route_after_task_resolution, {'has_more': 'dispatch_to_dev', 'finished': 'finalize_workflow'})
        graph.add_edge('finalize_workflow', END)
        return graph.compile()
