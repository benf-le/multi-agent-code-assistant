from langgraph.graph import END, START, StateGraph

from app.graph.nodes import WorkflowNodes
from app.graph.router import route_by_qc_result
from app.graph.state import WorkflowState


class WorkflowGraphFactory:
    """Builds two separate LangGraph graphs:

    1. **PO Phase graph** — linear pipeline, no cycles:
       START → ingest_brd → orchestrator_init → po_analyze_brd
             → po_create_user_stories → po_create_backlog_and_tasks → END

    2. **Task graph** — single-task lifecycle with bounded retry cycle:
       START → dispatch_to_dev → dev_implement → dispatch_to_qc → qc_validate
              ├─ pass       → mark_task_done      → END
              ├─ retry      → create_bug           → dispatch_to_dev  (cycle)
              ├─ max_retry  → max_retry_exceeded   → END
              └─ loop_block → max_retry_exceeded   → END
    """

    def __init__(self, nodes: WorkflowNodes):
        self.nodes = nodes

    def build_po_graph(self):
        """Build the PO analysis phase graph (linear, no cycles)."""
        graph = StateGraph(WorkflowState)
        graph.add_node('ingest_brd', self.nodes.ingest_brd)
        graph.add_node('orchestrator_init', self.nodes.orchestrator_init)
        graph.add_node('po_analyze_brd', self.nodes.po_analyze_brd)
        graph.add_node('po_create_user_stories', self.nodes.po_create_user_stories)
        graph.add_node('po_create_backlog_and_tasks', self.nodes.po_create_backlog_and_tasks)

        graph.add_edge(START, 'ingest_brd')
        graph.add_edge('ingest_brd', 'orchestrator_init')
        graph.add_edge('orchestrator_init', 'po_analyze_brd')
        graph.add_edge('po_analyze_brd', 'po_create_user_stories')
        graph.add_edge('po_create_user_stories', 'po_create_backlog_and_tasks')
        graph.add_edge('po_create_backlog_and_tasks', END)
        return graph.compile()

    def build_task_graph(self):
        """Build the single-task execution graph (bounded retry cycle).

        The only cycle is the DEV→QC→bug→DEV retry loop, bounded by
        max_retry and loop detection.  Every path leads to END.
        """
        graph = StateGraph(WorkflowState)
        graph.add_node('dispatch_to_dev', self.nodes.dispatch_to_dev)
        graph.add_node('dev_implement', self.nodes.dev_implement)
        graph.add_node('dispatch_to_qc', self.nodes.dispatch_to_qc)
        graph.add_node('qc_validate', self.nodes.qc_validate)
        graph.add_node('create_bug', self.nodes.create_bug)
        graph.add_node('mark_task_done', self.nodes.mark_task_done)
        graph.add_node('max_retry_exceeded', self.nodes.max_retry_exceeded)

        graph.add_edge(START, 'dispatch_to_dev')
        graph.add_edge('dispatch_to_dev', 'dev_implement')
        graph.add_edge('dev_implement', 'dispatch_to_qc')
        graph.add_edge('dispatch_to_qc', 'qc_validate')

        graph.add_conditional_edges(
            'qc_validate',
            route_by_qc_result,
            {
                'pass': 'mark_task_done',
                'retry': 'create_bug',
                'max_retry': 'max_retry_exceeded',
                'loop_block': 'max_retry_exceeded',
            },
        )
        graph.add_edge('create_bug', 'dispatch_to_dev')

        # Terminal nodes → END (no more looping back for next task)
        graph.add_edge('mark_task_done', END)
        graph.add_edge('max_retry_exceeded', END)

        return graph.compile()

    def build(self):
        """Legacy method — returns the task graph for backward compatibility.

        New code should use build_po_graph() or build_task_graph() directly.
        """
        return self.build_task_graph()
