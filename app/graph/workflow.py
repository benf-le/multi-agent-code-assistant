from langgraph.graph import END, START, StateGraph

from app.graph.nodes import WorkflowNodes
from app.graph.router import route_by_qc_result, route_by_po_review, route_by_build_result, route_by_dependency_result, route_by_final_qa_result
from app.graph.state import WorkflowState
from app.core.logging_helper import WorkflowLogger


class WorkflowGraphFactory:
    """Builds two separate LangGraph graphs:

    1. **PO Phase graph** — pipeline with review gate:
       START → ingest_brd → orchestrator_init → po_analyze_brd
             → po_create_user_stories → po_create_backlog_and_tasks
             → po_review
             ├─ pass  → END
             ├─ retry → po_analyze_brd  (bounded cycle)
             └─ fail  → po_review_failed → END

    2. **Task graph** — single-task lifecycle with bounded retry cycle:
       START → dispatch_to_dev → dev_implement → ensure_dependencies
              ├─ skip_gate  → dispatch_to_qc
              ├─ pass       → build_candidate
              │      ├─ pass       → dispatch_to_qc → qc_validate
              │      │      ├─ pass       → mark_task_done      → END
              │      │      ├─ retry      → create_bug           → dispatch_to_dev  (cycle)
              │      │      ├─ max_retry  → max_retry_exceeded   → END
              │      │      └─ loop_block → max_retry_exceeded   → END
              │      ├─ retry      → create_build_bug     → dispatch_to_dev  (cycle)
              │      └─ max_retry  → max_retry_exceeded   → END
              ├─ retry      → create_dependency_bug → dispatch_to_dev (cycle)
              └─ max_retry  → max_retry_exceeded   → END
    """

    def __init__(self, nodes: WorkflowNodes):
        self.nodes = nodes

    def build_po_graph(self):
        """Build the PO analysis phase graph with review gate.

        Flow:
        START → ingest_brd → orchestrator_init → po_analyze_brd
              → po_create_user_stories → po_create_backlog_and_tasks
              → po_local_validate
              ├─ pass  → po_review
              ├─ retry → po_analyze_brd
              └─ fail  → po_review_failed
        po_review
              ├─ pass  → END
              ├─ retry → po_analyze_brd
              └─ fail  → po_review_failed
        """
        graph = StateGraph(WorkflowState)
        graph.add_node('ingest_brd', self.nodes.ingest_brd)
        graph.add_node('orchestrator_init', self.nodes.orchestrator_init)
        graph.add_node('po_analyze_brd', self.nodes.po_analyze_brd)
        graph.add_node('po_create_user_stories', self.nodes.po_create_user_stories)
        graph.add_node('po_create_backlog_and_tasks', self.nodes.po_create_backlog_and_tasks)
        graph.add_node('po_local_validate', self.nodes.po_local_validate)
        graph.add_node('po_review', self.nodes.po_review)
        graph.add_node('po_review_failed', self.nodes.po_review_failed)

        graph.add_edge(START, 'ingest_brd')
        graph.add_edge('ingest_brd', 'orchestrator_init')
        graph.add_edge('orchestrator_init', 'po_analyze_brd')
        graph.add_edge('po_analyze_brd', 'po_create_user_stories')
        graph.add_edge('po_create_user_stories', 'po_create_backlog_and_tasks')
        graph.add_edge('po_create_backlog_and_tasks', 'po_local_validate')

        # Conditional routing from po_local_validate
        from app.graph.router import route_by_po_local_validate
        graph.add_conditional_edges(
            'po_local_validate',
            route_by_po_local_validate,
            {
                'pass': 'po_review',
                'retry': 'po_analyze_brd',
                'fail': 'po_review_failed',
            },
        )

        # Conditional routing from po_review
        graph.add_conditional_edges(
            'po_review',
            route_by_po_review,
            {
                'pass': END,
                'retry': 'po_analyze_brd',
                'fail': 'po_review_failed',
            },
        )
        graph.add_edge('po_review_failed', END)

        compiled = graph.compile()
        self._log_topology("po_graph", compiled)
        return compiled

    def _log_topology(self, name: str, compiled_graph):
        """Log a summary of the graph topology."""
        try:
            nodes = list(compiled_graph.nodes.keys())
            # Simplistic edge extraction from internal structure if possible, 
            # or just log the intent since we know the structure.
            # LangGraph doesn't make it easy to traverse edges in a standard way without deeper inspection.
            # But we can log the known structure from the factory methods.
            WorkflowLogger.info("workflow.graph.static_view", 
                graph_name=name,
                nodes=nodes,
                message=f"Graph '{name}' built with {len(nodes)} nodes."
            )
        except Exception as e:
            WorkflowLogger.warning("workflow.graph.static_view_failed", graph_name=name, error=str(e))

    def build_task_graph(self):
        """Build the single-task execution graph (bounded retry cycle).

        The only cycle is the DEV→QC→bug→DEV retry loop, bounded by
        max_retry and loop detection.  Every path leads to END.
        """
        graph = StateGraph(WorkflowState)
        graph.add_node('dispatch_to_dev', self.nodes.dispatch_to_dev)
        graph.add_node('dev_implement', self.nodes.dev_implement)
        graph.add_node('ensure_dependencies', self.nodes.ensure_dependencies)
        graph.add_node('create_dependency_bug', self.nodes.create_dependency_bug)
        graph.add_node('build_candidate', self.nodes.build_candidate)
        graph.add_node('create_build_bug', self.nodes.create_build_bug)
        graph.add_node('dispatch_to_qc', self.nodes.dispatch_to_qc)
        graph.add_node('qc_validate', self.nodes.qc_validate)
        graph.add_node('create_bug', self.nodes.create_bug)
        graph.add_node('mark_task_done', self.nodes.mark_task_done)
        graph.add_node('max_retry_exceeded', self.nodes.max_retry_exceeded)

        graph.add_edge(START, 'dispatch_to_dev')
        graph.add_edge('dispatch_to_dev', 'dev_implement')
        graph.add_edge('dev_implement', 'ensure_dependencies')

        graph.add_conditional_edges(
            'ensure_dependencies',
            route_by_dependency_result,
            {
                'pass': 'build_candidate',
                'skip_gate': 'dispatch_to_qc',
                'retry': 'create_dependency_bug',
                'max_retry': 'max_retry_exceeded',
            },
        )
        graph.add_edge('create_dependency_bug', 'dispatch_to_dev')

        graph.add_conditional_edges(
            'build_candidate',
            route_by_build_result,
            {
                'pass': 'dispatch_to_qc',
                'retry': 'create_build_bug',
                'max_retry': 'max_retry_exceeded',
            },
        )
        graph.add_edge('create_build_bug', 'dispatch_to_dev')

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

        compiled = graph.compile()
        self._log_topology("task_graph", compiled)
        return compiled

    def build(self):
        """Legacy method — returns the task graph for backward compatibility.

        New code should use build_po_graph() or build_task_graph() directly.
        """
        return self.build_task_graph()
        
    def build_final_qa_graph(self):
        """Builds the final project QA graph.
        
        START → final_qa_validate
              ├─ pass  → END
              ├─ retry → create_final_qa_fix_task → END
              └─ fail  → END
        """
        graph = StateGraph(WorkflowState)
        
        graph.add_node('final_qa_validate', self.nodes.final_qa_validate)
        graph.add_node('create_final_qa_fix_task', self.nodes.create_final_qa_fix_task)
        
        graph.add_edge(START, 'final_qa_validate')
        
        graph.add_conditional_edges(
            'final_qa_validate',
            route_by_final_qa_result,
            {
                'pass': END,
                'retry': 'create_final_qa_fix_task',
                'fail': END
            }
        )
        
        graph.add_edge('create_final_qa_fix_task', END)
        
        compiled = graph.compile()
        self._log_topology("final_qa_graph", compiled)
        return compiled
