from langchain_openai import ChatOpenAI
from app.agents.base import POResult, DevResult, QCResult

class OpenAIAPIAgent:
    def __init__(self, api_key: str, model: str = "gpt-5-nano"):
        self.llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

class POAgent(OpenAIAPIAgent):
    def analyze(self, brd_content: str) -> POResult:
        structured_llm = self.llm.with_structured_output(POResult, method="function_calling")
        prompt = f"""
        You are a Product Owner. Analyze the following Business Requirement Document (BRD) and:
        1. Summarize the main features.
        2. Break it down into User Stories (title, description, priority, acceptance_criteria).
        3. Create Backlog Items (title, description, team).
        4. Create specific Implementation Tasks (title, description, assignee_team, acceptance_criteria, required_markers).
        
        BRD Content:
        {brd_content}
        """
        return structured_llm.invoke(prompt)

class DevAgent(OpenAIAPIAgent):
    def implement(self, task: dict, acceptance_criteria: list[str], bug_reports: list[dict] | None = None) -> DevResult:
        structured_llm = self.llm.with_structured_output(DevResult, method="function_calling")
        bug_context = ""
        if bug_reports:
            bug_context = f"\nPlease fix these bugs from previous QC:\n{bug_reports}"
            
        prompt = f"""
        You are a Senior Software Engineer. Implement the following task as part of a larger software project.
        
        Task: {task.get('title')}
        Description: {task.get('description')}
        Acceptance Criteria: {acceptance_criteria}
        {bug_context}
        
        Requirements for your response:
        1. file_path: A relative path inside the project (e.g., 'app/api/admin.py', 'app/services/user_service.py',
           'frontend/src/components/Login.tsx', 'tests/test_admin.py'). Reflect a proper project layout.
           Do NOT return just a bare filename like 'admin_api.py'—always include the subdirectory.
        2. code: The complete source code implementation.
        3. unit_tests: Comprehensive unit tests for the implementation.
        4. implementation_notes: Detailed notes about your implementation and how to use it.
        5. included_markers: A list of technical features or markers you implemented.
        
        Return the source code, unit tests, and implementation notes in the structured format.
        """
        return structured_llm.invoke(prompt)

class QCAgent(OpenAIAPIAgent):
    def validate(self, task: dict, acceptance_criteria: list[str], dev_output: dict) -> QCResult:
        structured_llm = self.llm.with_structured_output(QCResult, method="function_calling")
        prompt = f"""
        You are a Quality Control Engineer. Validate the developer's output against the task requirements.
        
        Task: {task.get('title')}
        Requirements: {acceptance_criteria}
        
        Developer Output:
        Code: {dev_output.get('code')}
        Unit Tests: {dev_output.get('unit_tests')}
        Implementation Notes: {dev_output.get('implementation_notes')}
        
        Requirements for your response:
        1. passed: Boolean (true/false) indicating if all acceptance criteria were met.
        2. status: A string status (e.g., 'PASSED', 'FAILED', 'NEEDS_REVISION').
        3. validation_report: A detailed explanation of your findings.
        4. failed_criteria: A list of specific criteria that were not met.
        5. severity: The severity of any issues found ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        
        Determine if it passed all criteria and provide the structured validation result.
        """
        return structured_llm.invoke(prompt)
