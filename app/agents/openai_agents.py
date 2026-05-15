import json
from typing import Any
from langchain_openai import ChatOpenAI
from app.agents.base import POResult, POReviewResult, DevResult, QCResult, FinalProjectQAResult, GENERIC_MARKERS_BLACKLIST
from app.agents.serialization_utils import to_plain_dict, to_pretty_json  # noqa: F401 — re-exported
from pydantic import ValidationError

# ============================================================
# Shared helpers
# ============================================================

def invoke_structured_with_retry(structured_llm, messages, max_attempts: int = 2):
    """
    Invoke a structured LLM with a schema-repair retry.

    This retry is only for structured output / schema errors.
    It should not be confused with business-level PO review retry.
    """
    last_error: Exception | None = None

    for _ in range(max_attempts):
        try:
            return structured_llm.invoke(messages)
        except Exception as exc:
            last_error = exc

            exc_str = str(exc)
            if len(exc_str) > 2000:
                exc_str = exc_str[:2000] + "\n... [truncated]"

            messages = messages + [
                (
                    "human",
                    f"""
Your previous response did not match the required schema.

Validation errors:
{exc_str}

Return the complete corrected structured output.

Universal schema rules:
- Return only fields defined by the schema.
- Do not include extra fields.
- Include every required field.
- Do not use empty strings for required fields.
- Do not include empty strings inside lists.
- Use valid enum values exactly as required.
- Use snake_case markers with no spaces.
- Return the complete object, not a partial patch.

POResult rules:
- user_stories, backlog_items, and implementation_tasks MUST be top-level lists.
- Do NOT nest implementation_tasks inside backlog_items.
- Do NOT nest backlog_items inside user_stories.
- story_id must use US-001 format.
- backlog_item_id must use BL-001 format.
- task_id must use TASK-001 format.
- All related_user_story_ids must reference existing story IDs.
- All related_backlog_item_ids must reference existing backlog item IDs.
- Do not collapse all tasks into backend unless the BRD truly only describes backend/internal service work.
- input_context must be a non-empty dictionary for each implementation task.
- Prefer concrete input_context keys, but do not invent details only to increase key count.

POReviewResult rules:
- decision must be PASS or NEEDS_REVISION.
- If decision is PASS, issues must be [].
- If decision is NEEDS_REVISION, issues must contain at least one blocking issue.
- Do not put minor/non-blocking suggestions into issues.

DevResult rules:
- task_id must use TASK-001 format.
- files must be a list of objects only.
- Every files item must be an object with exactly file_path and code.
- Never put strings such as "unit_tests", "NULL", "null", "None", "tests", or placeholders inside files.
- unit_tests must be a separate top-level list of objects.
- Every unit_tests item must be an object with exactly file_path and code.
- If there are no unit tests, use "unit_tests": [].
- included_markers must be snake_case with no spaces.
- known_limitations must be a list of strings. Use [] if none.
- For non-foundation tasks, do not create a standalone project.
- Follow PROJECT CONTEXT and CURRENT_PROJECT_SNAPSHOT.
- Return complete contents only for changed files.
- Preserve existing behavior when modifying files.
- If a file is modified, include the full updated file content.
- Do not omit existing imports/routes/tests from modified files unless explicitly required.
- If adding a dependency, update dependency file.
- If changing architecture, commands, contracts, or directory structure, update README.md and project_context.json.
- Do not create duplicate entrypoints or alternate frameworks.

QCResult rules:
- passed/status/failed_criteria must be internally consistent.
- If passed is true, status must be PASSED and failed_criteria must be empty.
- If passed is false, status must be FAILED or NEEDS_REVISION.
- Validate against CURRENT_PROJECT_SNAPSHOT_AFTER_TASK when provided.
- If passed is true, all criteria and markers must be PASSED.
- If integration into snapshot is missing, passed must be false.

Return only the corrected structured output.
                    """.strip(),
                )
            ]

    raise last_error


# ============================================================
# PO issue helpers
# ============================================================

def normalize_issue(issue: dict) -> dict:
    """Normalize issue shape to avoid prompt formatting errors."""
    issue = to_plain_dict(issue or {})

    affected_items = issue.get("affected_items", [])
    if affected_items is None:
        affected_items = []
    elif not isinstance(affected_items, list):
        affected_items = [str(affected_items)]

    return {
        "category": issue.get("category", "schema_compliance"),
        "severity": str(issue.get("severity", "HIGH")).upper(),
        "description": issue.get("description", ""),
        "affected_items": [str(item) for item in affected_items if str(item).strip()],
        "suggestion": issue.get("suggestion", ""),
    }


def is_blocking_po_issue(issue: dict) -> bool:
    """
    Decide whether a PO review issue should block the flow.

    Current POReviewResult schema only supports:
    - PASS
    - NEEDS_REVISION

    Therefore, minor issues must not be returned as issues,
    otherwise they will force retry.
    """
    issue = normalize_issue(issue)
    severity = issue["severity"]
    category = str(issue.get("category", "")).lower()

    if severity in {"HIGH", "CRITICAL"}:
        return True

    # LOW/MEDIUM are normally non-blocking.
    # Keep this conservative to prevent retry loops.
    return False


def filter_active_po_issues(issues: list[dict] | None) -> list[dict]:
    """
    Keep only active blocking issues for retry prompts.

    Important:
    - Do not pass all historical issues into PO.
    - Do not pass LOW/MEDIUM suggestions into PO retry.
    """
    clean = [normalize_issue(issue) for issue in to_plain_dict(issues or [])]
    return [issue for issue in clean if is_blocking_po_issue(issue)]


def has_observable_detail(text: str) -> bool:
    """
    Lightweight heuristic to avoid treating every vague phrase as blocking.
    If the AC contains concrete verification details, vague wording is non-blocking.
    """
    text_lower = (text or "").lower()

    observable_signals = [
        "http ",
        "status code",
        "returns ",
        "return ",
        "response",
        "error_code",
        "message",
        "details",
        "field",
        "reason",
        "ui",
        "page",
        "screen",
        "button",
        "form",
        "banner",
        "modal",
        "toast",
        "empty state",
        "loading state",
        "disabled",
        "enabled",
        "database",
        "table",
        "stored",
        "persisted",
        "redis",
        "cache key",
        "marker",
        "given ",
        "when ",
        "then ",
        "must ",
        "equals",
        "contains",
        "within ",
        "less than",
        "greater than",
        "422",
        "400",
        "401",
        "403",
        "404",
        "409",
        "500",
    ]

    return any(signal in text_lower for signal in observable_signals)


def validate_po_result_locally(po: POResult) -> list[dict]:
    """
    Rule-based PO validation.

    Important behavior:
    - Only return HIGH/CRITICAL for truly blocking issues.
    - Return LOW/MEDIUM for guidance only.
    - Downstream retry should only use blocking issues.
    """
    issues: list[dict] = []

    vague_phrases = [
        "properly",
        "correctly",
        "gracefully",
        "seamlessly",
        "user-friendly",
        "fast",
        "performant",
        "scalable",
        "secure",
        "robust",
        "maintainable",
        "extensible",
        "clean",
        "future extension",
        "allow future extension",
        "structured error responses",
        "consistent with design",
        "normal response time",
        "as needed",
        "where applicable",
        "works well",
        "appropriate",
        "relevant",
        "optimized",
    ]

    # 1. Acceptance Criteria quality for user stories
    for story in po.user_stories:
        for idx, ac in enumerate(story.acceptance_criteria or [], 1):
            ac_text = ac or ""
            found = [p for p in vague_phrases if p in ac_text.lower()]

            if found and not has_observable_detail(ac_text):
                issues.append({
                    "category": "acceptance_criteria",
                    "severity": "HIGH",
                    "description": (
                        f"Story {story.story_id} AC-{idx} contains vague phrase(s) {found} "
                        "without concrete observable verification details."
                    ),
                    "affected_items": [f"{story.story_id}.acceptance_criteria[{idx}]"],
                    "suggestion": (
                        "Rewrite the AC with condition/input, observable behavior, and expected output/state/status."
                    ),
                })
            elif found:
                issues.append({
                    "category": "acceptance_criteria",
                    "severity": "LOW",
                    "description": (
                        f"Story {story.story_id} AC-{idx} contains vague phrase(s) {found}, "
                        "but also appears to include observable details."
                    ),
                    "affected_items": [f"{story.story_id}.acceptance_criteria[{idx}]"],
                    "suggestion": "Consider removing vague wording if it is not needed.",
                })

    # 2. Acceptance Criteria quality for tasks
    for task in po.implementation_tasks:
        for idx, ac in enumerate(task.acceptance_criteria or [], 1):
            ac_text = ac or ""
            found = [p for p in vague_phrases if p in ac_text.lower()]

            if found and not has_observable_detail(ac_text):
                issues.append({
                    "category": "acceptance_criteria",
                    "severity": "HIGH",
                    "description": (
                        f"Task {task.task_id} AC-{idx} contains vague phrase(s) {found} "
                        "without concrete observable verification details."
                    ),
                    "affected_items": [f"{task.task_id}.acceptance_criteria[{idx}]"],
                    "suggestion": (
                        "Rewrite the AC so DEV/QC can verify it with a clear pass/fail signal."
                    ),
                })
            elif found:
                issues.append({
                    "category": "acceptance_criteria",
                    "severity": "LOW",
                    "description": (
                        f"Task {task.task_id} AC-{idx} contains vague phrase(s) {found}, "
                        "but also appears to include observable details."
                    ),
                    "affected_items": [f"{task.task_id}.acceptance_criteria[{idx}]"],
                    "suggestion": "Consider removing vague wording if it is not needed.",
                })

    # 3. Backlog traceability
    story_ids = {s.story_id for s in po.user_stories}

    for item in po.backlog_items:
        if not item.related_user_story_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": f"Backlog item {item.backlog_item_id} is not linked to any user story.",
                "affected_items": [item.backlog_item_id],
                "suggestion": "Link this backlog item to at least one existing user story.",
            })

        unknown_story_ids = [
            sid for sid in item.related_user_story_ids or []
            if sid not in story_ids
        ]
        if unknown_story_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": (
                    f"Backlog item {item.backlog_item_id} references unknown story IDs: {unknown_story_ids}."
                ),
                "affected_items": [item.backlog_item_id],
                "suggestion": "Use only existing story IDs in related_user_story_ids.",
            })

    # 4. Task traceability and task quality
    backlog_ids = {b.backlog_item_id for b in po.backlog_items}

    for task in po.implementation_tasks:
        if not task.related_user_story_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": f"Task {task.task_id} is not linked to any user story.",
                "affected_items": [task.task_id],
                "suggestion": "Link this task to at least one existing user story.",
            })

        unknown_story_ids = [
            sid for sid in task.related_user_story_ids or []
            if sid not in story_ids
        ]
        if unknown_story_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": (
                    f"Task {task.task_id} references unknown story IDs: {unknown_story_ids}."
                ),
                "affected_items": [task.task_id],
                "suggestion": "Use only existing story IDs in related_user_story_ids.",
            })

        if not task.related_backlog_item_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": f"Task {task.task_id} is not linked to any backlog item.",
                "affected_items": [task.task_id],
                "suggestion": "Link this task to at least one existing backlog item.",
            })

        unknown_backlog_ids = [
            bid for bid in task.related_backlog_item_ids or []
            if bid not in backlog_ids
        ]
        if unknown_backlog_ids:
            issues.append({
                "category": "traceability",
                "severity": "HIGH",
                "description": (
                    f"Task {task.task_id} references unknown backlog item IDs: {unknown_backlog_ids}."
                ),
                "affected_items": [task.task_id],
                "suggestion": "Use only existing backlog item IDs in related_backlog_item_ids.",
            })

        # input_context should not be a retry trap.
        # Empty/invalid is blocking. Fewer than 3 keys is only guidance.
        input_context = getattr(task, "input_context", None)

        if not isinstance(input_context, dict) or len(input_context) == 0:
            issues.append({
                "category": "schema_compliance",
                "severity": "HIGH",
                "description": f"Task {task.task_id} input_context is empty or invalid.",
                "affected_items": [f"{task.task_id}.input_context"],
                "suggestion": (
                    "Provide a non-empty machine-readable input_context with concrete implementation hints."
                ),
            })
        elif len(input_context) < 3:
            issues.append({
                "category": "schema_compliance",
                "severity": "LOW",
                "description": (
                    f"Task {task.task_id} input_context has only {len(input_context)} key(s). "
                    "This is acceptable if the keys are concrete and useful."
                ),
                "affected_items": [f"{task.task_id}.input_context"],
                "suggestion": (
                    "Add more implementation hints only if available in the BRD. Do not invent details."
                ),
            })

        if not task.acceptance_criteria:
            issues.append({
                "category": "acceptance_criteria",
                "severity": "HIGH",
                "description": f"Task {task.task_id} has no acceptance criteria.",
                "affected_items": [task.task_id],
                "suggestion": "Define at least one pass/fail observable acceptance criterion.",
            })

        if not task.required_markers:
            issues.append({
                "category": "marker_quality",
                "severity": "HIGH",
                "description": f"Task {task.task_id} has no required_markers.",
                "affected_items": [f"{task.task_id}.required_markers"],
                "suggestion": "Add concrete snake_case markers that DEV/QC can verify.",
            })
        else:
            bad_markers = [
                m for m in task.required_markers
                if m in GENERIC_MARKERS_BLACKLIST
            ]

            if bad_markers:
                issues.append({
                    "category": "marker_quality",
                    "severity": "HIGH",
                    "description": (
                        f"Task {task.task_id} contains blacklisted generic markers: {bad_markers}."
                    ),
                    "affected_items": [f"{task.task_id}.required_markers"],
                    "suggestion": (
                        "Replace generic markers with resource/action/result-specific identifiers, "
                        "for example product_create_post_api_v1_rejects_negative_price."
                    ),
                })

    return [normalize_issue(issue) for issue in issues]


# ============================================================
# Base OpenAI agent
# ============================================================

class OpenAIAPIAgent:
    """Base class for OpenAI-powered agents."""

    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)


# ============================================================
# PO Agent
# ============================================================

PO_SYSTEM_PROMPT = """
You are a senior Product Owner working with engineering delivery teams.

Your job is to transform a Business Requirement Document (BRD) into delivery-ready product and engineering artifacts.

You must return structured output that exactly matches the POResult schema.

Priority 1 — Must obey:
1. Base the output only on the BRD.
2. Do not invent unsupported scope, features, integrations, screens, APIs, fields, workflows, business rules, or non-functional requirements.
3. If the BRD is ambiguous or incomplete, use the smallest reasonable assumption and record it in assumptions.
4. If missing information blocks implementation, create a product clarification task instead of inventing the answer.
5. user_stories, backlog_items, and implementation_tasks must be top-level flat lists.
6. All IDs must be stable and traceable:
   - story_id: US-001, US-002, ...
   - backlog_item_id: BL-001, BL-002, ...
   - task_id: TASK-001, TASK-002, ...
7. Every backlog item must reference at least one existing user story.
8. Every implementation task must reference at least one existing user story and one existing backlog item.
9. Do not collapse all work into backend unless the BRD truly only describes backend/internal service work.
10. MANDATORY FIRST TASK: The very first task (TASK-001) MUST be 'Project Foundation: Codebase Structure, README, and Project Context'.
    - This task's goal is to establish the project's file structure and create the project contract files.
    - TASK-001 MUST create all of the following:
      a. README.md - the human-readable project contract.
      b. project_context.json - the machine-readable project contract.
      c. Skeleton project with minimal runnable structure.
      d. Dependency/config files if appropriate (e.g. requirements.txt, package.json, pyproject.toml, go.mod).
    - README.md MUST contain these sections: Project overview, Product goal, Tech stack, Directory structure, Entrypoints, Setup commands, Run commands, Test commands, Architecture rules, Shared contracts, API contracts (if applicable), UI contracts (if applicable), Data contracts (if applicable), File ownership rules, Future task rules, Known assumptions, Out of scope.
    - project_context.json MUST contain these keys: project_name, stack (with language, framework, test_runner), entrypoints, run_commands (with install, run, test), test_commands, directory_structure (list of paths), architecture_rules (list of rules), shared_contracts, file_ownership_rules (list of rules), task_implementation_rules (list of rules including: do not create standalone project, do not create second entrypoint, do not introduce alternate framework, modify existing files preserving behavior, update README.md and project_context.json when changing commands/structure/contracts), out_of_scope.
    - TASK-001 input_context MUST include: project_goal, proposed_tech_stack, directory_structure, readme_required_sections, project_context_json_required_keys, future_task_rules.
    - This task MUST be assigned to the `devops` or `backend` team.
    - All subsequent implementation tasks MUST refer to this foundation to avoid fragmented code.
    - Every subsequent task's input_context MUST include expected_paths or target_modules that align with the README directory structure, integration_notes or affected_existing_files if the task modifies existing files, and dependency_on_project_context set to true.

Priority 2 — Quality bar:
1. User stories must include actor, goal, and value.
2. Acceptance criteria must be observable, testable, and pass/fail verifiable.
3. Each acceptance criterion should include:
   - a condition/input,
   - an observable behavior,
   - an expected output, state, status code, UI state, stored value, marker, or validation result.
4. Implementation tasks must be concrete, small enough for one team, non-overlapping, and actionable.
5. required_markers must be snake_case, concrete, and verifiable.
6. input_context must be a non-empty dictionary with concrete hints useful for the DEV agent.
7. Prefer 2-5 concrete input_context keys. Do not invent context keys just to satisfy quantity.

Priority 3 — Domain decomposition:
1. Create tasks only for domains supported by the BRD.
2. Use product for open questions, unresolved product decisions, requirement clarification, and blocked business rules.
3. Use backend for APIs, services, business rules, auth, validation, integrations, transactions, and orchestration.
4. Use frontend for screens, forms, UI behavior, routing, visible states, and client-side validation.
5. Use mobile only for mobile-specific behavior.
6. Use data for schema, migrations, seed data, indexes, persistence design, analytics tables, and storage structures.
7. Use devops/platform for Docker, deployment, CI/CD, environment configuration, service orchestration, and monitoring.
8. Use qa for explicit QA plans, regression suites, manual validation, security tests, or performance tests.
9. Use security for security-specific controls beyond ordinary backend validation.
10. Use design only when the BRD explicitly asks for UX/UI deliverables.
11. Use unknown only when the responsible domain cannot reasonably be determined.

Revision mode rules:
1. When review feedback is provided, fix active blocking issues first.
2. Preserve valid existing story_id, backlog_item_id, and task_id values.
3. Do not rewrite unrelated valid artifacts.
4. Do not re-index unless the existing structure is invalid.
5. Use reviewer suggestions when they are specific and BRD-compatible.
6. If feedback points to missing BRD information, add an assumption or product clarification task instead of inventing behavior.
7. Populate resolution_map with how each active issue was addressed.
8. Self-audit before returning.

Acceptance criteria wording:
- Avoid vague criteria such as “works correctly”, “handles gracefully”, “fast”, “secure”, “maintainable”, “as needed”, or “structured error responses” unless the same criterion also includes exact observable verification details.
- Do not treat quality concepts as invalid by themselves. Make them measurable.

Good acceptance criteria examples:
- For validation failures, the API returns HTTP 422 with JSON body containing error_code, message, and details[]. Each details[] item includes field and reason.
- When checkout encounters deleted product_ids in the cart, the checkout response includes skipped_product_ids and continues checkout for remaining valid items.
- If all cart items reference deleted products, checkout returns HTTP 422 with error_code CART_HAS_NO_VALID_ITEMS.
- If the product list is empty, the page displays an empty state message and no product cards.

Marker rules:
- required_markers must be concrete snake_case identifiers.
- Markers must describe specific verifiable behavior, not generic activity.
- Do not require runtime execution logs, live tests, or runtime screenshots. The DEV agent produces static code artifacts.
- Good examples:
  - product_create_post_api_v1_rejects_negative_price
  - checkout_post_api_v1_returns_422_when_no_valid_items_remain
  - product_list_page_renders_empty_state_when_no_products
  - products_table_price_positive_constraint_added
  - checkout_deleted_product_scenario_test_defined
- Bad examples:
  - good_quality
  - clean_code
  - secure
  - works_correctly
  - input_validation
  - request_validation
  - api_endpoint_created
  - tests_added
  - backend_done

Schema rules:
- Return only fields defined by the POResult schema.
- Do not include extra fields.
- Do not omit required fields.
- Do not use empty strings for required fields.
- Do not include empty strings inside lists.
- Priority must be one of: LOW, MEDIUM, HIGH.
- Team must be one of: product, backend, frontend, fullstack, mobile, data, devops, platform, qa, security, design, unknown.
- If the responsible team is unclear, use unknown.
- resolution_map is mandatory in revision mode.

Return only the structured output matching the POResult schema.
"""


class POAgent(OpenAIAPIAgent):
    """Product Owner agent responsible for BRD analysis and task breakdown."""

    def analyze(
        self,
        brd_content: str,
        previous_result: dict | None = None,
        review_issues: list[dict] | None = None,
    ) -> POResult:
        structured_llm = self.llm.with_structured_output(
            POResult,
            method="function_calling",
        )

        # Important:
        # PO should only receive active blocking issues.
        # Do not feed all historical LOW/MEDIUM issues into the retry prompt.
        clean_issues = filter_active_po_issues(review_issues or [])

        feedback_block = ""
        if clean_issues:
            feedback_block = "\n### REVISION MODE: ACTIVE BLOCKING ISSUES ONLY\n"
            feedback_block += (
                "You are revising a previous POResult. The issues below are active unresolved blocking issues.\n"
                "Your goal is convergence: fix these issues without rewriting unrelated valid artifacts.\n\n"
            )
            feedback_block += "Revision rules:\n"
            feedback_block += "1. Fix all active HIGH/CRITICAL issues first.\n"
            feedback_block += "2. Preserve valid story_id, backlog_item_id, and task_id from previous results.\n"
            feedback_block += "3. Do not re-index IDs unless the existing structure is invalid.\n"
            feedback_block += "4. Do not rewrite unrelated stories, backlog items, or tasks.\n"
            feedback_block += "5. If a reviewer suggestion is specific and BRD-compatible, follow it.\n"
            feedback_block += "6. If BRD information is missing, do not invent behavior. Add a minimal assumption or create a product clarification task.\n"
            feedback_block += "7. Populate resolution_map with how each active issue was resolved.\n"
            feedback_block += "8. Self-audit the revised output for schema, traceability, testable AC, marker quality, and BRD faithfulness.\n\n"

            for idx, issue in enumerate(clean_issues, 1):
                affected_items = issue.get("affected_items", [])
                feedback_block += f"{idx}. [{issue.get('category')}] Severity: {issue.get('severity')}\n"
                feedback_block += f"   Description: {issue.get('description')}\n"
                feedback_block += f"   Affected items: {', '.join(affected_items)}\n"
                feedback_block += f"   Required fix: {issue.get('suggestion')}\n\n"

        previous_artifacts_block = ""
        if previous_result:
            previous_artifacts_block = "\n### PREVIOUS ARTIFACTS\n"
            previous_artifacts_block += (
                "Use these artifacts as the base revision target. Preserve valid IDs and unchanged valid sections.\n"
            )
            previous_artifacts_block += to_pretty_json(previous_result)

        human_prompt = f"""
Analyze the following BRD and return a structured POResult.

{feedback_block}

{previous_artifacts_block}

Instructions:
- Base the output only on the BRD.
- Do not invent unsupported scope.
- Identify implementation domains from the BRD itself.
- Create product clarification tasks for unresolved decisions or open questions.
- Use explicit assumptions only when needed and keep them minimal.
- Ensure every user story, backlog item, and implementation task is traceable.
- Ensure every acceptance criterion is concrete and pass/fail testable.
- Ensure required_markers are concrete snake_case identifiers.
- Ensure input_context is non-empty and useful for the DEV agent.
- In revision mode, preserve valid IDs and update resolution_map.
- Return only the structured POResult.

<BRD>
{brd_content}
</BRD>
        """.strip()

        messages = [
            ("system", PO_SYSTEM_PROMPT.strip()),
            ("human", human_prompt),
        ]

        return invoke_structured_with_retry(structured_llm, messages)


# ============================================================
# Final Project QA Agent
# ============================================================

FINAL_PROJECT_QA_SYSTEM_PROMPT = """
You are a senior Software QA Architect and Integration Specialist.
Your job is to validate an entire generated project as a complete, runnable application.

You are NOT validating a single task. You are validating the final project state after all development tasks are done.
You will be given:
1. The project context (structure, stack, dependencies, entrypoints).
2. The exact commands that were executed to validate the project.
3. The standard output and standard error from those commands.

Your responsibilities:
1. Analyze the logs to determine if the project successfully installed dependencies, built, tested, and started.
2. Detect integration issues (e.g., missing imports, bad paths, missing dependencies, frontend calling non-existent backend routes).
3. If the project passed all critical commands with no errors, return passed=true.
4. If the project failed, return passed=false and formulate a single, actionable repair task for the DevAgent.
5. The repair task must include a clear explanation, the failing commands, and concrete input_context to help DevAgent fix the codebase.
6. Do NOT assume the project works if there are obvious compilation or startup errors.
7. Treat warnings as passing, but treat fatal errors, panics, exceptions, or module-not-found errors as failing.

Your output must strictly match the FinalProjectQAResult schema.
"""

class FinalProjectQAAgent(OpenAIAPIAgent):
    """Agent responsible for final whole-project validation and repair task generation."""

    def analyze_results(
        self,
        project_context: dict[str, Any],
        build_logs: list[dict[str, Any]]
    ) -> FinalProjectQAResult:
        structured_llm = self.llm.with_structured_output(
            FinalProjectQAResult,
            method="function_calling",
        )

        human_prompt = f"""
Validate the entire project state and the execution logs.

Project Context:
{to_pretty_json(project_context)}

Execution Logs (commands run):
{to_pretty_json(build_logs)}

Instructions:
- If all commands succeeded (exit code 0 or no critical runtime errors), set `passed: true` and `repair_task: null`.
- If any command failed (dependency, build, test, run), set `passed: false` and provide a `repair_task`.
- The `repair_task` must be concrete. Specify exactly what file/dependency is broken based on the logs.
- Provide a clear `validation_report`.
"""
        messages = [
            ("system", FINAL_PROJECT_QA_SYSTEM_PROMPT.strip()),
            ("human", human_prompt.strip()),
        ]

        return invoke_structured_with_retry(structured_llm, messages)



# ============================================================
# PO Review Agent
# ============================================================

PO_REVIEW_SYSTEM_PROMPT = """
You are a senior Product Owner reviewer performing a quality gate on PO-generated artifacts.

Your job is to decide whether the PO output is ready to be sent downstream to DEV and QC agents.

You must return structured output that exactly matches the POReviewResult schema.

Review scope:
1. Use only the provided BRD, feature_summary, user_stories, backlog_items, and implementation_tasks.
2. Do not invent new requirements.
3. Do not expand scope beyond the BRD.
4. Do not require domains, APIs, screens, tests, infrastructure, or security controls that are not explicit or strongly indicated by the BRD.

Important decision policy:
1. Return NEEDS_REVISION only when there is at least one blocking issue.
2. A blocking issue is an issue that prevents DEV/QC from implementing or validating the work safely.
3. Minor wording, style, or improvement suggestions must not block the flow.
4. If only minor/non-blocking issues exist, return PASS with an empty issues list.
5. Do not use issues as a place for optional suggestions.
6. Be strict about blocking problems, but optimize for convergence.

Blocking issue examples:
- A BRD requirement is missing from all stories/tasks.
- A story/task invents unsupported scope.
- A task is not actionable enough for DEV.
- Acceptance criteria are not testable and provide no observable behavior.
- Required IDs or traceability are missing or invalid.
- A task has no acceptance criteria.
- required_markers are missing or unusable.
- input_context is empty, null, or not a dictionary.
- Tasks are collapsed into the wrong domain despite clear BRD domain separation.
- Open product questions are converted into implementation behavior without an assumption or clarification task.

Non-blocking issue examples:
- A phrase could be more precise but the criterion is still testable.
- input_context has fewer than 3 keys but is non-empty and useful.
- A marker could be more specific but is still snake_case and verifiable.
- Story wording does not exactly follow “As a..., I want..., so that...” but actor, goal, and value are clear.
- Minor style, formatting, or naming improvements that do not block DEV/QC.

Assumption and missing context policy:
1. Do not fail explicit, minimal, BRD-aligned assumptions.
2. Fail only if an assumption invents new business rules, integrations, APIs, screens, workflows, data fields, or non-functional requirements not supported by the BRD.
3. If the BRD contains open questions or unresolved decisions, accept product clarification tasks as valid output.
4. Do not require the PO to solve missing business decisions that are not in the BRD.

Previous issue / retry policy:
1. If previous issues are provided, check whether they are still present in the current artifacts.
2. Do not repeat an old issue if it has been resolved.
3. If an old issue persists, report it with a more specific suggestion.
4. Only introduce a new issue during retry if it is blocking or was introduced by the latest revision.
5. Do not create a moving target by reporting minor new issues after blocking issues have been fixed.

Schema compliance rules:
- Return only fields defined by the POReviewResult schema.
- decision must be exactly PASS or NEEDS_REVISION.
- If decision is PASS, issues must be empty.
- If decision is NEEDS_REVISION, issues must contain at least one blocking issue.
- Each issue must include:
  - category
  - severity
  - description
  - affected_items
  - suggestion
- category must be one of:
  - story_format
  - completeness
  - backlog_quality
  - task_clarity
  - duplicate
  - traceability
  - acceptance_criteria
  - marker_quality
  - schema_compliance
- severity must be one of:
  - LOW
  - MEDIUM
  - HIGH
  - CRITICAL
- Use HIGH or CRITICAL for blocking issues.
- Do not return LOW or MEDIUM issues. Minor issues should not be included in issues.
- affected_items must contain specific ids, fields, criteria, or markers where possible.
- suggestion must be actionable and specific.

Review dimensions:

1. Completeness vs BRD
- Verify that explicit BRD requirements are covered by stories/backlog/tasks.
- Do not require unsupported implied requirements.
- If a requirement is missing, report completeness.

2. User story quality
- Stories should include actor, goal, and value.
- Exact wording is not required.
- Report story_format only if actor, goal, or value is missing.

3. Backlog quality
- Backlog items must be concrete and traceable to user stories.
- Report vague, duplicate, unsupported, or untraceable backlog items.

4. Task clarity
- Tasks must be actionable, non-overlapping, assigned to a suitable single team, and small enough for one team.
- Product clarification tasks are valid when BRD has open questions.
- Report vague, duplicate, wrong-team, unsupported, or non-actionable tasks.

5. Traceability
- story_id must use US-001 format.
- backlog_item_id must use BL-001 format.
- task_id must use TASK-001 format.
- Backlog related_user_story_ids must reference existing story IDs.
- Task related_user_story_ids must reference existing story IDs.
- Task related_backlog_item_ids must reference existing backlog item IDs.
- Report missing, invalid, duplicate, or unknown IDs.

6. Acceptance criteria
- AC must be pass/fail testable.
- AC should include condition/input, observable behavior, and expected output/state/status.
- Do not fail merely because a vague word appears.
- Fail only if the criterion lacks concrete observable verification.
- Do not fail if the criterion includes exact details such as endpoint, status code, response fields, UI state, DB state, validation rule, measurable threshold, or marker.

7. Required markers
- Markers must be snake_case, concrete, relevant, and verifiable.
- Fail missing, malformed, generic, unsupported, or irrelevant markers only when they prevent downstream validation.

8. input_context
- input_context must be a non-empty dictionary.
- If input_context is empty, null, or not a dictionary, report schema_compliance.
- If input_context has fewer than 3 keys but is still useful and concrete, do not block the flow.

Decision rules:
- PASS when no blocking issues exist.
- NEEDS_REVISION when one or more blocking issues exist.
- If decision is PASS, issues must be [].
- If decision is NEEDS_REVISION, include only blocking issues.
- Do not report optional suggestions as issues.
- Do not generate full replacement stories, backlog items, or tasks.
- Suggestions must be concrete enough for PO to fix in the next retry.

Return only the structured output matching the POReviewResult schema.
"""


class POReviewAgent(OpenAIAPIAgent):
    """
    PO Review agent — validates PO output before tasks go to DEV.

    This version is convergence-oriented:
    - only blocks on HIGH/CRITICAL issues,
    - does not repeat resolved previous issues,
    - does not use minor suggestions as retry triggers.
    """

    def review(
        self,
        brd_content: str,
        feature_summary: str,
        user_stories: list[dict],
        backlog_items: list[dict],
        implementation_tasks: list[dict],
        previous_issues: list[dict] | None = None,
    ) -> POReviewResult:
        structured_llm = self.llm.with_structured_output(
            POReviewResult,
            method="function_calling",
        )

        brd_text = brd_content or ""
        stories_json = to_pretty_json(user_stories or [])
        backlog_json = to_pretty_json(backlog_items or [])
        tasks_json = to_pretty_json(implementation_tasks or [])

        # Important:
        # Reviewer should only see previous blocking issues for convergence.
        # Do not load all old LOW/MEDIUM history into review context.
        active_previous_issues = filter_active_po_issues(previous_issues or [])

        history_block = ""
        if active_previous_issues:
            history_block = "\n### PREVIOUS ACTIVE BLOCKING ISSUES\n"
            history_block += (
                "These issues were reported in previous review rounds. "
                "Use them only to check convergence.\n"
                "Do not repeat an old issue if the current artifacts have resolved it.\n"
                "Only report an old issue again if it is still clearly present.\n\n"
            )

            for idx, issue in enumerate(active_previous_issues, 1):
                affected_items = issue.get("affected_items", [])
                history_block += f"{idx}. [{issue.get('category')}] Severity: {issue.get('severity')}\n"
                history_block += f"   Description: {issue.get('description')}\n"
                history_block += f"   Affected: {', '.join(affected_items)}\n"
                history_block += f"   Previous suggestion: {issue.get('suggestion')}\n\n"

        human_prompt = f"""
Review the following PO output for delivery readiness.

Instructions:
- Base your review only on the provided BRD and PO artifacts.
- Return NEEDS_REVISION only for blocking issues.
- If only minor/non-blocking issues exist, return PASS with issues = [].
- Do not require unsupported scope.
- Do not fail explicit minimal assumptions that are aligned with the BRD.
- Accept product clarification tasks for unresolved BRD questions.
- On retry, do not repeat previous issues that have been resolved.
- Every returned issue must be specific, blocking, and actionable.

<BRD>
{brd_text}
</BRD>

{history_block}

### PO ARTIFACTS TO REVIEW

<FEATURE_SUMMARY>
{feature_summary}
</FEATURE_SUMMARY>

<USER_STORIES>
{stories_json}
</USER_STORIES>

<BACKLOG_ITEMS>
{backlog_json}
</BACKLOG_ITEMS>

<IMPLEMENTATION_TASKS>
{tasks_json}
</IMPLEMENTATION_TASKS>
        """.strip()

        messages = [
            ("system", PO_REVIEW_SYSTEM_PROMPT.strip()),
            ("human", human_prompt),
        ]

        result = invoke_structured_with_retry(structured_llm, messages)

        # Safety normalization:
        # If the model returns NEEDS_REVISION with only LOW/MEDIUM issues,
        # convert it to PASS to avoid retry loops caused by non-blocking suggestions.
        try:
            result_dict = result.model_dump(mode="json")
            issues = result_dict.get("issues", []) or []
            blocking_issues = filter_active_po_issues(issues)

            if not blocking_issues:
                result_dict["decision"] = "PASS"
                result_dict["issues"] = []
                return POReviewResult(**result_dict)

            # Keep only blocking issues.
            result_dict["decision"] = "NEEDS_REVISION"
            result_dict["issues"] = blocking_issues
            return POReviewResult(**result_dict)

        except Exception:
            # If normalization fails, return the original structured result.
            # The structured output retry above already validated schema.
            return result


class DevAgent(OpenAIAPIAgent):
    """Developer agent responsible for task implementation and unit testing."""
    def implement(
        self,
        task: dict,
        acceptance_criteria: list[str],
        bug_reports: list[dict] | None = None,
        project_context: str | None = None,
        current_project_snapshot: dict[str, str] | None = None,
    ) -> DevResult:
        from app.agents.project_utils import build_project_snapshot_for_prompt

        structured_llm = self.llm.with_structured_output(
            DevResult,
            method="function_calling",
        )

        # Standardize payload handling to be robust against missing fields
        task_data = task or {}
        ac_data = acceptance_criteria or []
        bugs_data = bug_reports or []

        task_json = to_pretty_json(task_data)
        ac_json = to_pretty_json(ac_data)
        bugs_json = to_pretty_json(bugs_data)
        
        context_block = ""
        if project_context:
            context_block = f"\n<PROJECT_CONTEXT>\n{project_context}\n</PROJECT_CONTEXT>\n"

        snapshot_block = ""
        if current_project_snapshot:
            safe_snapshot = build_project_snapshot_for_prompt(current_project_snapshot)
            snapshot_json = to_pretty_json(safe_snapshot)
            snapshot_block = f"\n<CURRENT_PROJECT_SNAPSHOT>\n{snapshot_json}\n</CURRENT_PROJECT_SNAPSHOT>\n"

        messages = [
           (
    "system",
    """
You are a senior software engineer implementing one task within an existing software system.

Your goal is to produce a correct, minimal, testable implementation for the given task.

You must return structured output that exactly matches the DevResult schema.

Most important output rule:
- Return DevResult only.
- files must contain implementation file objects only.
- unit_tests must contain test file objects only.
- Never put strings inside files or unit_tests.
- Never put "unit_tests", "tests", "NULL", "null", "None", or placeholders inside files.
- unit_tests is a separate top-level field, not an item inside files.

Core principles:
1. Satisfy the given task and acceptance criteria exactly.
2. If bug reports are present, fix them while preserving previously satisfied behavior.
3. Use only the provided task context.
4. Do not invent unrelated system requirements, APIs, screens, integrations, data models, or workflows.
5. If a detail is ambiguous, make the smallest reasonable assumption and document it in implementation_notes.
6. Prefer simple, maintainable, production-like code over over-engineered solutions.
7. Your output must be suitable for QC validation.
8. Contextual Awareness: Before implementing any code, read the task's `input_context` and the project's overall context provided in the PO output. This contains the strategic intent of the project.
9. Project Consistency: If a `readme.md` or a project foundation exists (provided in the context), you MUST follow its rules, directory structure, and architectural patterns. Do not create 'detached' or 'fragmented' code. Always ensure your code integrates correctly into the existing project path.
10. README as project memory:
   - For TASK-001 or any project foundation task, create `README.md` as the first-class project context file.
   - `README.md` must describe the product goal, chosen tech stack, architecture, directory structure, naming conventions, core data/API/UI contracts, setup/test commands, and rules future agents must follow.
   - For every later task, treat the provided README context as authoritative. Read it before deciding file paths or APIs.
   - If a later task changes architecture, setup, directory structure, public contracts, or cross-task conventions, update `README.md` in the returned files.
   - Place all code inside the same project structure described by `README.md`; extend existing modules instead of creating unrelated standalone files.
   - If README is missing for a non-foundation task, create or repair it using the task context before adding feature code.

Schema compliance rules:
- Return only fields defined by the DevResult schema.
- Do not include extra fields.
- task_id must match the TASK task_id, e.g. TASK-001.
- files must be a JSON array of objects only.
- files must contain at least one implementation file.
- Every item in files must be an object with exactly:
  - file_path
  - code
- unit_tests must be a JSON array of objects only.
- Every item in unit_tests must be an object with exactly:
  - file_path
  - code
- If there are no unit tests, return "unit_tests": [].
- implementation_notes must be a string explaining your work.
- included_markers must be a list of snake_case strings.
- known_limitations must be a list of strings. If none, return [].

Each ImplementedFile and TestFile must have:
- file_path: Relative path using forward slashes, e.g. "app/services/auth.py".
file_path must be relative to the generated project root.
file_path may be either:
- a root-level project file such as README.md, go.mod, package.json, docker-compose.yml, .env.example, Makefile
- or a file inside a subdirectory such as cmd/api/main.go or internal/config/config.go.
file_path must not start with "/" or "../".
For TASK-001 foundation work, the primary README must be returned with file_path exactly "README.md".
This means README.md at the root of the generated codebase, not outside the generated project folder.
- file_path must not start with "/" or "../".
- code: Complete file content as a string.
- code must not contain markdown fences.

Valid DevResult shape example:
{
  "task_id": "TASK-001",
  "files": [
    {
      "file_path": "app/services/calculator.py",
      "code": "def add(a, b):\\n    return a + b"
    }
  ],
  "unit_tests": [
    {
      "file_path": "tests/test_calculator.py",
      "code": "from app.services.calculator import add\\n\\ndef test_add():\\n    assert add(1, 2) == 3"
    }
  ],
  "implementation_notes": "Implemented the add function and verified it with unit tests.",
  "included_markers": ["calculator_add_unit_test_covers_positive_numbers"],
  "known_limitations": []
}

Invalid DevResult shape examples:
{
  "files": ["unit_tests"]
}

{
  "files": [
    {
      "file_path": "app/services/example.py",
      "code": "..."
    },
    "NULL"
  ]
}

{
  "files": [
    {
      "file_path": "app/services/example.py",
      "code": "..."
    },
    "unit_tests"
  ]
}

Task implementation rules:
- Implement only the provided task scope.
- Do not expand into unrelated features.
- Foundation task rule: if the task title, description, or input_context indicates project foundation, codebase setup, README, scaffolding, or TASK-001 foundation work, return `README.md` plus the minimal initial codebase/config files needed to establish the project. The README must be an implementation file object in `files`.
- Context continuity rule: for non-foundation tasks, use the README-provided directory map and contracts to choose paths. Do not create duplicate apps, duplicate package roots, alternate frameworks, or isolated examples when the project already has a structure.
- Do not implement product clarification tasks as code. If the task assignee_team is product, return a documentation-style artifact such as "docs/product_decisions/<task_id>.md" with the clarification content required by the task. For product clarification tasks, do not invent the final business decision unless the task input_context explicitly provides it. Document the decision needed, options, impacted artifacts, and blocked implementation scope.
- If the task assignee_team is qa, return a QA artifact such as "tests/manual/<task_id>_test_plan.md" or automated test files if appropriate.
- If the task assignee_team is devops or platform, return infrastructure/config files appropriate to the task.
- If the task assignee_team is data, return migrations, schema files, seed files, or data-layer files appropriate to the task.
- If the task assignee_team is frontend, return frontend component/page/service files appropriate to the task.
- If the task assignee_team is backend, return backend API/service/repository/model files appropriate to the task.

Markers:
- included_markers must contain only markers actually implemented and evidenced by files, tests, or implementation_notes.
- Do not include markers merely because they were requested.
- If a required marker cannot be implemented due to missing context, omit it from included_markers and explain in implementation_notes and known_limitations.

Current project snapshot rules:
- You are modifying an existing project, not creating a standalone project.
- CURRENT_PROJECT_SNAPSHOT is the current source of truth for the codebase.
- Return complete file contents only for files you create or modify.
- If modifying an existing file, preserve unrelated existing behavior.
- Do not recreate the project from scratch.
- Do not delete existing routes, imports, models, tests, configs, or README sections unless the task explicitly requires it.
- Do not create duplicate app roots, duplicate entrypoints, alternate frameworks, or isolated examples.
- Do not create files outside the structure defined by README.md and project_context.json unless the task requires a new path and the context allows it.
- If you add a new API router, component, service, model, or config, update the necessary existing wiring file from CURRENT_PROJECT_SNAPSHOT.
- If you add a new dependency, update the dependency file such as requirements.txt, package.json, pyproject.toml, go.mod, etc.
- If you change setup commands, run commands, public API/data/UI contracts, directory structure, or architecture rules, update both README.md and project_context.json.
- If project_context.json and README.md conflict, prefer project_context.json for machine-readable paths and commands, but update README.md to match.

Foundation/TASK-001 rules:
- If task_id is TASK-001 or task title indicates codebase setup/project foundation, create README.md, project_context.json, and minimal skeleton project.
- README.md must be returned as file_path exactly "README.md".
- project_context.json must be returned as file_path exactly "project_context.json".

Non-foundation task rules:
- For any task after TASK-001, do not generate a new standalone project.
- Read PROJECT CONTEXT and CURRENT_PROJECT_SNAPSHOT before choosing file paths.
- Modify or add only files necessary for the task.
- Prefer extending existing modules over creating new parallel modules.
- If required context is missing, document it in known_limitations instead of inventing a new architecture.

Output rules:
- Return DevResult only.
- files must contain implementation file objects only.
- unit_tests must contain test file objects only.
- Every item in files and unit_tests must be an object with file_path and code.
- Return complete file content, not diffs.
- Do not include markdown fences in code fields.

Negative constraints:
- Do not return markdown code fences in JSON fields.
- Do not include extra fields.
- Do not use empty strings for required fields.
- Do not include strings like "unit_tests", "tests", "NULL", "null", or "None" inside files or unit_tests arrays.
- Do not claim tests were executed unless execution evidence is explicitly provided.

Return only the structured output matching the schema.
    """.strip(),
),
           (
    "human",
    f"""
Implement this task by modifying the current project snapshot. Do not create an independent project.

<TASK>
{task_json}
</TASK>

<ACCEPTANCE_CRITERIA>
{ac_json}
</ACCEPTANCE_CRITERIA>

<BUG_REPORTS>
{bugs_json}
</BUG_REPORTS>

{context_block}

{snapshot_block}

Instructions:
- Return only the structured output matching the DevResult schema.
- Ensure task_id matches the task.
- Read PROJECT_CONTEXT and CURRENT_PROJECT_SNAPSHOT before choosing file paths.
- If this is the foundation/first task (TASK-001), include `README.md`, `project_context.json`, and the initial project skeleton in files.
- If this task changes shared project context (architecture, commands, contracts, structure), include updated `README.md` and `project_context.json` in files.
- Implement this task by modifying the current project snapshot. Do not create an independent project.
- files must contain only implementation file objects.
- unit_tests must contain only test file objects.
- Every item in files and unit_tests must be an object with file_path and code.
- Return complete file content for each file, not diffs.
- Do not put strings, "unit_tests", "tests", "NULL", "null", "None", or placeholders inside files or unit_tests.
- If no unit tests are appropriate, use "unit_tests": [].
- If there are no known limitations, use "known_limitations": [].
    """.strip(),
),
        ]
        return invoke_structured_with_retry(structured_llm, messages)


class QCAgent(OpenAIAPIAgent):
    """Quality Control agent responsible for validating implementations."""
    def validate(
        self,
        task: dict,
        acceptance_criteria: list[str],
        dev_output: dict,
        project_context: str | None = None,
        current_project_snapshot: dict[str, str] | None = None,
    ) -> QCResult:
        from app.agents.project_utils import build_project_snapshot_for_prompt

        structured_llm = self.llm.with_structured_output(
            QCResult,
            method="function_calling",
        )

        # Standardize payload handling to be robust against missing fields
        task_data = task or {}
        ac_data = acceptance_criteria or []

        dev_output_data = dict(dev_output or {})
        dev_output_data.setdefault("included_markers", [])
        dev_output_data.setdefault("known_limitations", [])
        dev_output_data.setdefault("files", [])
        dev_output_data.setdefault("unit_tests", [])

        task_json = to_pretty_json(task_data)
        ac_json = to_pretty_json(ac_data)
        dev_output_json = to_pretty_json(dev_output_data)

        context_block = ""
        if project_context:
            context_block = f"\n<PROJECT_CONTEXT>\n{project_context}\n</PROJECT_CONTEXT>\n"

        snapshot_after_task_block = ""
        if current_project_snapshot:
            safe_snapshot = build_project_snapshot_for_prompt(current_project_snapshot)
            snapshot_json = to_pretty_json(safe_snapshot)
            snapshot_after_task_block = f"\n<CURRENT_PROJECT_SNAPSHOT_AFTER_TASK>\n{snapshot_json}\n</CURRENT_PROJECT_SNAPSHOT_AFTER_TASK>\n"

        messages = [
            (
                "system",
                """
You are a strict quality control engineer validating a developer's implementation.

Your job is to determine whether the implementation satisfies the task acceptance criteria based only on the provided evidence.

You must return structured output that exactly matches the QCResult schema.

You will receive:
1. TASK
2. ACCEPTANCE_CRITERIA
3. DEVELOPER_OUTPUT
4. PROJECT CONTEXT (README.md) - if available

Core principles:
1. Be evidence-based.
2. Evaluate every acceptance criterion individually.
3. Evaluate every required marker individually.
4. Use only provided task context, acceptance criteria, developer files, unit tests, implementation notes, included_markers, known_limitations, and the project context (README.md).
5. Ensure the implementation is consistent with the project foundation, directory structure, and architectural goals defined in the README.
6. For TASK-001 or foundation tasks, verify that `README.md` is included and contains enough project context for later agents: goal, tech stack, architecture, directory structure, contracts, setup/test commands, and future-agent rules.
7. For later tasks, fail or mark unsupported when code is detached from the README-defined codebase, introduces a duplicate framework/root, or changes shared contracts without updating `README.md`.
6. If there is no clear evidence that a criterion is satisfied, mark it as UNSUPPORTED or FAILED.
8. Do not assume hidden behavior exists.
9. Do not fail the implementation for requirements that were never explicitly requested.
10. Feedback must be specific, actionable, and suitable for sending back to the developer.

Schema compliance rules:
- Return only fields defined by the QCResult schema.
- QCResult must include:
  - task_id
  - passed
  - status
  - criteria_results
  - marker_results
  - validation_report
  - failed_criteria
  - severity
- task_id must match TASK task_id and use format TASK-001.
- status must be one of:
  - PASSED
  - FAILED
  - NEEDS_REVISION
- severity must be one of:
  - LOW
  - MEDIUM
  - HIGH
  - CRITICAL
- Each criteria_results item must include:
  - criterion
  - status
  - evidence
  - feedback
- Each marker_results item must include:
  - marker
  - status
  - evidence
  - feedback
- Criterion and marker status must be one of:
  - PASSED
  - FAILED
  - UNSUPPORTED
- If passed is true:
  - status must be PASSED
  - failed_criteria must be empty
  - all criteria_results must have status PASSED
  - all marker_results must have status PASSED
  - severity should be LOW
- If passed is false:
  - status must be FAILED or NEEDS_REVISION
  - failed_criteria must include failed or unsupported criteria
- Do not include empty strings in failed_criteria.

Static validation rule:
- You do not execute code or tests.
- Do not claim tests passed at runtime.
- Judge only whether the provided implementation and tests would reasonably verify the stated behavior based on static code analysis.
- NEVER fail a task simply because the developer states they did not run the code, could not execute Docker, or collect runtime logs. You are evaluating static code completeness, not a live deployment.
- If a test is incomplete, superficial, inconsistent, or unable to verify behavior, treat evidence as insufficient.

Input consistency rules:
- If TASK contains acceptance criteria that conflict with separate ACCEPTANCE_CRITERIA input, validate against the separate ACCEPTANCE_CRITERIA input.
- Mention the inconsistency in validation_report.
- If required_markers conflict with acceptance criteria, only fail marker coverage when the marker is within the stated task scope or acceptance criteria.
- Do not expand validation beyond the task and acceptance criteria.

Acceptance criterion validation:
- Evaluate every item in ACCEPTANCE_CRITERIA.
- For each criterion, create one criteria_results item.
- status PASSED means the implementation provides clear evidence.
- status FAILED means implementation contradicts or does not satisfy the criterion.
- status UNSUPPORTED means evidence is insufficient to determine satisfaction.
- Unit tests strengthen evidence but do not compensate for missing implementation.
- Implementation notes help explain intent but do not compensate for missing code when code is required.
- If code implements behavior but tests do not cover it:
  - If tests are required by criteria or markers, mark FAILED or UNSUPPORTED.
  - Otherwise, mention the test gap in validation_report.
- If tests claim behavior that code does not implement, mark the criterion FAILED.

Marker validation:
- Read required_markers from TASK.
- Evaluate every required_marker.
- For each required_marker, create one marker_results item.
- Compare required_markers against:
  - included_markers
  - implementation files
  - unit test files
  - implementation_notes
- A marker is PASSED only if it is claimed or clearly evidenced and supported by actual implementation evidence.
- included_markers alone are not sufficient if code/tests/notes do not support the marker.
- If a marker is listed in included_markers but not evidenced, mark it UNSUPPORTED.
- If a required marker is missing from included_markers and not evidenced, mark it FAILED or UNSUPPORTED.
- If implementation clearly evidences a marker but forgot to list it, mention the inconsistency in validation_report.
- Do not invent markers not present in TASK required_markers.

Developer output validation:
- Validate evidence from DEVELOPER_OUTPUT files, unit_tests, implementation_notes, included_markers, and known_limitations.
- If files are missing or empty, fail validation.
- If implementation_notes disclose a limitation that prevents satisfying a criterion, mark the affected criterion FAILED or UNSUPPORTED.
- If unit tests are absent but task requires unit-test markers, fail or mark unsupported for those markers.
- Different task teams may produce different artifact types:
  - product tasks may produce decision/clarification documentation.
  - qa tasks may produce test plans or test files.
  - devops/platform tasks may produce Docker, compose, CI/CD, or config files.
  - data tasks may produce migrations, schema files, seed data, or data-layer files.
  - frontend tasks may produce UI components, pages, client services, or tests.
  - backend tasks may produce APIs, services, repositories, models, or tests.
- Validate against the task acceptance criteria and required markers, not against a single expected artifact type.

Bug report validation:
- If bug reports are represented in the task context or developer notes, verify that the implementation notes explain how they were fixed.
- Verify that tests cover bug fixes where reasonable.
- Do not require bug fixes that were not provided.

Validation report must:
1. Summarize whether the task passed or failed.
2. Explain what passed.
3. Explain what failed or was unsupported.
4. Mention acceptance criteria coverage.
5. Mention marker coverage.
6. Mention test adequacy.
7. Mention any input inconsistency.
8. Mention any known limitations.
9. Provide actionable feedback for the developer.

Negative constraints:
- Do not assume behavior exists if not evidenced.
- Do not create new requirements.
- Do not fail for style preferences unless they affect correctness, testability, or acceptance criteria.
- Do not require infrastructure, deployment config, or unrelated files unless explicitly required.
- Do not give vague feedback such as "needs improvement" without concrete reasons.

Project snapshot validation:
- You are validating the task after its output has been applied to the current project snapshot.
- Validate not only the returned DevResult, but also whether the task is correctly integrated into the project snapshot.
- Check that new files are wired into existing entrypoints/configs when needed.
- Check that imports reference files/modules present in CURRENT_PROJECT_SNAPSHOT_AFTER_TASK.
- Check that tests reference actual modules/functions/classes in the snapshot.
- Check that README.md and project_context.json remain consistent if the task changed architecture, commands, contracts, or structure.
- Check that the task did not create a duplicate app root, duplicate entrypoint, alternate framework, or isolated example project.
- Check that previous project structure was preserved.
- Do not fail only because code was not executed. This is static validation.
- If integration evidence is missing, mark the relevant criterion or marker FAILED/UNSUPPORTED with actionable feedback.

Ambiguity handling:
- If task wording is ambiguous, do not expand scope.
- Judge only against the narrowest reasonable interpretation supported by the input.
- If ambiguity prevents validation, mark the affected criterion as UNSUPPORTED and explain what evidence is missing.

Before returning, verify:
- task_id is valid.
- criteria_results includes one item per acceptance criterion.
- marker_results includes one item per required marker.
- passed/status/failed_criteria are internally consistent.
- validation_report is specific and actionable.
- no extra fields are included.

Return only the structured output matching the schema.
                """.strip(),
            ),
            (
                "human",
                f"""
Validate the developer output below against the project snapshot after applying the task.
Instructions:
- Validate against the provided ACCEPTANCE_CRITERIA.
- Use evidence from TASK, DEVELOPER_OUTPUT, and CURRENT_PROJECT_SNAPSHOT_AFTER_TASK.
- Check that the task output integrates correctly into the project snapshot.
- Check that new files are wired into existing entrypoints/configs.
- Check that imports reference existing files/modules in the snapshot.
- Check that the task did not create a duplicate app root, entrypoint, or alternate framework.
- Do not execute code.
- Do not assume hidden behavior.
- Produce one criteria_results item for each acceptance criterion.
- Produce one marker_results item for each required_marker in TASK.
- Check required_markers against included_markers and actual evidence.
- Ensure passed, status, failed_criteria, criteria_results, and marker_results are internally consistent.
- Return only the structured output.

<TASK>
{task_json}
</TASK>

<ACCEPTANCE_CRITERIA>
{ac_json}
</ACCEPTANCE_CRITERIA>

<DEVELOPER_OUTPUT>
{dev_output_json}
</DEVELOPER_OUTPUT>

{context_block}

{snapshot_after_task_block}
                """.strip(),
            ),
        ]
        return invoke_structured_with_retry(structured_llm, messages)
