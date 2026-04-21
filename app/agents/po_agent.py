from app.agents.base import POResult, StoryOut, TaskOut
from app.agents.mocks import extract_requirement_lines, normalize_marker, split_into_chunks, summarize_text


class MockPOAgent:
    def analyze(self, brd_content: str) -> POResult:
        requirements = extract_requirement_lines(brd_content)
        if not requirements:
            requirements = ['system should support create', 'system should support update', 'system should support validation']

        story_chunks = split_into_chunks(requirements, 2)
        stories: list[StoryOut] = []
        backlog_items: list[dict] = []
        implementation_tasks: list[TaskOut] = []

        for idx, chunk in enumerate(story_chunks, start=1):
            criteria = [f'System satisfies: {item}' for item in chunk]
            story_description = 'As a business user, I want ' + '; '.join(chunk)
            stories.append(
                StoryOut(
                    title=f'User story {idx}',
                    description=story_description,
                    acceptance_criteria=criteria,
                    priority='HIGH' if idx == 1 else 'MEDIUM',
                )
            )
            team = 'backend' if idx % 2 == 1 else 'frontend'
            backlog_items.append({'title': f'Backlog item {idx}', 'description': story_description, 'team': team})
            implementation_tasks.append(
                TaskOut(
                    title=f'{team.title()} implementation task {idx}',
                    description='Implement: ' + '; '.join(chunk),
                    assignee_team=team,
                    acceptance_criteria=criteria,
                    required_markers=[normalize_marker(item) for item in chunk],
                    input_context={'requirements': chunk, 'expected_failures': 1 if idx == 1 else 0},
                )
            )

        return POResult(
            feature_summary=summarize_text(requirements, max_items=4),
            user_stories=stories,
            backlog_items=backlog_items,
            implementation_tasks=implementation_tasks,
        )
