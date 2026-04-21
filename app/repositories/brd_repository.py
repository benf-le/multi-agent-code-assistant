from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AcceptanceCriteria, BRD, Feature, UserStory


class BRDRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_brd(self, title: str, content: str) -> BRD:
        brd = BRD(title=title, content=content)
        self.session.add(brd)
        self.session.flush()
        return brd

    def get_brd(self, brd_id: int) -> BRD | None:
        return self.session.get(BRD, brd_id)

    def create_feature(self, workflow_id: int, summary: str) -> Feature:
        feature = Feature(workflow_id=workflow_id, summary=summary)
        self.session.add(feature)
        self.session.flush()
        return feature

    def create_user_story(self, workflow_id: int, title: str, description: str, priority: str = 'MEDIUM') -> UserStory:
        story = UserStory(workflow_id=workflow_id, title=title, description=description, priority=priority)
        self.session.add(story)
        self.session.flush()
        return story

    def create_acceptance_criterion(self, workflow_id: int, text: str, user_story_id: int | None = None, task_id: int | None = None) -> AcceptanceCriteria:
        criterion = AcceptanceCriteria(workflow_id=workflow_id, text=text, user_story_id=user_story_id, task_id=task_id)
        self.session.add(criterion)
        self.session.flush()
        return criterion

    def list_user_stories(self, workflow_id: int) -> list[UserStory]:
        stmt = select(UserStory).where(UserStory.workflow_id == workflow_id).order_by(UserStory.id)
        return list(self.session.scalars(stmt).all())
