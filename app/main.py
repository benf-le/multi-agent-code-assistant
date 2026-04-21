from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.models  # noqa: F401
from app.api.routes.tasks import router as task_router
from app.api.routes.workflows import router as workflow_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import engine

configure_logging()
settings = get_settings()
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.include_router(workflow_router, prefix=settings.api_prefix)
app.include_router(task_router, prefix=settings.api_prefix)


@app.get('/health')
def health():
    return {'status': 'ok'}
