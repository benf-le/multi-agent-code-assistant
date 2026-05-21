from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    app_name: str = Field(default='Multi-Agent LangGraph Control Plane', alias='APP_NAME')
    api_prefix: str = Field(default='/api/v1', alias='API_PREFIX')
    database_url: str = Field(default='sqlite+pysqlite:///./multi_agent_demo.db', alias='DATABASE_URL')
    log_level: str = Field(default='INFO', alias='LOG_LEVEL')
    default_max_retry: int = Field(default=2, alias='DEFAULT_MAX_RETRY')
    poll_interval_ms: int = Field(default=3000, alias='POLL_INTERVAL_MS')
    openai_api_key: str = Field(default='', alias='OPENAI_API_KEY')
    openai_base_url: str = Field(default='https://api.openai.com/v1', alias='OPENAI_BASE_URL')
    openai_model: str = Field(default='gpt-4o', alias='OPENAI_MODEL')
    # Safety guard for LangGraph recursion. With 1-task-per-run design,
    # worst case is ~5 steps per retry cycle.  max_retry=3 → 20 steps max.
    # 50 is generous but safe.
    graph_recursion_limit: int = Field(default=50, alias='GRAPH_RECURSION_LIMIT')
    # Max identical loop signatures before force-blocking a task
    max_loop_signatures: int = Field(default=3, alias='MAX_LOOP_SIGNATURES')
    # Max retry attempts for PO review failures before marking workflow as FAILED.
    # Set to 0 to fail immediately on first NEEDS_REVISION.
    po_review_max_retries: int = Field(default=3, alias='PO_REVIEW_MAX_RETRIES')


@lru_cache
def get_settings() -> Settings:
    return Settings()
