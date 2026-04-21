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


@lru_cache
def get_settings() -> Settings:
    return Settings()
