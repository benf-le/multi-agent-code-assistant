import logging
import json
from typing import Any

logger = logging.getLogger("workflow_logger")

class WorkflowLogger:
    @staticmethod
    def _format(event: str, **kwargs) -> str:
        data = {"event": event}
        data.update(kwargs)
        # Truncate large values
        for k, v in data.items():
            if isinstance(v, str) and len(v) > 500:
                data[k] = v[:500] + "..."
        return json.dumps(data)

    @classmethod
    def log(cls, event: str, level: int = logging.INFO, **kwargs):
        msg = cls._format(event, **kwargs)
        logger.log(level, msg)

    @classmethod
    def info(cls, event: str, **kwargs):
        cls.log(event, level=logging.INFO, **kwargs)

    @classmethod
    def error(cls, event: str, **kwargs):
        cls.log(event, level=logging.ERROR, **kwargs)

    @classmethod
    def warning(cls, event: str, **kwargs):
        cls.log(event, level=logging.WARNING, **kwargs)
