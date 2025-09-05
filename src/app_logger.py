import logging
import logging.config
import os
from pathlib import Path

APP_NAME = "ladung_va"

def _default_logging_dict(level: str = "INFO") -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "verbose": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                # Minimal JSON-style log, good for log processors
                "format": '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            # Example rotating file handler
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": "verbose",
                "filename": str(Path(os.getenv("LOG_DIR", ".")) / f"{APP_NAME}.log"),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            # Root logger defaults — keep quiet libraries at INFO or WARNING
            "": {
                "level": level,
                "handlers": ["console"],
            },
            # App-specific base logger; child loggers will inherit
            APP_NAME: {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

def _configure_logging():
    # Allow override via env variables
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.config.dictConfig(_default_logging_dict(level))

# Configure once at import
_configure_logging()

# Export a module-level logger for the app
logger = logging.getLogger(APP_NAME)
