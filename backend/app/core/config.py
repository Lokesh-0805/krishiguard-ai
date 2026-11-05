"""Runtime configuration (12-factor: everything overridable via environment / .env)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KG_", env_file=".env", extra="ignore")

    app_name: str = "KrishiGuard AI API"
    environment: str = "development"
    log_level: str = "INFO"

    # model artefacts (produced by ml/evaluation/export_serving.py)
    serving_dir: Path = PROJECT_ROOT / "ml/artifacts/serving"
    evaluation_dir: Path = PROJECT_ROOT / "ml/artifacts/evaluation"
    advisory_path: Path = PROJECT_ROOT / "advisory/diseases.json"
    inference_threads: int = 2

    # upload policy
    max_upload_mb: float = 8.0
    max_batch_files: int = 8
    min_image_side: int = 64
    allowed_mime: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    blur_warning_threshold: float | None = None  # None -> value from serving config

    # persistence
    database_url: str = "postgresql+psycopg://krishiguard:krishiguard_dev@localhost:5432/krishiguard"

    # http
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    rate_limit_predict: str = "30/minute"
    rate_limit_default: str = "120/minute"
    rate_limit_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
