from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    app_name: str = "nombre_proyecto"
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = Path(__file__).parent.parent / ".env.dev"
        case_sensitive = False

settings = Settings()