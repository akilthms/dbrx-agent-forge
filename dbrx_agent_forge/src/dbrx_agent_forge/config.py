from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl

class Settings(BaseSettings):
    experiment_path: str
    external_storage_location: str

    class Config:
        env_file = ".env"


