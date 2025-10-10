from pydantic_settings import BaseSettings
from pydantic import AnyUrl

class Settings(BaseSettings):
    experiment_path: str


    class Config:
        env_file = ".env"


