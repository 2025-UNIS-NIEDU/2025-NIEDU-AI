from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_HOST: str | None = None
    DB_PORT: int | None = 5432
    DB_NAME: str | None = None
    DB_USERNAME: str | None = None
    DB_PASSWORD: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()