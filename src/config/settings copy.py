from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Database ---
    DB_HOST: str | None = None
    DB_PORT: int | None = 5432
    DB_NAME: str | None = None
    DB_USERNAME: str | None = None
    DB_PASSWORD: str | None = None

    # --- API Keys ---
    OPENAI_API_KEY: str | None = None
    HUGGINGFACEHUB_API_TOKEN: str | None = None
    GOOGLE_CSE_API_KEY: str | None = None
    GOOGLE_CSE_CX_NEWS: str | None = None
    GOOGLE_CSE_CX_GOV: str | None = None
    GOOGLE_CSE_CX_DICT: str | None = None
    DEEPSEARCH_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()