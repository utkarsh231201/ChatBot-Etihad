from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_price_in_per_m: float = 0.15  # USD per 1M tokens - verify against current pricing
    llm_price_out_per_m: float = 0.60

    # RAG
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieve_k: int = 20
    rerank_top_n: int = 5
    use_reranker: bool = True
    min_rerank_score: float = 0.05  # sigmoid score below which we refuse
    soft_min_score: float = 0.3  # stricter relevance bar for questions the router called unsupported
    min_similarity: float = 0.35  # cosine similarity floor when reranker is off
    enable_grounding_check: bool = True
    chroma_dir: str = str(ROOT / "data" / "chroma")
    collection_prefix: str = "etihad"
    policy_pdf: str = str(ROOT / "data" / "pdfs" / "etihad_policy1.pdf")  # the only knowledge source

    # Flight API / MCP
    aviationstack_api_key: str = ""
    flight_api_mock: bool = False

    # Auth
    auth_pepper: str = "change-me"  # set a long random value in .env; changing it orphans old histories
    session_ttl_seconds: int = 24 * 3600

    # Persistence
    sqlite_dir: str = str(ROOT / "data" / "sqlite")

    @property
    def checkpoint_db(self) -> str:
        return str(Path(self.sqlite_dir) / "checkpoints.sqlite")

    @property
    def app_db(self) -> str:
        return str(Path(self.sqlite_dir) / "app.sqlite")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    Path(s.sqlite_dir).mkdir(parents=True, exist_ok=True)
    Path(s.chroma_dir).mkdir(parents=True, exist_ok=True)
    return s
