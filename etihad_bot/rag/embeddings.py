"""BGE embeddings (bge-large-en-v1.5) with the recommended query instruction."""
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from etihad_bot.config import get_settings

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32,
                                 show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(QUERY_INSTRUCTION + text, normalize_embeddings=True).tolist()


@lru_cache
def get_embeddings() -> BGEEmbeddings:
    return BGEEmbeddings(get_settings().embedding_model)
