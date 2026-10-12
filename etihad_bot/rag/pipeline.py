"""RAG pipeline: split -> embed (BGE) -> ChromaDB -> retrieve -> cross-encoder rerank."""
import math
from dataclasses import asdict, dataclass
from functools import lru_cache

import chromadb
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document

from etihad_bot.config import Settings, get_settings
from etihad_bot.rag.embeddings import get_embeddings
from etihad_bot.rag.splitting import split_documents


@dataclass
class Chunk:
    id: str
    text: str
    document: str
    section: str
    page: int
    source: str
    score: float  # rerank probability (reranker on) or cosine similarity (off)

    def citation(self) -> dict:
        return {"chunk_id": self.id, "document": self.document, "section": self.section,
                "page": self.page, "source": self.source}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@lru_cache(maxsize=1)
def _reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, *, chunk_size: int | None = None,
                 chunk_overlap: int | None = None, use_reranker: bool | None = None,
                 retrieve_k: int | None = None, top_n: int | None = None):
        s = settings or get_settings()
        self.s = s
        self.chunk_size = chunk_size or s.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else s.chunk_overlap
        self.use_reranker = s.use_reranker if use_reranker is None else use_reranker
        self.retrieve_k = retrieve_k or s.retrieve_k
        self.top_n = top_n or s.rerank_top_n
        self.collection = f"{s.collection_prefix}_cs{self.chunk_size}_co{self.chunk_overlap}"
        self._store: Chroma | None = None

    @property
    def store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection, embedding_function=get_embeddings(),
                client=chromadb.PersistentClient(path=self.s.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"})
        return self._store

    # ---- ingestion -------------------------------------------------------
    def ingest(self, docs: list[Document]) -> int:
        chunks = split_documents(docs, self.chunk_size, self.chunk_overlap)
        ids = [c.metadata["chunk_id"] for c in chunks]
        self.store.add_documents(chunks, ids=ids)  # upsert -> idempotent re-ingestion
        return len(chunks)

    def count(self) -> int:
        return self.store._collection.count()

    # ---- retrieval -------------------------------------------------------
    def search(self, query: str) -> list[Chunk]:
        """Dense retrieval from ChromaDB (top retrieve_k by cosine similarity)."""
        hits = self.store.similarity_search_with_score(query, k=self.retrieve_k)
        return [self._to_chunk(d, 1.0 - dist) for d, dist in hits]  # cosine distance -> similarity

    def rerank(self, query: str, cands: list[Chunk]) -> list[Chunk]:
        """Cross-encoder rerank (if enabled) and truncate to top_n."""
        if cands and self.use_reranker:
            # CrossEncoder.predict applies a sigmoid by default; ask for raw logits so the single
            # sigmoid below gives a true 0-1 relevance probability (applying it twice squashed
            # every score into 0.5-0.73 and made the relevance gate a no-op).
            scores = _reranker(self.s.reranker_model).predict(
                [(query, c.text) for c in cands], activation_fn=torch.nn.Identity())
            for c, sc in zip(cands, scores):
                c.score = 1.0 / (1.0 + math.exp(-float(sc)))
            cands = sorted(cands, key=lambda c: c.score, reverse=True)
        return cands[: self.top_n]

    def retrieve(self, query: str) -> list[Chunk]:
        return self.rerank(query, self.search(query))

    def is_relevant(self, chunks: list[Chunk]) -> bool:
        if not chunks:
            return False
        floor = self.s.min_rerank_score if self.use_reranker else self.s.min_similarity
        return chunks[0].score >= floor

    @staticmethod
    def _to_chunk(d: Document, score: float) -> Chunk:
        m = d.metadata
        return Chunk(id=m["chunk_id"], text=d.page_content, document=m.get("doc_title", ""),
                     section=m.get("section", ""), page=int(m.get("page", 0)),
                     source=m.get("source", ""), score=score)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()

