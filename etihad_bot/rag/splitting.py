import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs: list[Document], chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        key = f"{c.metadata['source']}|{c.metadata['page']}|{i}|{c.page_content[:64]}"
        c.metadata["chunk_id"] = hashlib.sha1(key.encode()).hexdigest()[:16]
        c.metadata["chunk_index"] = i
    return chunks
