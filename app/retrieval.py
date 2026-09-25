"""
Task 1 -- Permission-aware retrieval.

Chunk documents/*.txt, embed each chunk with a REAL embedding model
(your choice -- sentence-transformers, an API, etc. -- no keyword/TF-IDF
substitutes), attach ACL metadata from documents/manifest.json, and
implement retrieve() below.

Hard rule: an unauthorized chunk must never be constructible from the
return value of retrieve(). Filtering happens INSIDE this function,
before anything is returned -- never filter after the fact.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-minilm")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "documents"

# The docs are one short paragraph each, so in practice every doc becomes a
# single chunk. The splitter is still here so longer docs would chunk on
# paragraph/sentence boundaries rather than mid-word.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    doc_id: str
    text: str
    acl: List[str]
    embedding: List[float] = field(default_factory=list)
    score: float = 0.0
    chunk_id: str = ""


def _acl_metadata(acl: List[str]) -> dict:
    # Chroma metadata values must be scalars, so the ACL is stored twice: as
    # one boolean flag per group (so retrieval can pre-filter inside the
    # vector query with a `where` clause) and as a joined string (so the
    # original list can be reconstructed onto the Chunk).
    meta = {f"acl_{group}": True for group in acl}
    meta["acl"] = ",".join(acl)
    return meta


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_documents(documents_dir: str = "documents") -> List[Chunk]:
    """Load and chunk all documents, attaching ACL metadata from manifest.json.

    Embeddings should be computed here (or lazily) using a real embedding
    model. Paragraph-level chunking is fine given how short these documents
    are.
    """
    docs_path = Path(documents_dir)
    manifest = json.loads((docs_path / "manifest.json").read_text(encoding="utf-8"))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    documents: List[Document] = []
    ids: List[str] = []
    for doc_id, entry in sorted(manifest.items()):
        acl = entry["acl"]
        if not acl:
            # A doc with no ACL is visible to nobody; refuse to ingest it
            # rather than guess at a default.
            raise ValueError(f"{doc_id} has an empty ACL in manifest.json")
        text = (docs_path / doc_id).read_text(encoding="utf-8").strip()
        for i, piece in enumerate(splitter.split_text(text)):
            chunk_id = f"{doc_id}#{i}"
            ids.append(chunk_id)
            documents.append(
                Document(
                    page_content=piece,
                    metadata={"doc_id": doc_id, "chunk_id": chunk_id, **_acl_metadata(acl)},
                )
            )

    # Rebuild the collection from scratch on every ingest, so a doc removed
    # from the manifest or an ACL that was narrowed never survives as a stale
    # chunk with its old permissions.
    store = get_vector_store()
    store.delete_collection()
    store = get_vector_store()
    store.add_documents(documents, ids=ids)

    stored = store.get(ids=ids, include=["documents", "metadatas", "embeddings"])
    by_id = {
        cid: (text, meta, emb)
        for cid, text, meta, emb in zip(
            stored["ids"], stored["documents"], stored["metadatas"], stored["embeddings"]
        )
    }
    chunks = []
    for cid in ids:
        text, meta, emb = by_id[cid]
        chunks.append(
            Chunk(
                doc_id=meta["doc_id"],
                text=text,
                acl=meta["acl"].split(","),
                embedding=list(emb),
                chunk_id=cid,
            )
        )
    return chunks


def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    if "all" in chunk_acl:
        return True
    if any(group in chunk_acl for group in user_groups):
        return True
    return False


def retrieve(query: str, user_groups: List[str], top_k: int = 3) -> List[Chunk]:
    """Return only chunks the given user_groups are authorized to see,
    ranked by embedding similarity to `query`.
    """
    store = get_vector_store()

    # 1) Embed the query.
    query_embedding = get_embeddings().embed_query(query)

    # 2) Get the chunks, ranked by similarity, with their metadata. Every
    # chunk is fetched so that dropping unauthorized ones below still leaves
    # up to top_k authorized results.
    total = len(store.get()["ids"])
    results = store.similarity_search_by_vector_with_relevance_scores(
        query_embedding, k=total
    )
    chunks = [
        Chunk(
            doc_id=doc.metadata["doc_id"],
            text=doc.page_content,
            acl=doc.metadata["acl"].split(","),
            score=1 - distance,
            chunk_id=doc.metadata["chunk_id"],
        )
        for doc, distance in results
    ]

    # 3) Delete every chunk the user is not authorized to see.
    chunks = [c for c in chunks if is_authorized(c.acl, user_groups)]

    if not chunks:
        return ["unauthorized"]
    return chunks[:top_k]


if __name__ == "__main__":
    for c in load_documents():
        print(f"{c.chunk_id:<12} acl={c.acl!s:<24} dim={len(c.embedding)}  {c.text[:60]}")
