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

from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    doc_id: str
    text: str
    acl: List[str]
    embedding: List[float] = field(default_factory=list)
    score: float = 0.0


def load_documents(documents_dir: str = "documents") -> List[Chunk]:
    """Load and chunk all documents, attaching ACL metadata from manifest.json.

    Embeddings should be computed here (or lazily) using a real embedding
    model. Paragraph-level chunking is fine given how short these documents
    are.
    """
    raise NotImplementedError("Implement chunking + embedding + ACL attachment")


def is_authorized(chunk_acl: List[str], user_groups: List[str]) -> bool:
    """A chunk is authorized for a user if 'all' is in its acl, or if any
    of the user's groups intersects the chunk's acl.
    """
    raise NotImplementedError("Implement the authorization check")


def retrieve(query: str, user_groups: List[str], top_k: int = 3) -> List[Chunk]:
    """Return only chunks the given user_groups are authorized to see,
    ranked by embedding similarity to `query`.
    """
    raise NotImplementedError("Implement permission-filtered, ranked retrieval")
