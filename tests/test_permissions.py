"""
Task 3 -- A few quick assertions proving permission enforcement works.

This is deliberately light (2-3 checks), not a full test suite -- the
point is the hard requirement they prove, not test-suite completeness.
"""

from app.retrieval import retrieve
from app.users import TEST_USERS


def test_different_users_get_differently_scoped_results():
    # user_a (finance) and user_b (propteam) ask the same query;
    # assert their retrieved chunk sets differ correctly.
    query = "Maple Ridge Apartments financial summary and maintenance log"
    a_docs = {c.doc_id for c in retrieve(query, TEST_USERS["user_a"], top_k=10)}
    b_docs = {c.doc_id for c in retrieve(query, TEST_USERS["user_b"], top_k=10)}

    assert a_docs != b_docs
    # Finance-only docs reach user_a but never user_b, and the reverse.
    assert {"doc1.txt", "doc4.txt"} <= a_docs and "doc3.txt" not in a_docs
    assert "doc3.txt" in b_docs and not b_docs & {"doc1.txt", "doc4.txt"}


def test_all_only_user_never_sees_restricted_chunks():
    # user_c (groups: ["all"]) must never receive a finance- or
    # propteam-only chunk.
    queries = [
        "What was Maple Ridge Apartments' total revenue in Q3 2026?",
        "When was the HVAC at Maple Ridge Apartments Unit 4B last repaired?",
        "Oak Hill Plaza Unit 12 lease renewal",
    ]
    for query in queries:
        chunks = retrieve(query, TEST_USERS["user_c"], top_k=10)
        assert chunks != ["unauthorized"]
        for chunk in chunks:
            assert "all" in chunk.acl, f"{chunk.chunk_id} leaked to user_c"
