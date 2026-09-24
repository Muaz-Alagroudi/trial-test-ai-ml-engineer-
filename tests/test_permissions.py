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
    raise NotImplementedError


def test_all_only_user_never_sees_restricted_chunks():
    # user_c (groups: ["all"]) must never receive a finance- or
    # propteam-only chunk.
    raise NotImplementedError
