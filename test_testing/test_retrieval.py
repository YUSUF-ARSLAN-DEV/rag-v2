import pytest

from embedder import RFF_TOP_PICKS
from pipelines import snippets_hit


@pytest.mark.parametrize("snippets, texts, expected", [
    ("fall risk", ["Fall-Risk assessment"], True),            # case and punctuation are ignored
    ("moni- toring", ["regular monitoring"], True),            # PDF hyphenation noise is ignored
    (["fall", "bed"], ["a fall", "the bed"], True),            # every snippet found, even in different texts
    (["fall", "stairs"], ["a fall"], False),                   # one snippet missing -> miss
])
def test_snippets_hit(snippets, texts, expected):
    assert snippets_hit(snippets, texts) is expected


def test_chunk_found_by_both_searches_ranks_first():
    faiss_rows = [(1, "a"), (2, "b")]
    bm25_rows = [(3, "c"), (2, "b")]   # chunk 2 appears in both lists, the others in one
    assert RFF_TOP_PICKS(faiss_rows, bm25_rows)[0] == "b"


def test_rrf_respects_the_limit():
    faiss_rows = [(1, "a"), (2, "b"), (3, "c")]
    assert len(RFF_TOP_PICKS(faiss_rows, [], limit=2)) == 2
