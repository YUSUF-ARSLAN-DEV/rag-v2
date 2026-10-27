import pytest

from chunker import chunk


def make_text(n_tokens):
    # " hello" is exactly one token in cl100k_base, so n repeats = n tokens (hand-computable)
    return " hello" * n_tokens


@pytest.fixture
def doc_400():
    return make_text(400)


@pytest.mark.parametrize("n_tokens, expected_chunks", [
    (100, 1),   # shorter than one chunk -> one chunk
    (200, 2),   # current behaviour: starts at 0 and 150, so the last 50 tokens appear twice
    (400, 3),   # starts at 0, 150, 300 (step = 200 - 50)
])
def test_chunk_count_and_max_size(n_tokens, expected_chunks):
    chunks = chunk([make_text(n_tokens)], user_id=1, chunk_size=200, overlap=50)
    assert len(chunks) == expected_chunks
    assert all(c["token_count"] <= 200 for c in chunks)


def test_chunk_starts_are_step_apart(doc_400):
    chunks = chunk([doc_400], user_id=1, chunk_size=200, overlap=50)
    assert [c["start_token"] for c in chunks] == [0, 150, 300]


def test_two_documents_get_unique_ids_and_user_id():
    chunks = chunk([make_text(100), make_text(100)], user_id=7)
    ids = [c["id"] for c in chunks]
    assert len(ids) == 2
    assert len(set(ids)) == len(ids)
    assert all(c["user_id"] == 7 for c in chunks)


def test_empty_input_gives_no_chunks():
    assert chunk([""], user_id=1) == []
    assert chunk([], user_id=1) == []
