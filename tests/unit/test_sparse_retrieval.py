"""Deterministic BM25 and CJK tokenization tests."""

from cutagent.retrieval.sparse import BM25Index, tokenize


def test_multilingual_tokenizer_has_cjk_unigrams_and_bigrams() -> None:
    tokens = tokenize("Find 蓝色圆形 EXIT")
    assert {"find", "蓝", "色", "圆", "形", "蓝色", "圆形", "exit"}.issubset(tokens)


def test_bm25_is_deterministic_and_field_sensitive() -> None:
    documents = ("red square moves right", "blue circle stays still", "red circle")
    first = BM25Index.build(documents).scores("red square")
    second = BM25Index.build(documents).scores("red square")
    assert first == second
    assert first[0] > first[2] > first[1]
