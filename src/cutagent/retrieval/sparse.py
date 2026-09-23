"""Small deterministic BM25 implementation with multilingual tokenization."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(value: str) -> tuple[str, ...]:
    """Tokenize Latin words and CJK unigram/bigram units without external state."""

    output: list[str] = []
    for match in _TOKEN_PATTERN.finditer(value.casefold()):
        token = match.group(0)
        if all("\u3400" <= char <= "\u9fff" for char in token):
            output.extend(token)
            output.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            output.append(token)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class BM25Index:
    term_frequencies: tuple[Counter[str], ...]
    document_lengths: tuple[int, ...]
    document_frequencies: dict[str, int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, documents: tuple[str, ...]) -> BM25Index:
        term_frequencies = tuple(Counter(tokenize(document)) for document in documents)
        lengths = tuple(sum(frequencies.values()) for frequencies in term_frequencies)
        document_frequencies: Counter[str] = Counter()
        for frequencies in term_frequencies:
            document_frequencies.update(frequencies.keys())
        average = sum(lengths) / len(lengths) if lengths else 0.0
        return cls(term_frequencies, lengths, dict(document_frequencies), average)

    def scores(self, query: str) -> tuple[float, ...]:
        query_terms = Counter(tokenize(query))
        count = len(self.term_frequencies)
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = (
                    self.document_lengths[index] / self.average_document_length
                    if self.average_document_length
                    else 0.0
                )
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)
                score += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )
            scores.append(score)
        return tuple(scores)
