"""Compatibility behavior for pinned Transformers encoder outputs."""

from dataclasses import dataclass

from cutagent.retrieval.encoders import pooled_feature_tensor


@dataclass
class _Output:
    pooler_output: object


def test_pooled_feature_tensor_supports_transformers_5_output() -> None:
    feature = object()
    assert pooled_feature_tensor(_Output(feature)) is feature
    assert pooled_feature_tensor(feature) is feature
