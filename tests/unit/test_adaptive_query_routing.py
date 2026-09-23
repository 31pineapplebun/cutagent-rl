"""M2B query analysis and routing use public text only."""

from cutagent.retrieval.query import QueryAnalyzer, RetrievalRouter
from cutagent.schemas.retrieval import AdaptiveRetrievalConfig, RetrievalQuery


def test_query_analyzer_detects_general_intents_without_private_labels() -> None:
    analyzer = QueryAnalyzer()
    visible = analyzer.analyze(
        RetrievalQuery(query_id="q-visible", text='Find the visible exact text "SALE 42"')
    )
    motion = analyzer.analyze(
        RetrievalQuery(query_id="q-motion", text="找到红色方块向右移动的场景")
    )
    assert visible.primary_intent == "visible_text"
    assert visible.quoted_phrases == ("SALE 42",)
    assert motion.primary_intent == "action_or_motion"
    assert "move_right" in motion.action_terms
    serialized = visible.model_dump(mode="json")
    assert "query_type" not in serialized
    assert "split" not in serialized
    assert "source_group_id" not in serialized


def test_router_prioritizes_specialized_channels_and_logs_plan() -> None:
    config = AdaptiveRetrievalConfig()
    analyzer = QueryAnalyzer()
    router = RetrievalRouter(config)
    query = RetrievalQuery(query_id="q-speech", text='Find where the narrator says "hello"')
    plan = router.plan(query, analyzer.analyze(query))
    assert plan.primary_intent == "speech_or_quote"
    assert plan.channel_weights["bm25_transcript"] > plan.channel_weights["dense_text"]
    assert plan.native_video_policy == "disabled"
    assert plan.query_id == query.query_id


def test_router_ablation_disables_only_requested_channel() -> None:
    config = AdaptiveRetrievalConfig(disabled_channels=("visual",))
    analyzer = QueryAnalyzer()
    query = RetrievalQuery(query_id="q-entity", text="Find a red square object")
    plan = RetrievalRouter(config).plan(query, analyzer.analyze(query))
    assert "visual" not in plan.enabled_channels
    assert "dense_text" in plan.enabled_channels
