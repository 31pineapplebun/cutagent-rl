"""Public-query-only analysis and deterministic M2B routing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from cutagent.schemas.retrieval import (
    AdaptiveRetrievalConfig,
    QueryAnalysis,
    QueryIntent,
    RetrievalChannel,
    RetrievalPlan,
    RetrievalQuery,
)

RoutingStrategy = Literal["uniform", "static_weighted", "query_aware"]

_INTENT_ORDER: tuple[QueryIntent, ...] = (
    "visible_text",
    "speech_or_quote",
    "action_or_motion",
    "static_visual_entity",
    "semantic_scene",
    "ambiguous",
)
_VISIBLE_TERMS = (
    "visible text",
    "exact text",
    "on screen",
    "screen reads",
    "showing the text",
    "ocr",
    "屏幕文字",
    "画面文字",
    "显示文字",
    "写着",
    "字幕为",
)
_SPEECH_TERMS = (
    "narration",
    "narrator",
    "says",
    "said",
    "spoken",
    "speech",
    "hear",
    "voice",
    "旁白",
    "说出",
    "说了",
    "语音",
    "听到",
)
_ACTION_GROUPS: Mapping[str, tuple[str, ...]] = {
    "move_left": ("move left", "moving left", "leftward", "向左", "左移"),
    "move_right": ("move right", "moving right", "rightward", "向右", "右移"),
    "move_up": ("move up", "moving up", "upward", "向上", "上移"),
    "move_down": ("move down", "moving down", "downward", "向下", "下移"),
    "stationary": ("stationary", "not moving", "stands still", "静止", "不动"),
    "appear": ("appear", "appears", "becomes visible", "出现"),
    "disappear": ("disappear", "disappears", "vanish", "消失"),
    "enter_frame": ("enter frame", "enters the frame", "进入画面", "入画"),
    "exit_frame": ("exit frame", "leaves the frame", "离开画面", "出画"),
    "approach": ("approach", "closer", "toward camera", "靠近", "接近镜头"),
    "move_away": ("move away", "farther", "recede", "远离"),
    "pick_up": ("pick up", "picks up", "lift", "拿起", "捡起"),
    "drop": ("drop", "drops", "put down", "放下", "掉落"),
    "start": ("start moving", "starts moving", "开始移动"),
    "stop": ("stop moving", "stops moving", "停止移动"),
    "before": (" before ", "先于", "之前"),
    "after": (" after ", "晚于", "之后"),
}
_ENTITY_WORDS = {
    "ball",
    "circle",
    "square",
    "box",
    "object",
    "person",
    "logo",
    "phone",
    "product",
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "cyan",
    "magenta",
    "球",
    "圆形",
    "方块",
    "物体",
    "人物",
    "人",
    "标志",
    "手机",
    "产品",
    "红色",
    "蓝色",
    "绿色",
    "黄色",
    "紫色",
    "橙色",
}
_STATIC_TERMS = (
    "containing",
    "contains",
    "showing",
    "find a",
    "object",
    "entity",
    "包含",
    "有一个",
    "物体",
)
_SEMANTIC_TERMS = ("scene where", "a scene", "moment when", "场景", "画面", "时刻")
_QUOTED = re.compile(
    r'["\u201c\u201d\u2018\u2019]([^"\u201c\u201d\u2018\u2019]+)'
    r'["\u201c\u201d\u2018\u2019]'
)
_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


class QueryAnalyzer:
    """Conservative lexical analyzer that cannot represent evaluator labels."""

    version = "m2b-query-analyzer-v1"

    def analyze(self, query: RetrievalQuery) -> QueryAnalysis:
        normalized = " ".join(query.text.casefold().split())
        quoted = tuple(dict.fromkeys(match.strip() for match in _QUOTED.findall(query.text)))
        actions = tuple(
            code for code, variants in _ACTION_GROUPS.items() if _contains_any(normalized, variants)
        )
        latin_tokens = set(_LATIN_TOKEN.findall(normalized))
        entities = tuple(
            sorted(word for word in _ENTITY_WORDS if word in normalized or word in latin_tokens)
        )

        visible = 1.0 if _contains_any(normalized, _VISIBLE_TERMS) else 0.0
        speech = 1.0 if _contains_any(normalized, _SPEECH_TERMS) else 0.0
        action = min(1.0, 0.85 + 0.10 * (len(actions) - 1)) if actions else 0.0
        static = 0.6 if entities else 0.0
        if _contains_any(normalized, _STATIC_TERMS):
            static = max(static, 0.75)
        semantic = 0.80 if _contains_any(normalized, _SEMANTIC_TERMS) else 0.2
        if quoted and not visible and not speech:
            speech = 0.55
        if action:
            semantic = max(semantic, 0.45)
        ambiguous = 0.5 if max(visible, speech, action, static) < 0.55 else 0.05
        scores: dict[QueryIntent, float] = {
            "speech_or_quote": speech,
            "visible_text": visible,
            "static_visual_entity": static,
            "semantic_scene": semantic,
            "action_or_motion": action,
            "ambiguous": ambiguous,
        }
        primary = max(
            _INTENT_ORDER, key=lambda intent: (scores[intent], -_INTENT_ORDER.index(intent))
        )
        reasons = [f"primary_intent={primary}"]
        if quoted:
            reasons.append("quoted_phrase_detected")
        if actions:
            reasons.append("action_terms=" + ",".join(actions))
        if entities:
            reasons.append("entity_terms=" + ",".join(entities))
        return QueryAnalysis(
            analyzer_version=self.version,
            primary_intent=primary,
            intent_scores=scores,
            quoted_phrases=quoted,
            action_terms=actions,
            entity_terms=entities,
            reasons=tuple(reasons),
        )


_QUERY_AWARE_WEIGHTS: Mapping[QueryIntent, dict[RetrievalChannel, float]] = {
    "visible_text": {
        "bm25_ocr": 4.0,
        "bm25_combined": 0.5,
        "dense_text": 1.0,
        "visual": 0.5,
    },
    "speech_or_quote": {
        "bm25_transcript": 3.0,
        "bm25_combined": 0.5,
        "dense_text": 1.5,
        "visual": 0.25,
    },
    "static_visual_entity": {
        "bm25_combined": 0.5,
        "dense_text": 1.5,
        "visual": 2.5,
    },
    "semantic_scene": {
        "bm25_combined": 0.75,
        "dense_text": 2.0,
        "visual": 1.25,
    },
    "action_or_motion": {
        "bm25_structured": 1.0,
        "dense_text": 2.0,
        "visual": 1.25,
    },
    "ambiguous": {"bm25_combined": 1.0, "dense_text": 1.0, "visual": 1.0},
}


class RetrievalRouter:
    """Create a plan from public analysis; evaluator labels are not accepted."""

    def __init__(self, config: AdaptiveRetrievalConfig) -> None:
        self.config = config

    def plan(
        self,
        query: RetrievalQuery,
        analysis: QueryAnalysis,
        *,
        strategy: RoutingStrategy = "query_aware",
        evidence_reranking: bool = True,
        native_video: bool = True,
    ) -> RetrievalPlan:
        if analysis.analyzer_version != self.config.analyzer_version:
            raise ValueError("query analysis version does not match retrieval configuration")
        if strategy == "uniform":
            weights: dict[RetrievalChannel, float] = {
                "bm25_combined": 1.0,
                "dense_text": 1.0,
                "visual": 1.0,
            }
        elif strategy == "static_weighted":
            weights = {"bm25_combined": 1.0, "dense_text": 1.25, "visual": 1.25}
        else:
            weights = dict(_QUERY_AWARE_WEIGHTS[analysis.primary_intent])
        disabled = set(self.config.disabled_channels)
        weights = {
            channel: weight for channel, weight in weights.items() if channel not in disabled
        }
        if not weights:
            raise ValueError("routing configuration disabled every retrieval channel")
        motion_sensitive = analysis.primary_intent == "action_or_motion"
        native_enabled = self.config.native_video_enabled and native_video and motion_sensitive
        return RetrievalPlan(
            query_id=query.query_id,
            analyzer_version=self.config.analyzer_version,
            routing_policy_version=self.config.routing_policy_version,
            primary_intent=analysis.primary_intent,
            enabled_channels=tuple(weights),
            channel_weights=weights,
            candidate_depth=self.config.candidate_depth,
            rerank_depth=min(self.config.rerank_depth, self.config.candidate_depth),
            reranking_policy="evidence_rules" if evidence_reranking else "none",
            native_video_policy=("ambiguous_motion_only" if native_enabled else "disabled"),
            native_video_depth=self.config.native_video_depth if native_enabled else 0,
            reasons=(f"strategy={strategy}", f"intent={analysis.primary_intent}"),
        )
