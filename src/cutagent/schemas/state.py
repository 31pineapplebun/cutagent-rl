"""Public Agent state used by the event reducer."""

from pydantic import Field, field_validator, model_validator

from cutagent.schemas.base import Identifier, SchemaModel
from cutagent.schemas.event import PlanNode, TerminalEvent, ToolObservation, VerificationResult
from cutagent.schemas.task_input import TaskInput


class ExecutionBudget(SchemaModel):
    max_steps: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_search_calls: int = Field(default=3, gt=0)
    max_edit_calls: int = Field(default=8, gt=0)
    max_repeated_identical_actions: int = Field(default=2, gt=0)
    max_structured_output_repairs: int = Field(default=2, ge=0)
    max_model_tokens: int | None = Field(default=None, gt=0)
    max_wall_time_ms: int = Field(gt=0)
    used_steps: int = Field(default=0, ge=0)
    used_tool_calls: int = Field(default=0, ge=0)
    used_search_calls: int = Field(default=0, ge=0)
    used_edit_calls: int = Field(default=0, ge=0)
    used_structured_output_repairs: int = Field(default=0, ge=0)
    used_model_tokens: int = Field(default=0, ge=0)
    used_wall_time_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "ExecutionBudget":
        limits = (
            (self.used_steps, self.max_steps, "steps"),
            (self.used_tool_calls, self.max_tool_calls, "tool calls"),
            (self.used_search_calls, self.max_search_calls, "search calls"),
            (self.used_edit_calls, self.max_edit_calls, "edit calls"),
            (
                self.used_structured_output_repairs,
                self.max_structured_output_repairs,
                "structured output repairs",
            ),
            (self.used_wall_time_ms, self.max_wall_time_ms, "wall time"),
        )
        for used, maximum, label in limits:
            if used > maximum:
                raise ValueError(f"used {label} cannot exceed its maximum")
        if self.max_model_tokens is not None and self.used_model_tokens > self.max_model_tokens:
            raise ValueError("used model tokens cannot exceed their maximum")
        return self

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.used_steps

    @property
    def remaining_tool_calls(self) -> int:
        return self.max_tool_calls - self.used_tool_calls

    @property
    def remaining_search_calls(self) -> int:
        return self.max_search_calls - self.used_search_calls

    @property
    def remaining_edit_calls(self) -> int:
        return self.max_edit_calls - self.used_edit_calls

    @property
    def remaining_structured_output_repairs(self) -> int:
        return self.max_structured_output_repairs - self.used_structured_output_repairs

    @property
    def remaining_model_tokens(self) -> int | None:
        if self.max_model_tokens is None:
            return None
        return self.max_model_tokens - self.used_model_tokens

    @property
    def remaining_wall_time_ms(self) -> int:
        return self.max_wall_time_ms - self.used_wall_time_ms


class AgentState(SchemaModel):
    task_input: TaskInput
    execution_budget: ExecutionBudget
    plan_revision: int = Field(default=0, ge=0)
    plan_steps: tuple[PlanNode, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    verification_results: tuple[VerificationResult, ...] = ()
    terminal_event: TerminalEvent | None = None
    processed_event_ids: tuple[Identifier, ...] = ()
    last_sequence_no: int = Field(default=0, ge=0)
    state_version: int = Field(default=0, ge=0)

    @field_validator("processed_event_ids")
    @classmethod
    def validate_unique_events(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("processed event identifiers must be unique")
        return value

    @model_validator(mode="after")
    def validate_event_counters(self) -> "AgentState":
        if self.last_sequence_no != len(self.processed_event_ids):
            raise ValueError("last_sequence_no must match the number of reduced events")
        if self.state_version != self.last_sequence_no:
            raise ValueError("state_version must increment exactly once per reduced event")
        return self

    @classmethod
    def initial(cls, task_input: TaskInput, budget: ExecutionBudget) -> "AgentState":
        return cls(task_input=task_input, execution_budget=budget)
