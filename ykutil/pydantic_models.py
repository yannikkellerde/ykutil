from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


# ── Low‑level helpers ──────────────────────────────────────────────────────────
class FilterCategory(BaseModel):
    filtered: bool
    severity: Optional[str] = None  # e.g. "safe", "low", "medium", …
    detected: Optional[bool] = None  # only present on jailbreak

    model_config = ConfigDict(extra="allow")  # tolerate future keys


class ContentFilterResults(BaseModel):
    hate: Optional[FilterCategory] = None
    self_harm: Optional[FilterCategory] = None
    sexual: Optional[FilterCategory] = None
    violence: Optional[FilterCategory] = None
    jailbreak: Optional[FilterCategory] = None

    model_config = ConfigDict(extra="allow")


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    name: Optional[str] = None  # for function messages, etc.

    model_config = ConfigDict(extra="allow")


# ── Mid‑level wrappers ─────────────────────────────────────────────────────────
class Choice(BaseModel):
    index: int
    message: Optional[Message] = None  # absent in streaming deltas
    delta: Optional[Message] = None  # present in streaming deltas
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    content_filter_results: Optional[ContentFilterResults] = None

    model_config = ConfigDict(extra="allow")


class PromptFilterResult(BaseModel):
    prompt_index: int
    content_filter_results: ContentFilterResults

    model_config = ConfigDict(extra="allow")


class Usage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    model_config = ConfigDict(extra="allow")


# ── Top‑level “body” for chat completions ──────────────────────────────────────
class Body(BaseModel):
    id: str
    created: int  # Unix‑epoch seconds
    model: str
    object: str
    choices: List[Choice]
    prompt_filter_results: Optional[List[PromptFilterResult]] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[Usage] = None

    model_config = ConfigDict(extra="allow")


# ── Envelope that most SDKs wrap around the body ──────────────────────────────
class OpenAIResponse(BaseModel):
    body: Body
    request_id: Optional[str] = None
    status_code: int

    model_config = ConfigDict(extra="allow")


class BatchResponse(BaseModel):
    custom_id: str
    response: OpenAIResponse
    error: Optional[str] = None
