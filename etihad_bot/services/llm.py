"""Thin OpenAI wrapper that logs user/conversation, tokens, latency, cost and status."""
import contextvars
import json
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from etihad_bot.config import get_settings
from etihad_bot.services import store

request_ctx: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_ctx", default={"user_id": "anonymous", "conversation_id": "none"})


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key or None)
    return _client


def complete(system: str, user: str, *, purpose: str, max_tokens: int = 1024,
             temperature: float = 0.0, json_mode: bool = False) -> LLMResult:
    s = get_settings()
    ctx = request_ctx.get()
    start = time.perf_counter()
    status, error, in_tok, out_tok, text = "ok", None, 0, 0, ""
    try:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = _get_client().chat.completions.create(
            model=s.llm_model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}], **extra)
        text = resp.choices[0].message.content or ""
        in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
    except Exception as e:  # logged, then re-raised
        status, error = "error", f"{type(e).__name__}: {e}"
        raise
    finally:
        cost = (in_tok * s.llm_price_in_per_m + out_tok * s.llm_price_out_per_m) / 1e6
        store.log_llm_call(
            user_id=ctx["user_id"], conversation_id=ctx["conversation_id"], purpose=purpose,
            model=s.llm_model, input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=(time.perf_counter() - start) * 1000, cost_usd=cost,
            status=status, error=error)
    return LLMResult(text, in_tok, out_tok)


def complete_json(system: str, user: str, *, purpose: str, max_tokens: int = 512) -> dict:
    """Ask for a JSON object; tolerant of code fences / surrounding prose."""
    r = complete(system + "\nRespond with a single JSON object and nothing else.", user,
                 purpose=purpose, max_tokens=max_tokens, json_mode=True)
    m = re.search(r"\{.*\}", r.text, re.DOTALL)
    if not m:
        return {}
    try:
        # strict=False: quotes copied from the PDF can contain raw line breaks inside strings
        return json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return {}
