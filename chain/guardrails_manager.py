"""Helper utilities for validating user inputs with simple LLM-based guardrails."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

_logger = logging.getLogger(__name__)
_GUARDRAIL_LOCK = asyncio.Lock()
_BLOCKED_MESSAGE = "Ihre Eingabe wurde durch die Sicherheitsrichtlinien blockiert. Bitte formulieren Sie Ihre Anfrage anders."


class GuardrailRuntimeError(RuntimeError):
    """Raised if the guardrail check could not be executed."""


def _guardrails_disabled() -> bool:
    """Return True when guardrails are explicitly disabled via env var."""
    return os.getenv("DISABLE_GUARDRAILS", "").strip().lower() in {"1", "true", "yes"}


def _get_llm() -> Optional["ChatOpenAI"]:
    """Create a ChatOpenAI instance for guardrail checks."""
    if not _LANGCHAIN_AVAILABLE:
        _logger.error("LangChain OpenAI is not installed")
        return None
    
    try:
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        if not openai_api_key:
            _logger.error("OPENAI_API_KEY not set")
            return None
            
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
            base_url=openai_base_url if openai_base_url else None,
        )
        return llm
    except Exception as e:
        _logger.error(f"Failed to create LLM: {e}")
        return None


async def _check_input_with_llm(user_input: str) -> Tuple[bool, Optional[str]]:
    """Check user input using direct LLM call with a clear, permissive prompt."""
    llm = _get_llm()
    if llm is None:
        _logger.warning("LLM not available, allowing input (fail-open)")
        return (True, None)
    
    prompt = f"""You are a content moderation system. Determine if a user message should be BLOCKED.

ONLY block messages that:
- contain references to illegal activities
- promote harm or violence
- contain references to self-harm 
- hate speech or discrimination


Allow everything else, including:
- Normal questions and conversations
- Test messages and simple inputs
- Technical questions
- Requests for information

User message: "{user_input}"

Question: Should this message be BLOCKED?
Answer with ONLY one word: "yes" (block) or "no" (allow).
Answer:"""

    try:
        _logger.info(f"Checking input: '{user_input[:50]}...'")
        response = await llm.ainvoke(prompt)
        answer = response.content.strip().lower()
        
        _logger.info(f"LLM response: '{answer}'")
        
        if "yes" in answer:
            _logger.warning(f"Input blocked: '{user_input}'")
            return (False, _BLOCKED_MESSAGE)
        else:
            _logger.info(f"Input allowed: '{user_input[:50]}...'")
            return (True, None)
            
    except Exception as e:
        _logger.error(f"Guardrail check failed: {e}, allowing input (fail-open)")
        return (True, None)


async def ensure_input_allowed(user_input: str) -> Tuple[bool, Optional[str]]:
    """Validate user input via guardrails. Returns (allowed, error_message)."""
    if _guardrails_disabled():
        _logger.info("Guardrails disabled by environment variable")
        return (True, None)
    
    if not _LANGCHAIN_AVAILABLE:
        _logger.warning("LangChain not available, guardrails disabled (fail-open)")
        return (True, None)
    
    async with _GUARDRAIL_LOCK:
        return await _check_input_with_llm(user_input)
