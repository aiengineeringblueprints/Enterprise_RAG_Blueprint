"""Helper utilities for validating user inputs with simple LLM-based guardrails."""
from __future__ import annotations

import asyncio
import os
from typing import Optional, Tuple
from prompts.promt_manager import PromptKey, PromptManager
from handle_llms import load_llm_model


_GUARDRAIL_LOCK = asyncio.Lock()
_BLOCKED_MESSAGE = "Your input has been blocked by the security guidelines. Please rephrase your request."



async def _check_input_with_llm(user_input: str) -> Tuple[bool, Optional[str]]:
    """Check user input using LLM with prompt template from PromptManager."""
    llm_source = os.getenv("LLM_SOURCE_ANSWER", "openai")
    llm = load_llm_model(llm_source)
    
    # Load and render the guardrails template
    prompt_manager = PromptManager()
    prompt = prompt_manager.render(PromptKey.INPUT_GUARDRAILS, user_input=user_input)

    try:
        response = await llm.ainvoke(prompt)
        answer = response.content.strip().lower()
        
        if "yes" in answer:
            return (False, _BLOCKED_MESSAGE)
        elif "no" in answer:
            return (True, None)
        else:
            return (True, None)
            
    except Exception as e:
        # On error, allow the input to pass through
        return (True, None)


async def ensure_input_allowed(user_input: str) -> Tuple[bool, Optional[str]]:
    """Validate user input via guardrails. Returns (allowed, error_message)."""
    # Check if guardrails are disabled via environment variable
    if os.getenv("DISABLE_GUARDRAILS", "").strip().lower() in {"1", "true", "yes"}:
        return (True, None)
    
    async with _GUARDRAIL_LOCK:
        return await _check_input_with_llm(user_input)
