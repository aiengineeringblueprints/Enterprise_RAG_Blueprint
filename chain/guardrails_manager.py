"""Helper utilities for validating user inputs with simple LLM-based guardrails."""
import os
from typing import Optional, Tuple
from prompts.promt_manager import PromptKey, PromptManager


_BLOCKED_MESSAGE = "Your input has been blocked by the security guidelines. Please rephrase your request."


def ensure_input_allowed(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input via guardrails.
    
    Args:
        user_input: The user's input string to validate
        
    Returns:
        Tuple of (allowed, error_message). If allowed is False, error_message contains the reason.
    """
    # Check if guardrails are disabled via environment variable
    if os.getenv("DISABLE_GUARDRAILS", "").strip().lower() in {"1", "true", "yes"}:
        return (True, None)
    
    # Lazy import to avoid circular dependency
    from handle_llms import load_llm_model
    
    llm_source = os.getenv("LLM_SOURCE_ANSWER", "openai")
    llm = load_llm_model(llm_source)
    
    # Load and render the guardrails template
    prompt_manager = PromptManager()
    prompt = prompt_manager.render(PromptKey.INPUT_GUARDRAILS, user_input=user_input)

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip().lower()
        
        if "yes" in answer:
            return (False, _BLOCKED_MESSAGE)
        return (True, None)
            
    except Exception:
        # On error, allow the input to pass through (fail-open)
        return (True, None)
