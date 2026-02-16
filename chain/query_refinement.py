import os
from prompts.promt_manager import PromptKey, PromptManager


def refine_query(user_query: str) -> str:
    """
    Refine user query for better retrieval results.
    
    Args:
        user_query: The original user query string
        
    Returns:
        The refined query string, or original if refinement is disabled/fails
    """
    # Check if query refinement is disabled via environment variable
    if os.getenv("DISABLE_QUERY_REFINEMENT", "").strip().lower() in {"1", "true", "yes"}:
        return user_query
    
    # Skip refinement for very short queries
    if len(user_query.strip()) < 3:
        return user_query
    
    # Lazy import to avoid circular dependency with handle_llms
    from handle_llms import load_llm_model
    
    llm_source = os.getenv("LLM_SOURCE_ANSWER", "openai")
    llm = load_llm_model(llm_source)
    
    # Load and render the query refinement template
    prompt_manager = PromptManager()
    prompt = prompt_manager.render(PromptKey.QUERY_REFINEMENT, user_query=user_query)

    try:
        response = llm.invoke(prompt)
        refined = response.content.strip()
        return refined if refined else user_query
    except Exception:
        # On error, return original query (fail-safe)
        return user_query
