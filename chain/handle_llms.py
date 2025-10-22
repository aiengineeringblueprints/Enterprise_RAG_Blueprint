import os
from prompts.promt_manager import PromptKey
from langchain_openai import ChatOpenAI
from load_chain import load_chain, rag_chain
from retriever import create_retriever


LLM_SOURCE_ANSWER = os.getenv("LLM_SOURCE_ANSWER")
LLM_SOURCE_CHECK = os.getenv("LLM_SOURCE_CHECK")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen15_72b_chat")  # Default model name if not set
RETRIEVER_SIMILARITY_THRESHOLD = os.getenv("RETRIEVER_SIMILARITY_THRESHOLD", "0.5")


def _parse_float(value, default=0.5, min_v=0.0, max_v=1.0) -> float:
    """Parse floats robustly (accepts strings like '0,5'), with clamping."""
    try:
        if isinstance(value, (int, float)):
            v = float(value)
        else:
            v = float(str(value).replace(",", ".").strip())
    except Exception:
        v = float(default)
    # clamp
    if v < min_v:
        v = min_v
    if v > max_v:
        v = max_v
    return v


def _normalize_roles(roles) -> list:
    """Ensure a non-empty list of roles; default to ['General'] when empty/invalid."""
    if not roles:
        return ["General"]
    if isinstance(roles, (str, bytes)):
        # single string provided
        return [roles.decode() if isinstance(roles, bytes) else roles]
    try:
        # filter out falsy values
        norm = [r for r in roles if r]
        return norm or ["General"]
    except Exception:
        return ["General"]

def call_llm(
    question: str,
    prompt_key: PromptKey = PromptKey.SUMMARY,
    used_model: str = LLM_SOURCE_ANSWER,
    show_sources: bool = True,
    user_roles: list = ["General"],
    chat_history: list = None,
) -> str:
    """
    Calls a language model (LLM) to process a given question using a specified prompt template and model.

    Args:
        question (str): The input question to be processed by the LLM.
        promt_key (PromptKey, optional): The name of the prompt template to use. Defaults to PromptKey.SUMMARY.
        used_model (str, optional): The identifier of the LLM model to use. Defaults to LLM_SOURCE_ANSWER.
        show_sources (bool, optional): Whether to include source information in the result. Defaults to True.
        user_roles (list, optional): List of roles/categories the user has for role-based filtering.
        chat_history (list, optional): List of previous messages [{role, content}] for conversational context.

    Returns:
        object: The result of the LLM processing, which may include the answer and optionally the sources.

    Raises:
        ValueError: If any of the following components are not found or fail to load:
            - Retriever
            - Prompt template
            - LLM model
            - Chain
            - Result
    """

    # Normalize inputs and load the retriever with role-based filtering
    roles = _normalize_roles(user_roles)
    threshold = _parse_float(RETRIEVER_SIMILARITY_THRESHOLD, default=0.5, min_v=0.0, max_v=1.0)
    retriever = create_retriever(
        returned_docs=5,
        similarity_threshold=threshold,
        user_roles=roles,
    )
    if retriever is None:
        raise ValueError("No retriever found. Please check your configuration.")

    model = load_llm_model(used_model)
    if model is None:
        raise ValueError("No LLM model found. Please check your configuration.")

    # Limit chat history to last 10 messages to avoid token overflow
    limited_chat_history = None
    if chat_history and len(chat_history) > 0:
        limited_chat_history = chat_history[-10:]

    chain = load_chain(
        retriever,
        prompt_key,
        model,
        include_doc_names=True,
        chat_history=limited_chat_history,
    )
    if chain is None:
        raise ValueError("No chain found. Please check your configuration.")

    result, relevant_documents = rag_chain(
        question, 
        chain, 
        retriever, 
        show_sources=True
    )
    if result is None:
        raise ValueError("No result found. Please check your configuration.")

    return result, relevant_documents, prompt_key



def load_llm_model(source: str) -> object:
    """
    Loads a language model (LLM) based on the specified source.
    Args:
        source (str): The source of the LLM to load. Currently only supports "openai".
    Returns:
        object: An instance of the ChatOpenAI model.
    Notes:
        - The temperature parameter controls the balance between fact-correctness and creativity.
          A lower value (e.g., 0.1) favors fact-correctness.
        - The model uses the MODEL_NAME environment variable with a default of "qwen15_72b_chat".
    """
    llm = None
    if source == "openai":
        llm = ChatOpenAI(
            # model_name="gpt-3.5-turbo",
            model_name=MODEL_NAME, # 32k token context length
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM source: {source}. Please use 'openai'.")
    return llm
