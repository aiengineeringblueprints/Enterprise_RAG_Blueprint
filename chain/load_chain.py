import langchain_core
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from prompts.promt_manager import PromptManager, PromptKey


def load_chain(context, prompt_key: PromptKey, model, include_doc_names: bool = True, chat_history: list = None):    
    """
    Loads a Retrieval-Augmented Generation (RAG) chain that can work with either a fixed text context 
    or a retriever for dynamic document retrieval.
    
    Args:
        context (Union[str, Callable]): The context for the RAG chain. This can either be a fixed text 
            (as a string) or a retriever (a callable that retrieves documents).
        prompt_key (PromptKey): The prompt template
        model (Callable): The model to be used in the RAG chain for generating responses.
        include_doc_names (bool, optional): Whether to include document names when formatting retrieved 
            documents. Defaults to True.
        chat_history (list, optional): List of previous messages for conversational context.
            
    Returns:
        Callable: A configured RAG chain that processes input questions and generates responses.
    """

    raw_template = PromptManager().load_template(prompt_key)

    # Check if we should use chat history
    use_chat_history = chat_history is not None and len(chat_history) > 0

    if use_chat_history:
        # Use ChatPromptTemplate with MessagesPlaceholder for conversational context
        # Remove triple quotes if present and clean up
        cleaned_template = raw_template.strip('"""').strip("'''").strip()

        # Build a system message that includes the context variable
        # The question will come from the human message
        system_template = """You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If the answer is not clear from the context, use your own knowledge to answer the question.
        Limit your response to three sentences. If used, say explicitly which sources you used to answer the question. Name the document and the page number.

        Take into account the conversation history when answering.

        <context>
        {context}
        </context>"""

        rag_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    else:
        # Traditional template without chat history
        rag_prompt = ChatPromptTemplate.from_template(raw_template)

    # context is either a fixed text or a retriever
    if isinstance(context, str):
        prepared_context = lambda _: context # For fixed text: text in callable

    elif isinstance(context, langchain_core.vectorstores.base.VectorStoreRetriever):
        if context.invoke is None:  # empty retriever
            prepared_context = lambda _: "<unknown>"
        else:  # normal retriever, but defensive against Chroma errors
            def prepared_context(question: str):
                try:
                    docs = context.invoke(question)
                except Exception as e:
                    logging.warning("Retriever invoke failed in chain context, using empty context: %s", e)
                    docs = []
                return format_docs(docs, include_doc_names)
    else:
        raise ValueError(f"Unknown context type: {type(context)}. Expected str or VectorStoreRetriever.")

    # Build the chain with or without chat history
    if use_chat_history:
        from langchain_core.messages import HumanMessage, AIMessage

        # Convert chat_history to LangChain message format
        formatted_history = []
        for msg in chat_history:
            if msg.get("role") == "user":
                formatted_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                formatted_history.append(AIMessage(content=msg.get("content", "")))

        # For chains with chat history, we need to pass the context and question
        # The question is the input to the chain, and we add context and history
        def build_inputs(question_input):
            """Build the input dictionary for the prompt with context and history."""
            if isinstance(question_input, str):
                question = question_input
            else:
                question = str(question_input)

            return {
                "context": prepared_context(question),
                "question": question,
                "chat_history": formatted_history,
            }

        chain = build_inputs | rag_prompt | model | StrOutputParser()
    else:
        chain = (
            {"context": prepared_context, "question": RunnablePassthrough()}
            | rag_prompt
            | model
            | StrOutputParser()
        )
    return chain


def rag_chain(question, chain, retriever, show_sources=True) -> tuple:
    """
    Runs a Retrieval-Augmented Generation (RAG) chain with a given question.

    Args:
        question (str): The input question to query the RAG chain.
        chain (object): The RAG chain object responsible for generating responses.
        retriever (object): The retriever object used to fetch relevant documents.
        show_sources (bool, optional): If True, retrieves and includes source documents. Defaults to True.
        user_roles (list, optional): List of roles/categories the user has for role-based filtering.

    Returns:
        tuple: A tuple containing:
            - The response generated by the RAG chain.
            - A list of source documents (if `show_sources` is True), where each entry is a string
              combining the document's metadata source and its content. Returns None if `show_sources` is False.
    """


    if retriever is None:
        raise ValueError("No retriever found. Please check your configuration of available docs.")

    sources = []
    if show_sources:
        try:
            retrieved_docs = retriever.invoke(question)
        except Exception as e:
            logging.warning("Retriever invoke failed, returning no sources: %s", e)
            retrieved_docs = []
        for doc in retrieved_docs:
            # Be defensive: some docs may miss 'source' or have None content
            src = str(getattr(doc, "metadata", {}) .get("source", "unknown"))
            content = getattr(doc, "page_content", None) or ""
            sources.append(f"{src}->{content}")
    else:
        sources = None

    return chain.invoke(question), sources



def format_docs(docs: list[Document], include_doc_names: bool = False) -> str:
    """Returns formatted documents with optional names from metadata."""
    texts = []
    for doc in docs:
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        # Assumption: The document name is stored in the metadata dictionary under 'name' or 'source'
        if include_doc_names:
            doc_name = doc.metadata.get("source", "Unknown")
            texts.append(f"Document: {doc_name}\n{lines}")
        else:
            texts.append("\n".join(lines))
    return "\n\n".join(texts)
