
import pytest
from unittest.mock import Mock, patch
import sys
import os
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_chain import format_docs, load_chain, rag_chain
from tests.conftest import MockChainComponent

class TestFormatDocs:

    def test_format_docs_with_doc_names(self):

        docs = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "python_guide.pdf", "page": 1}
            ),
            Document(
                page_content="Java is another programming language",
                metadata={"source": "java_basics.pdf", "page": 5}
            )
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "python_guide.pdf" in result
        assert "java_basics.pdf" in result
        assert "Python is a programming language" in result
        assert "Java is another programming language" in result
        
    def test_format_docs_without_doc_names(self):

        docs = [
            Document(
                page_content="Python is a programming language",
                metadata={"source": "python_guide.pdf", "page": 1}
            ),
            Document(
                page_content="Java is another programming language",
                metadata={"source": "java_basics.pdf", "page": 5}
            )
        ]
        
        result = format_docs(docs, include_doc_names=False)
        
        assert "python_guide.pdf" not in result
        assert "java_basics.pdf" not in result
        
        assert "Python is a programming language" in result
        assert "Java is another programming language" in result
        
    def test_format_docs_empty_list(self):

        result = format_docs([], include_doc_names=True)
        
        assert result == "" or result.strip() == ""
        
    def test_format_docs_single_document(self):

        docs = [
            Document(
                page_content="Single document content",
                metadata={"source": "single.pdf", "page": 1}
            )
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "single.pdf" in result
        assert "Single document content" in result
        
    def test_format_docs_with_page_numbers(self):

        docs = [
            Document(
                page_content="Content from page 1",
                metadata={"source": "doc.pdf", "page": 1}
            ),
            Document(
                page_content="Content from page 5",
                metadata={"source": "doc.pdf", "page": 5}
            )
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "page" in result.lower() or "1" in result
        assert "5" in result or "page 5" in result.lower()
        
    def test_format_docs_missing_metadata(self):

        docs = [
            Document(
                page_content="Content without full metadata",
                metadata={}
            )
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "Content without full metadata" in result
        
    def test_format_docs_with_chunk_index(self):

        docs = [
            Document(
                page_content="First chunk",
                metadata={"source": "doc.pdf", "page": 1, "chunk_index": 0}
            ),
            Document(
                page_content="Second chunk",
                metadata={"source": "doc.pdf", "page": 1, "chunk_index": 1}
            )
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "First chunk" in result
        assert "Second chunk" in result
        
    def test_format_docs_multiple_sources(self):

        docs = [
            Document(page_content="Content 1", metadata={"source": "doc1.pdf"}),
            Document(page_content="Content 2", metadata={"source": "doc2.pdf"}),
            Document(page_content="Content 3", metadata={"source": "doc3.pdf"})
        ]
        
        result = format_docs(docs, include_doc_names=True)
        
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result
        assert "doc3.pdf" in result

class TestLoadChain:

    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_retriever_context(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        mock_retriever = Mock()
        mock_retriever.invoke = Mock(return_value=[
            Document(page_content="Retrieved content", metadata={"source": "doc.pdf"})
        ])
        
        import langchain_core.vectorstores.base
        mock_retriever.__class__ = langchain_core.vectorstores.base.VectorStoreRetriever
        
        chain = load_chain(mock_retriever, PromptKey.SOURCE, mock_model)
        
        assert chain is not None
        
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_invalid_context_type(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        with pytest.raises(ValueError) as exc_info:
            load_chain(12345, PromptKey.SOURCE, mock_model)
        
        assert "Unknown context type" in str(exc_info.value)
        
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_empty_retriever(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        mock_retriever = Mock()
        mock_retriever.invoke = None
        
        import langchain_core.vectorstores.base
        mock_retriever.__class__ = langchain_core.vectorstores.base.VectorStoreRetriever
        
        chain = load_chain(mock_retriever, PromptKey.SOURCE, mock_model)
        
        assert chain is not None
        
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_include_doc_names_true(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        chain = load_chain("Fixed context", PromptKey.SOURCE, mock_model, include_doc_names=True)
        
        assert chain is not None
        
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_include_doc_names_false(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        chain = load_chain("Fixed context", PromptKey.SOURCE, mock_model, include_doc_names=False)
        
        assert chain is not None

class TestRagChain:

    def test_rag_chain_with_sources(self):

        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is the answer"
        
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="Doc 1 content", metadata={"source": "doc1.pdf"}),
            Document(page_content="Doc 2 content", metadata={"source": "doc2.pdf"})
        ]
        mock_retriever.invoke.return_value = mock_docs
        
        question = "What is Python?"
        answer, sources = rag_chain(question, mock_chain, mock_retriever, show_sources=True)
        
        assert answer == "This is the answer"
        assert sources is not None
        assert len(sources) > 0
        mock_chain.invoke.assert_called_once_with(question)
        mock_retriever.invoke.assert_called_once_with(question)
        
    def test_rag_chain_without_sources(self):

        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is the answer"
        
        mock_retriever = Mock()
        
        question = "What is Python?"
        answer, sources = rag_chain(question, mock_chain, mock_retriever, show_sources=False)
        
        assert answer == "This is the answer"
        assert sources is None or sources == []
        mock_chain.invoke.assert_called_once_with(question)
        
    def test_rag_chain_no_retriever(self):

        mock_chain = Mock()
        
        with pytest.raises(ValueError) as exc_info:
            rag_chain("Question?", mock_chain, None, show_sources=True)
        
        assert "No retriever found" in str(exc_info.value)
        
    def test_rag_chain_empty_question(self):

        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer to empty question"
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        answer, sources = rag_chain("", mock_chain, mock_retriever, show_sources=True)
        
        assert answer == "Answer to empty question"
        

        
    def test_rag_chain_chain_exception(self):

        mock_chain = Mock()
        mock_chain.invoke.side_effect = Exception("Chain failed")
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        
        with pytest.raises(Exception) as exc_info:
            rag_chain("Question?", mock_chain, mock_retriever, show_sources=False)
        
        assert "Chain failed" in str(exc_info.value)

class TestIntegration:

    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_full_chain_pipeline(self, mock_prompt_mgr, mock_chat_prompt):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}\nAnswer:"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        context = "Python is a programming language used for web development, data science, and automation."
        chain = load_chain(context, PromptKey.SOURCE, mock_model)
        
        assert chain is not None
        mock_prompt_instance.load_template.assert_called_once_with(PromptKey.SOURCE)
        
    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_retriever_integration(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):

        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        
        mock_format_docs.return_value = "Formatted documents"
        
        mock_retriever = Mock()
        mock_docs = [
            Document(page_content="Python documentation", metadata={"source": "python.pdf"})
        ]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        import langchain_core.vectorstores.base
        mock_retriever.__class__ = langchain_core.vectorstores.base.VectorStoreRetriever
        
        chain = load_chain(mock_retriever, PromptKey.SOURCE, mock_model, include_doc_names=True)
        
        assert chain is not None


class TestLoadChainWithChatHistory:
    """Test load_chain functionality with chat history."""

    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_chat_history(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):
        """Test that load_chain correctly handles chat history."""
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_messages.return_value = mock_template
        
        mock_model = MockChainComponent()
        mock_format_docs.return_value = "Formatted documents"
        
        chat_history = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."}
        ]
        
        chain = load_chain("Test context", PromptKey.SOURCE, mock_model, include_doc_names=True, chat_history=chat_history)
        
        assert chain is not None
        mock_chat_prompt.from_messages.assert_called_once()
        
        # Verify that from_messages was called (indicates chat history mode)
        call_args = mock_chat_prompt.from_messages.call_args[0][0]
        assert len(call_args) == 3  # system, chat_history placeholder, human
        assert call_args[0][0] == "system"
        assert call_args[2][0] == "human"

    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_empty_chat_history(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):
        """Test that empty chat history is treated as no chat history."""
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        mock_format_docs.return_value = "Formatted documents"
        
        chain = load_chain("Test context", PromptKey.SOURCE, mock_model, include_doc_names=True, chat_history=[])
        
        assert chain is not None
        # Empty chat history should use from_template, not from_messages
        mock_chat_prompt.from_template.assert_called_once()
        mock_chat_prompt.from_messages.assert_not_called()

    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_none_chat_history(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):
        """Test that None chat history uses traditional template."""
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_model = MockChainComponent()
        mock_format_docs.return_value = "Formatted documents"
        
        chain = load_chain("Test context", PromptKey.SOURCE, mock_model, include_doc_names=True, chat_history=None)
        
        assert chain is not None
        mock_chat_prompt.from_template.assert_called_once()
        mock_chat_prompt.from_messages.assert_not_called()

    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_converts_chat_history_to_langchain_messages(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):
        """Test that chat history is converted to LangChain message format."""
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_messages.return_value = mock_template
        
        mock_model = MockChainComponent()
        mock_format_docs.return_value = "Formatted documents"
        
        chat_history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]
        
        chain = load_chain("Test context", PromptKey.SOURCE, mock_model, include_doc_names=True, chat_history=chat_history)
        
        assert chain is not None
        # Verify the chain was built with chat history
        mock_chat_prompt.from_messages.assert_called_once()

    @patch('load_chain.format_docs')
    @patch('load_chain.ChatPromptTemplate')
    @patch('load_chain.PromptManager')
    def test_load_chain_with_retriever_and_chat_history(self, mock_prompt_mgr, mock_chat_prompt, mock_format_docs):
        """Test load_chain with retriever and chat history."""
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = MockChainComponent()
        mock_chat_prompt.from_messages.return_value = mock_template
        
        mock_model = MockChainComponent()
        mock_format_docs.return_value = "Formatted documents"
        
        mock_retriever = Mock()
        mock_docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        mock_retriever.invoke = Mock(return_value=mock_docs)
        
        import langchain_core.vectorstores.base
        mock_retriever.__class__ = langchain_core.vectorstores.base.VectorStoreRetriever
        
        chat_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        
        chain = load_chain(mock_retriever, PromptKey.SOURCE, mock_model, include_doc_names=True, chat_history=chat_history)
        
        assert chain is not None
        mock_chat_prompt.from_messages.assert_called_once()
