
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from handle_llms import _parse_float, _normalize_roles

class TestParseFloat:

    def test_parse_float_from_float(self):

        assert _parse_float(0.7) == 0.7
        
    def test_parse_float_from_int(self):

        assert _parse_float(1) == 1.0
        
    def test_parse_float_from_string(self):

        assert _parse_float("0.5") == 0.5
        
    def test_parse_float_from_string_with_comma(self):

        assert _parse_float("0,5") == 0.5
        assert _parse_float("1,23") == 1.0
        assert _parse_float("1,23", max_v=2.0) == 1.23
        
    def test_parse_float_with_spaces(self):

        assert _parse_float("  0.5  ") == 0.5
        assert _parse_float("  0,7  ") == 0.7
        
    def test_parse_float_clamping_below_min(self):

        result = _parse_float(-0.5, min_v=0.0, max_v=1.0)
        assert result == 0.0
        
    def test_parse_float_clamping_above_max(self):

        result = _parse_float(1.5, min_v=0.0, max_v=1.0)
        assert result == 1.0
        
    def test_parse_float_within_range(self):

        result = _parse_float(0.5, min_v=0.0, max_v=1.0)
        assert result == 0.5
        
    def test_parse_float_invalid_string(self):

        result = _parse_float("invalid", default=0.5)
        assert result == 0.5
        
    def test_parse_float_custom_default(self):

        result = _parse_float("invalid", default=0.7)
        assert result == 0.7
        
    def test_parse_float_custom_range(self):

        result = _parse_float(50, min_v=0.0, max_v=100.0)
        assert result == 50.0
        
        result = _parse_float(-10, min_v=0.0, max_v=100.0)
        assert result == 0.0
        
        result = _parse_float(150, min_v=0.0, max_v=100.0)
        assert result == 100.0

class TestNormalizeRoles:

    def test_normalize_roles_valid_list(self):

        result = _normalize_roles(["IT & Technology", "General"])
        assert result == ["IT & Technology", "General"]
        
    def test_normalize_roles_empty_list(self):

        result = _normalize_roles([])
        assert result == ["General"]
        
    def test_normalize_roles_none(self):

        result = _normalize_roles(None)
        assert result == ["General"]
        
    def test_normalize_roles_single_string(self):

        result = _normalize_roles("IT & Technology")
        assert result == ["IT & Technology"]
        
    def test_normalize_roles_bytes(self):

        result = _normalize_roles(b"IT & Technology")
        assert result == ["IT & Technology"]
        
    def test_normalize_roles_list_with_falsy_values(self):

        result = _normalize_roles(["IT & Technology", "", None, "General", False])
        assert result == ["IT & Technology", "General"]
        
    def test_normalize_roles_all_falsy(self):

        result = _normalize_roles(["", None, False])
        assert result == ["General"]
        
    def test_normalize_roles_exception_handling(self):

        result = _normalize_roles(123)
        assert result == ["General"]

class TestCallLLM:

    @patch.dict(os.environ, {"DISABLE_QUERY_REFINEMENT": "true"})
    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.load_llm_model')
    @patch('handle_llms.rag_chain')
    def test_call_llm_basic(self, mock_rag_chain, mock_load_llm, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        from prompts.promt_manager import PromptKey
        
        mock_retriever = Mock()
        mock_docs = [
            Mock(page_content="Doc 1", metadata={"source": "doc1.pdf"}),
            Mock(page_content="Doc 2", metadata={"source": "doc2.pdf"})
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Test answer"
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Test answer", mock_docs)
        
        result, relevant_docs, prompt_key = call_llm(
            question="What is Python?",
            prompt_key=PromptKey.SOURCE,
            user_roles=["General"]
        )
        
        assert result == "Test answer"
        assert len(relevant_docs) > 0
        mock_create_retriever.assert_called_once()
        mock_load_llm.assert_called_once()
        mock_load_chain_func.assert_called_once_with(
            mock_retriever,
            PromptKey.SOURCE,
            mock_llm,
            include_doc_names=True,
            chat_history=None
        )
    
    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.load_llm_model')
    @patch('handle_llms.rag_chain')
    def test_call_llm_with_admin_role(self, mock_rag_chain, mock_load_llm, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        from prompts.promt_manager import PromptKey
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Admin answer"
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Admin answer", [])
        
        result, relevant_docs, prompt_key = call_llm(
            question="What is Python?",
            user_roles=["admin", "IT & Technology"]
        )
        
        assert result == "Admin answer"
        mock_create_retriever.assert_called_once()
        
        call_args = mock_create_retriever.call_args
        assert "admin" in call_args[1]['user_roles']
        

    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.load_llm_model')
    @patch('handle_llms.rag_chain')
    def test_call_llm_with_multiple_roles(self, mock_rag_chain, mock_load_llm, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer"
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Answer", [])
        
        roles = ["IT & Technology", "Marketing & Sales", "Finance & Controlling"]
        result, relevant_docs, prompt_key = call_llm(
            question="What is the budget?",
            user_roles=roles
        )
        
        call_args = mock_create_retriever.call_args
        assert call_args[1]['user_roles'] == roles
        

    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.load_llm_model')
    @patch('handle_llms.rag_chain')
    def test_call_llm_show_sources_false(self, mock_rag_chain, mock_load_llm, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Answer without sources"
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Answer without sources", [])
        
        result, relevant_docs, prompt_key = call_llm(
            question="What is Python?",
            show_sources=False
        )
        
        assert relevant_docs == []
        

    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.load_llm_model')
    @patch('handle_llms.rag_chain')
    def test_call_llm_custom_model(self, mock_rag_chain, mock_load_llm, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Custom model answer"
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Custom model answer", [])
        
        custom_model = "gpt-4"
        result, relevant_docs, prompt_key = call_llm(
            question="What is Python?",
            used_model=custom_model
        )
        
        call_args = mock_load_llm.call_args
        assert call_args[0][0] == custom_model
        

    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    def test_call_llm_retriever_exception(self, mock_load_chain, mock_create_retriever):

        from handle_llms import call_llm
        
        mock_create_retriever.side_effect = Exception("Retriever error")
        
        with pytest.raises(Exception) as exc_info:
            call_llm(question="What is Python?")
        
        assert "Retriever error" in str(exc_info.value)

class TestLoadLLMModel:

    @patch('handle_llms.ChatOpenAI')
    def test_load_llm_model_openai(self, mock_chat_openai):

        from handle_llms import load_llm_model
        
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = load_llm_model(source="openai")
        
        mock_chat_openai.assert_called_once()
        assert result == mock_llm
        call_args = mock_chat_openai.call_args
        assert 'model_name' in call_args[1]
        assert call_args[1]['temperature'] == 0
        
    @patch('handle_llms.ChatOpenAI')
    def test_load_llm_model_with_temperature_fixed(self, mock_chat_openai):

        from handle_llms import load_llm_model
        
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        result = load_llm_model(source="openai")
        
        call_args = mock_chat_openai.call_args
        assert call_args[1]['temperature'] == 0
        
    def test_load_llm_model_invalid_source(self):

        from handle_llms import load_llm_model
        
        with pytest.raises(ValueError, match="Unsupported LLM source"):
            load_llm_model(source="invalid-source")

class TestIntegration:

    @patch.dict(os.environ, {"DISABLE_QUERY_REFINEMENT": "true"})
    @patch('handle_llms.load_chain')
    @patch('handle_llms.create_retriever')
    @patch('handle_llms.ChatOpenAI')
    @patch('handle_llms.rag_chain')
    def test_full_llm_pipeline(self, mock_rag_chain, mock_chat_openai, mock_create_retriever, mock_load_chain_func):

        from handle_llms import call_llm
        from prompts.promt_manager import PromptKey
        
        mock_retriever = Mock()
        mock_docs = [
            Mock(
                page_content="Python is a programming language.",
                metadata={"source": "python.pdf", "page": 1}
            )
        ]
        mock_retriever.invoke.return_value = mock_docs
        mock_create_retriever.return_value = mock_retriever
        
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        mock_chain = Mock()
        mock_chain.invoke.return_value = "Python is a high-level programming language."
        mock_load_chain_func.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Python is a high-level programming language.", mock_docs)
        
        question = "What is Python?"
        result, relevant_docs, prompt_key = call_llm(
            question=question,
            prompt_key=PromptKey.SOURCE,
            user_roles=["IT & Technology"]
        )
        
        assert "Python" in result
        assert len(relevant_docs) > 0
        assert relevant_docs[0].page_content == "Python is a programming language."
        assert prompt_key == PromptKey.SOURCE
        
        mock_create_retriever.assert_called_once()
        mock_chat_openai.assert_called_once()
        
        # Verify load_chain was called with chat_history parameter
        call_args = mock_load_chain_func.call_args
        assert call_args[1]['chat_history'] is None
        assert call_args[1]['include_doc_names'] is True
        
        mock_rag_chain.assert_called_once()
