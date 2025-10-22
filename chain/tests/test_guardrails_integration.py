"""
Integration tests for guardrails in chain API.
Tests the complete flow from API endpoint through guardrails to LLM.
"""
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_api import app
from prompts.promt_manager import PromptKey

client = TestClient(app)


class TestGuardrailsIntegrationInAPI:
    """Integration tests for guardrails in the /call_llm endpoint."""

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_allowed_by_guardrails(self, mock_call_llm, mock_ensure):
        """Should process request when guardrails allow input."""
        # Guardrails allow the input
        mock_ensure.return_value = (True, None)
        
        # Mock LLM response
        mock_call_llm.return_value = (
            "Python is a programming language",
            ["doc1.pdf", "doc2.pdf"],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "What is Python?",
                "prompt_key": "rag_source",
                "used_model": "test-model",
                "show_sources": True,
                "user_roles": ["General"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"] == "Python is a programming language"
        
        # Verify guardrails were called
        mock_ensure.assert_called_once()
        assert mock_ensure.call_args[0][0] == "What is Python?"

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_blocked_by_guardrails(self, mock_call_llm, mock_ensure):
        """Should return 400 when guardrails block input."""
        # Guardrails block the input
        mock_ensure.return_value = (False, "Ihre Eingabe wurde blockiert")
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Harmful content",
                "prompt_key": "rag_source"
            }
        )
        
        assert response.status_code == 400
        assert "blockiert" in response.json()["detail"]
        
        # Verify call_llm was NOT called
        mock_call_llm.assert_not_called()

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_with_chat_history_allowed(self, mock_call_llm, mock_ensure):
        """Should allow request with chat history when guardrails pass."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = (
            "Answer with context",
            ["doc1.pdf"],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Follow-up question",
                "prompt_key": "rag_source",
                "chat_history": [
                    {"role": "human", "content": "First question"},
                    {"role": "ai", "content": "First answer"}
                ]
            }
        )
        
        assert response.status_code == 200
        mock_ensure.assert_called_once_with("Follow-up question")

    @patch("chain_api.ensure_input_allowed")
    def test_call_llm_guardrail_check_only_on_question(self, mock_ensure):
        """Should only check the current question, not chat history."""
        mock_ensure.return_value = (True, None)
        
        with patch("chain_api.call_llm") as mock_call:
            mock_call.return_value = ("Answer", [], PromptKey.SOURCE)
            
            client.post(
                "/call_llm",
                json={
                    "question": "Current question",
                    "chat_history": [
                        {"role": "human", "content": "Past question 1"},
                        {"role": "ai", "content": "Past answer 1"},
                        {"role": "human", "content": "Past question 2"},
                        {"role": "ai", "content": "Past answer 2"}
                    ]
                }
            )
            
            # Should only check the current question
            mock_ensure.assert_called_once_with("Current question")

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_with_custom_params(self, mock_call_llm, mock_ensure):
        """Should apply guardrails even with custom parameters."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Answer", [], PromptKey.SUMMARY)
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Summarize this",
                "prompt_key": "rag_summary",
                "used_model": "gpt-4",
                "show_sources": False,
                "user_roles": ["IT & Technology"]
            }
        )
        
        assert response.status_code == 200
        mock_ensure.assert_called_once_with("Summarize this")

    @patch("chain_api.ensure_input_allowed")
    def test_call_llm_guardrail_runtime_error(self, mock_ensure):
        """Should return 500 when guardrail check raises GuardrailRuntimeError."""
        from guardrails_manager import GuardrailRuntimeError
        mock_ensure.side_effect = GuardrailRuntimeError("Check failed")
        
        response = client.post(
            "/call_llm",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 500
        assert "Guardrail check failed" in response.json()["detail"]

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_with_minimum_question(self, mock_call_llm, mock_ensure):
        """Should apply guardrails to minimal requests."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        response = client.post(
            "/call_llm",
            json={"question": "Q"}  # Minimum length = 1
        )
        
        assert response.status_code == 200
        mock_ensure.assert_called_once_with("Q")

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_with_special_characters(self, mock_call_llm, mock_ensure):
        """Should handle special characters in questions."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        question = "What is C++? @#$% & special chars: <>"
        response = client.post(
            "/call_llm",
            json={"question": question}
        )
        
        assert response.status_code == 200
        mock_ensure.assert_called_once_with(question)

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_with_unicode(self, mock_call_llm, mock_ensure):
        """Should handle Unicode characters properly."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Antwort", [], PromptKey.SOURCE)
        
        question = "Was ist KI? 🤖 Äöü"
        response = client.post(
            "/call_llm",
            json={"question": question}
        )
        
        assert response.status_code == 200
        mock_ensure.assert_called_once_with(question)


class TestGuardrailsDisabledViaEnvironment:
    """Tests for disabling guardrails via environment variable."""

    @patch.dict(os.environ, {"DISABLE_GUARDRAILS": "true"})
    @patch("chain_api.call_llm")
    def test_call_llm_with_guardrails_disabled(self, mock_call_llm):
        """Should skip guardrail checks when disabled via env var."""
        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        # Should not check guardrails at all
        with patch("chain_api.ensure_input_allowed") as mock_ensure:
            mock_ensure.return_value = (True, None)
            
            response = client.post(
                "/call_llm",
                json={"question": "Any question"}
            )
            
            assert response.status_code == 200
            # Guardrails will still be called, but internally disabled
            mock_ensure.assert_called_once()


class TestGuardrailsFailOpen:
    """Tests for fail-open behavior when guardrails have errors."""

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_when_guardrails_fail_open(self, mock_call_llm, mock_ensure):
        """Should allow request when guardrails fail-open (return True, None)."""
        # Simulate fail-open: guardrails allow despite internal error
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        response = client.post(
            "/call_llm",
            json={"question": "Question during failure"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data


class TestGuardrailsLogging:
    """Tests for guardrails logging behavior."""

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    @patch("chain_api.logging")
    def test_call_llm_logs_guardrail_runtime_error(self, mock_logging, mock_call_llm, mock_ensure):
        """Should log GuardrailRuntimeError exceptions."""
        from guardrails_manager import GuardrailRuntimeError
        mock_ensure.side_effect = GuardrailRuntimeError("Test error")
        
        response = client.post(
            "/call_llm",
            json={"question": "Test"}
        )
        
        assert response.status_code == 500
        mock_logging.exception.assert_called_once()
        log_call_args = mock_logging.exception.call_args[0]
        assert "guardrail failure" in log_call_args[0]


class TestGuardrailsPerformance:
    """Tests for guardrails performance characteristics."""

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_async_behavior(self, mock_call_llm, mock_ensure):
        """Should handle async guardrail checks properly."""
        mock_ensure.return_value = (True, None)
        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        response = client.post(
            "/call_llm",
            json={"question": "Test question"}
        )
        
        assert response.status_code == 200

    @patch("chain_api.ensure_input_allowed")
    @patch("chain_api.call_llm")
    def test_call_llm_guardrails_called_before_llm(self, mock_call_llm, mock_ensure):
        """Should call guardrails before calling LLM."""
        call_order = []
        
        def ensure_side_effect(*args):
            call_order.append("guardrails")
            return (True, None)
        
        def llm_side_effect(*args, **kwargs):
            call_order.append("llm")
            return ("Answer", [], PromptKey.SOURCE)
        
        mock_ensure.side_effect = ensure_side_effect
        mock_call_llm.side_effect = llm_side_effect
        
        response = client.post(
            "/call_llm",
            json={"question": "Test"}
        )
        
        assert response.status_code == 200
        assert call_order == ["guardrails", "llm"]
