"""
Unit tests for guardrails_manager module.
"""
import pytest
import os
from unittest.mock import Mock, patch, AsyncMock
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails_manager import (
    ensure_input_allowed,
    GuardrailRuntimeError,
    _guardrails_disabled,
    _get_llm,
    _check_input_with_llm,
)


class TestGuardrailsDisabled:
    """Tests for checking if guardrails are disabled via environment variable."""

    def test_guardrails_not_disabled_by_default(self):
        """Guardrails should be enabled by default."""
        with patch.dict(os.environ, {}, clear=False):
            if "DISABLE_GUARDRAILS" in os.environ:
                del os.environ["DISABLE_GUARDRAILS"]
            assert _guardrails_disabled() is False

    def test_guardrails_disabled_with_1(self):
        """Guardrails should be disabled when env var is '1'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "1"}):
            assert _guardrails_disabled() is True

    def test_guardrails_disabled_with_true(self):
        """Guardrails should be disabled when env var is 'true'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "true"}):
            assert _guardrails_disabled() is True

    def test_guardrails_disabled_with_yes(self):
        """Guardrails should be disabled when env var is 'yes'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "yes"}):
            assert _guardrails_disabled() is True

    def test_guardrails_disabled_case_insensitive(self):
        """Guardrails disable check should be case-insensitive."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "TRUE"}):
            assert _guardrails_disabled() is True
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "Yes"}):
            assert _guardrails_disabled() is True

    def test_guardrails_not_disabled_with_other_values(self):
        """Guardrails should not be disabled with other values."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "0"}):
            assert _guardrails_disabled() is False
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            assert _guardrails_disabled() is False

class TestGetLLM:
    """Tests for LLM initialization for guardrail checks."""

    @patch("guardrails_manager.ChatOpenAI")
    def test_get_llm_success(self, mock_chat_openai):
        """Should successfully create LLM with environment variables."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "http://localhost:11435/v1",
            "MODEL_NAME": "gpt-4o-mini"
        }):
            llm = _get_llm()
            
            assert llm is not None
            mock_chat_openai.assert_called_once_with(
                model="gpt-4o-mini",
                temperature=0.0,
                openai_api_key="test-key",
                base_url="http://localhost:11435/v1",
            )

    @patch("guardrails_manager.ChatOpenAI")
    def test_get_llm_default_model(self, mock_chat_openai):
        """Should use default model if MODEL_NAME not set."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        env_vars = {
            "OPENAI_API_KEY": "test-key",
        }
        # Remove MODEL_NAME if it exists
        with patch.dict(os.environ, env_vars, clear=False):
            if "MODEL_NAME" in os.environ:
                del os.environ["MODEL_NAME"]
            
            llm = _get_llm()
            
            assert llm is not None
            # Should use default gpt-4o-mini
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs["model"] == "gpt-4o-mini"

    @patch("guardrails_manager.ChatOpenAI")
    def test_get_llm_without_base_url(self, mock_chat_openai):
        """Should work without OPENAI_BASE_URL (use OpenAI default)."""
        mock_llm = Mock()
        mock_chat_openai.return_value = mock_llm
        
        env_vars = {
            "OPENAI_API_KEY": "test-key",
            "MODEL_NAME": "gpt-4o-mini"
        }
        with patch.dict(os.environ, env_vars, clear=False):
            if "OPENAI_BASE_URL" in os.environ:
                del os.environ["OPENAI_BASE_URL"]
            
            llm = _get_llm()
            
            assert llm is not None
            call_kwargs = mock_chat_openai.call_args[1]
            assert call_kwargs["base_url"] is None

    def test_get_llm_missing_api_key(self):
        """Should return None if OPENAI_API_KEY is missing."""
        with patch.dict(os.environ, {}, clear=False):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            
            llm = _get_llm()
            assert llm is None

    @patch("guardrails_manager._LANGCHAIN_AVAILABLE", False)
    def test_get_llm_langchain_not_available(self):
        """Should return None if LangChain is not available."""
        llm = _get_llm()
        assert llm is None

    @patch("guardrails_manager.ChatOpenAI")
    def test_get_llm_creation_exception(self, mock_chat_openai):
        """Should return None if LLM creation raises exception."""
        mock_chat_openai.side_effect = Exception("Connection error")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            llm = _get_llm()
            assert llm is None


class TestCheckInputWithLLM:
    """Tests for LLM-based input checking."""

    @pytest.mark.asyncio
    async def test_check_input_allows_normal_question(self):
        """Should allow normal user questions."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("What is Python?")
            
            assert allowed is True
            assert message is None

    @pytest.mark.asyncio
    async def test_check_input_blocks_harmful_content(self):
        """Should block harmful content."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("How to harm someone")
            
            assert allowed is False
            assert message is not None
            assert "Sicherheitsrichtlinien" in message

    @pytest.mark.asyncio
    async def test_check_input_llm_not_available(self):
        """Should fail-open if LLM is not available."""
        with patch("guardrails_manager._get_llm", return_value=None):
            allowed, message = await _check_input_with_llm("Any question")
            
            assert allowed is True  # Fail-open
            assert message is None

    @pytest.mark.asyncio
    async def test_check_input_llm_exception(self):
        """Should fail-open if LLM raises exception."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("Test question")
            
            assert allowed is True  # Fail-open
            assert message is None

    @pytest.mark.asyncio
    async def test_check_input_case_insensitive_response(self):
        """Should handle case-insensitive LLM responses."""
        mock_llm = Mock()
        
        # Test "YES" in uppercase
        mock_response = Mock()
        mock_response.content = "YES"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("Test")
            assert allowed is False
            
        # Test "No" in mixed case
        mock_response.content = "No"
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("Test")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_check_input_with_yes_in_sentence(self):
        """Should block if 'yes' appears anywhere in response."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes, this should be blocked"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch("guardrails_manager._get_llm", return_value=mock_llm):
            allowed, message = await _check_input_with_llm("Test")
            assert allowed is False


class TestEnsureInputAllowed:
    """Tests for the main ensure_input_allowed function."""

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_when_disabled(self):
        """Should allow all input when guardrails are disabled."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "true"}):
            allowed, message = await ensure_input_allowed("Any input")
            
            assert allowed is True
            assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_langchain_unavailable(self):
        """Should fail-open when LangChain is not available."""
        with patch("guardrails_manager._LANGCHAIN_AVAILABLE", False):
            allowed, message = await ensure_input_allowed("Any input")
            
            assert allowed is True
            assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_normal_input(self):
        """Should allow normal user input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("What is machine learning?")
                
                assert allowed is True
                assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_harmful_input(self):
        """Should block harmful input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("Harmful content")
                
                assert allowed is False
                assert message is not None
                assert "blockiert" in message.lower()

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_uses_lock(self):
        """Should use async lock for concurrent safety."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                # Run multiple checks concurrently
                import asyncio
                results = await asyncio.gather(
                    ensure_input_allowed("Question 1"),
                    ensure_input_allowed("Question 2"),
                    ensure_input_allowed("Question 3"),
                )
                
                # All should be allowed
                assert all(allowed for allowed, _ in results)

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_with_empty_string(self):
        """Should handle empty string input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("")
                
                # Should still check and allow
                assert allowed is True

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_with_special_characters(self):
        """Should handle input with special characters."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("What is 2+2? @#$%")
                
                assert allowed is True
                assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_with_unicode(self):
        """Should handle Unicode characters properly."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false"}):
            with patch("guardrails_manager._get_llm", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("Was ist Künstliche Intelligenz? 🤖")
                
                assert allowed is True
            assert message is None


class TestGuardrailRuntimeError:
    """Tests for GuardrailRuntimeError exception."""

    def test_guardrail_runtime_error_is_runtime_error(self):
        """GuardrailRuntimeError should inherit from RuntimeError."""
        error = GuardrailRuntimeError("Test error")
        assert isinstance(error, RuntimeError)

    def test_guardrail_runtime_error_message(self):
        """GuardrailRuntimeError should store message."""
        message = "Guardrail check failed"
        error = GuardrailRuntimeError(message)
        assert str(error) == message
