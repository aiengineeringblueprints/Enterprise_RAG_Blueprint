"""
Unit tests for guardrails_manager module.
"""
import pytest
import os
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guardrails_manager import ensure_input_allowed


class TestEnsureInputAllowed:
    """Tests for the ensure_input_allowed function."""

    def test_ensure_input_allowed_when_disabled_with_1(self):
        """Should allow all input when guardrails are disabled with '1'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "1"}):
            allowed, message = ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None

    def test_ensure_input_allowed_when_disabled_with_true(self):
        """Should allow all input when guardrails are disabled with 'true'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "true"}):
            allowed, message = ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None

    def test_ensure_input_allowed_when_disabled_with_yes(self):
        """Should allow all input when guardrails are disabled with 'yes'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "yes"}):
            allowed, message = ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None
    
    def test_ensure_input_allowed_when_disabled_case_insensitive(self):
        """Should handle DISABLE_GUARDRAILS case-insensitively."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "TRUE"}):
            allowed, message = ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None

    def test_ensure_input_allowed_normal_input(self):
        """Should allow normal user input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("What is machine learning?")
                
                assert allowed is True
                assert message is None

    def test_ensure_input_allowed_harmful_input(self):
        """Should block harmful input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Harmful content")
                
                assert allowed is False
                assert message is not None
                assert "blocked" in message.lower()

    def test_ensure_input_allowed_case_insensitive_response(self):
        """Should handle case-insensitive LLM responses."""
        mock_llm = Mock()
        mock_response = Mock()
        
        # Test "YES" in uppercase
        mock_response.content = "YES"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Test")
                assert allowed is False
                
            # Test "No" in mixed case
            mock_response.content = "No"
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Test")
                assert allowed is True

    def test_ensure_input_allowed_with_yes_in_sentence(self):
        """Should block if 'yes' appears anywhere in response."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes, this should be blocked"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Test")
                assert allowed is False

    def test_ensure_input_allowed_with_empty_string(self):
        """Should handle empty string input."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("")
                
                assert allowed is True

    def test_ensure_input_allowed_with_special_characters(self):
        """Should handle input with special characters."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("What is 2+2? @#$%")
                
                assert allowed is True
                assert message is None

    def test_ensure_input_allowed_with_unicode(self):
        """Should handle Unicode characters properly."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.invoke = Mock(return_value=mock_response)
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Was ist Künstliche Intelligenz? 🤖")
                
                assert allowed is True
                assert message is None

    def test_ensure_input_allowed_fail_open_on_exception(self):
        """Should fail-open when LLM raises an exception."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("API error"))
        
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = ensure_input_allowed("Test question")
                
                assert allowed is True  # Fail-open
                assert message is None
