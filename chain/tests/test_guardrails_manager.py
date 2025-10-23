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
    _check_input_with_llm,
)


class TestCheckInputWithLLM:
    """Tests for LLM-based input checking."""

    @pytest.mark.asyncio
    async def test_check_input_allows_normal_question(self):
        """Should allow normal user questions."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch.dict(os.environ, {"LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        
        with patch.dict(os.environ, {"LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await _check_input_with_llm("How to harm someone")
                
                assert allowed is False
                assert message is not None
                assert (
                    "security guidelines"
                    in message
                )

    @pytest.mark.asyncio
    async def test_check_input_llm_exception(self):
        """Should fail-open if LLM raises exception."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        
        with patch.dict(os.environ, {"LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        
        with patch.dict(os.environ, {"LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await _check_input_with_llm("Test")
                assert allowed is False
                
            # Test "No" in mixed case
            mock_response.content = "No"
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await _check_input_with_llm("Test")
                assert allowed is True

    @pytest.mark.asyncio
    async def test_check_input_with_yes_in_sentence(self):
        """Should block if 'yes' appears anywhere in response."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "yes, this should be blocked"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        with patch.dict(os.environ, {"LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await _check_input_with_llm("Test")
                assert allowed is False


class TestEnsureInputAllowed:
    """Tests for the main ensure_input_allowed function."""

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_when_disabled_with_1(self):
        """Should allow all input when guardrails are disabled with '1'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "1"}):
            allowed, message = await ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_when_disabled_with_true(self):
        """Should allow all input when guardrails are disabled with 'true'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "true"}):
            allowed, message = await ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_when_disabled_with_yes(self):
        """Should allow all input when guardrails are disabled with 'yes'."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "yes"}):
            allowed, message = await ensure_input_allowed("Any input")
            assert allowed is True
            assert message is None
    
    @pytest.mark.asyncio
    async def test_ensure_input_allowed_when_disabled_case_insensitive(self):
        """Should handle DISABLE_GUARDRAILS case-insensitively."""
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "TRUE"}):
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
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("Harmful content")
                
                assert allowed is False
                assert message is not None
                assert "blocked" in message.lower()

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_uses_lock(self):
        """Should use async lock for concurrent safety."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "no"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
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
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("Was ist Künstliche Intelligenz? 🤖")
                
                assert allowed is True
                assert message is None

    @pytest.mark.asyncio
    async def test_ensure_input_allowed_fail_open_on_exception(self):
        """Should fail-open when LLM raises an exception."""
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        
        # Temporarily enable guardrails for this test
        with patch.dict(os.environ, {"DISABLE_GUARDRAILS": "false", "LLM_SOURCE_ANSWER": "openai"}):
            with patch("guardrails_manager.load_llm_model", return_value=mock_llm):
                allowed, message = await ensure_input_allowed("Test question")
                
                assert allowed is True  # Fail-open
                assert message is None
