
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_api import app
from prompts.promt_manager import PromptKey

client = TestClient(app)


class TestCallLLMEndpoint:

    @patch('chain_api.call_llm')
    def test_call_llm_basic_request(self, mock_call_llm):
        mock_call_llm.return_value = (
            "This is the answer to your question.",
            ["doc1.pdf", "doc2.txt"],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "What is RAG?",
                "prompt_key": "rag_source",
                "user_roles": ["General"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "result" in data
        assert "relevant_documents" in data
        assert "prompt_key" in data
        assert data["result"] == "This is the answer to your question."
        assert len(data["relevant_documents"]) == 2
        assert data["prompt_key"] == "rag_source"
        
        mock_call_llm.assert_called_once()

    @patch('chain_api.call_llm')
    def test_call_llm_with_enum_prompt_key(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Summary of the document",
            ["source.pdf"],
            PromptKey.SUMMARY
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Summarize this",
                "prompt_key": "rag_summary"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["prompt_key"] == "rag_summary"

    @patch('chain_api.call_llm')
    def test_call_llm_with_custom_model(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Custom model response",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Test question",
                "used_model": "custom-llm-model",
                "show_sources": False
            }
        )
        
        assert response.status_code == 200
        call_args = mock_call_llm.call_args
        assert call_args[1]["used_model"] == "custom-llm-model"
        assert call_args[1]["show_sources"] is False

    @patch('chain_api.call_llm')
    def test_call_llm_with_multiple_user_roles(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Role-based response",
            ["restricted_doc.pdf"],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Sensitive question",
                "user_roles": ["General", "IT_Technology", "admin"]
            }
        )
        
        assert response.status_code == 200
        call_args = mock_call_llm.call_args
        assert call_args[1]["user_roles"] == ["General", "IT_Technology", "admin"]

    @patch('chain_api.call_llm')
    def test_call_llm_exception_handling(self, mock_call_llm):
        mock_call_llm.side_effect = Exception("LLM service unavailable")
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Will this fail?"
            }
        )
        
        assert response.status_code == 500
        assert "LLM service unavailable" in response.json()["detail"]

    @patch('chain_api.call_llm')
    def test_call_llm_no_sources(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Answer without sources",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Simple question",
                "show_sources": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["relevant_documents"] == []

    @patch('chain_api.call_llm')
    def test_call_llm_default_parameters(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Default response",
            ["doc.pdf"],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={"question": "Minimal request"}
        )
        
        assert response.status_code == 200
        call_args = mock_call_llm.call_args
        assert call_args[1]["show_sources"] is True
        assert call_args[1]["user_roles"] == ["General"]


class TestCheckAnswerEndpoint:

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_valid_response(self, mock_check):
        mock_check.return_value = "yes"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "relevant_documents": ["python_guide.pdf"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert data["evaluation"] == "yes"

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_negative_evaluation(self, mock_check):
        mock_check.return_value = "no"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "What is Java?",
                "answer": "Java is a coffee drink.",
                "relevant_documents": ["java_programming.pdf"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["evaluation"] == "no"

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_with_extended_check(self, mock_check):
        mock_check.return_value = "The answer correctly explains the concept with appropriate detail."
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Explain RAG",
                "answer": "RAG is Retrieval Augmented Generation...",
                "relevant_documents": ["rag_paper.pdf"],
                "prompt_key": "rag_check_extended"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "correctly explains" in data["evaluation"]

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_with_custom_model(self, mock_check):
        mock_check.return_value = "yes"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Test?",
                "answer": "Test answer",
                "relevant_documents": [],
                "used_model": "custom-check-model"
            }
        )
        
        assert response.status_code == 200
        call_args = mock_check.call_args
        assert call_args[1]["used_model"] == "custom-check-model"

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_multiple_documents(self, mock_check):
        mock_check.return_value = "yes"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Complex topic?",
                "answer": "Detailed answer...",
                "relevant_documents": ["doc1.pdf", "doc2.txt", "doc3.md"]
            }
        )
        
        assert response.status_code == 200
        call_args = mock_check.call_args
        assert len(call_args[1]["relevant_documents"]) == 3

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_exception_handling(self, mock_check):
        mock_check.side_effect = Exception("Check service failed")
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Test",
                "answer": "Test",
                "relevant_documents": []
            }
        )
        
        assert response.status_code == 500
        assert "Check service failed" in response.json()["detail"]

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_with_different_prompt_keys(self, mock_check):
        mock_check.return_value = "yes"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Question",
                "answer": "Answer",
                "relevant_documents": [],
                "prompt_key": "rag_check",
                "question_prompt_key": "rag_summary"
            }
        )
        
        assert response.status_code == 200
        call_args = mock_check.call_args
        assert call_args[1]["prompt"] == PromptKey.CHECK
        assert call_args[1]["question_prompt_key"] == PromptKey.SUMMARY


class TestCheckKeywordsEndpoint:

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_all_present(self, mock_check):
        mock_check.return_value = {
            "is_valid": True,
            "match_ratio": 1.0,
            "matched_keywords": ["python", "programming", "language"],
            "missing_keywords": []
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language.",
                "expected_keywords": ["python", "programming", "language"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["match_ratio"] == 1.0
        assert len(data["matched_keywords"]) == 3

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_partial_match(self, mock_check):
        mock_check.return_value = {
            "is_valid": True,
            "match_ratio": 0.67,
            "matched_keywords": ["python", "programming"],
            "missing_keywords": ["interpreted"]
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Describe Python",
                "answer": "Python is a programming language",
                "expected_keywords": ["python", "programming", "interpreted"],
                "threshold": 0.6
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["match_ratio"] >= 0.6

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_below_threshold(self, mock_check):
        mock_check.return_value = {
            "is_valid": False,
            "match_ratio": 0.33,
            "matched_keywords": ["python"],
            "missing_keywords": ["programming", "language"]
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "What is Python?",
                "answer": "Python is something",
                "expected_keywords": ["python", "programming", "language"],
                "threshold": 0.8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert data["match_ratio"] < 0.8

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_no_keywords(self, mock_check):
        mock_check.return_value = {
            "is_valid": True,
            "match_ratio": 1.0,
            "matched_keywords": [],
            "missing_keywords": []
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Free form question",
                "answer": "Any answer works",
                "expected_keywords": []
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_custom_threshold(self, mock_check):
        mock_check.return_value = {
            "is_valid": True,
            "match_ratio": 0.75,
            "matched_keywords": ["key1", "key2", "key3"],
            "missing_keywords": ["key4"]
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Question",
                "answer": "Answer with key1, key2, and key3",
                "expected_keywords": ["key1", "key2", "key3", "key4"],
                "threshold": 0.7
            }
        )
        
        assert response.status_code == 200
        call_args = mock_check.call_args
        assert call_args[1]["threshold"] == 0.7

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_exception_handling(self, mock_check):
        mock_check.side_effect = Exception("Keyword check failed")
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Test",
                "answer": "Test",
                "expected_keywords": ["test"]
            }
        )
        
        assert response.status_code == 500
        assert "Keyword check failed" in response.json()["detail"]


class TestAPIIntegration:

    @patch('chain_api.call_llm')
    @patch('chain_api.check_answer_with_second_llm')
    def test_full_qa_workflow(self, mock_check, mock_call_llm):
        mock_call_llm.return_value = (
            "Python is a high-level programming language.",
            ["python_docs.pdf"],
            PromptKey.SOURCE
        )
        mock_check.return_value = "yes"
        
        llm_response = client.post(
            "/call_llm",
            json={
                "question": "What is Python?",
                "user_roles": ["General"]
            }
        )
        
        assert llm_response.status_code == 200
        llm_data = llm_response.json()
        
        check_response = client.post(
            "/check_answer",
            json={
                "question": "What is Python?",
                "answer": llm_data["result"],
                "relevant_documents": llm_data["relevant_documents"]
            }
        )
        
        assert check_response.status_code == 200
        check_data = check_response.json()
        assert check_data["evaluation"] == "yes"

    @patch('chain_api.call_llm')
    @patch('chain_api.check_answer_with_keywords')
    def test_qa_with_keyword_validation(self, mock_keywords, mock_call_llm):
        mock_call_llm.return_value = (
            "Python is a programming language with dynamic typing.",
            ["python_guide.pdf"],
            PromptKey.SOURCE
        )
        mock_keywords.return_value = {
            "is_valid": True,
            "match_ratio": 1.0,
            "matched_keywords": ["python", "programming", "dynamic"],
            "missing_keywords": []
        }
        
        llm_response = client.post(
            "/call_llm",
            json={"question": "What is Python?"}
        )
        
        assert llm_response.status_code == 200
        answer = llm_response.json()["result"]
        
        keyword_response = client.post(
            "/check_keywords",
            json={
                "question": "What is Python?",
                "answer": answer,
                "expected_keywords": ["python", "programming", "dynamic"]
            }
        )
        
        assert keyword_response.status_code == 200
        assert keyword_response.json()["is_valid"] is True

    def test_openapi_schema_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema
        assert "/call_llm" in schema["paths"]
        assert "/check_answer" in schema["paths"]
        assert "/check_keywords" in schema["paths"]


class TestErrorHandling:

    def test_call_llm_missing_question(self):
        response = client.post(
            "/call_llm",
            json={}
        )
        
        assert response.status_code == 422

    def test_check_answer_missing_required_fields(self):
        response = client.post(
            "/check_answer",
            json={
                "question": "Test"
            }
        )
        
        assert response.status_code == 422

    def test_check_keywords_invalid_threshold(self):
        response = client.post(
            "/check_keywords",
            json={
                "question": "Test",
                "answer": "Test",
                "expected_keywords": [],
                "threshold": -1.5
            }
        )
        
        assert response.status_code == 422

    @patch('chain_api.call_llm')
    def test_call_llm_invalid_prompt_key(self, mock_call_llm):
        response = client.post(
            "/call_llm",
            json={
                "question": "Test",
                "prompt_key": "invalid_key"
            }
        )
        
        assert response.status_code == 422


class TestRequestValidation:

    @patch('chain_api.call_llm')
    def test_call_llm_empty_question(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Empty question handled",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": ""
            }
        )
        
        assert response.status_code == 422
        assert "question" in response.json()["detail"][0]["loc"]

    @patch('chain_api.call_llm')
    def test_call_llm_empty_user_roles(self, mock_call_llm):
        mock_call_llm.return_value = (
            "Answer",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Test",
                "user_roles": []
            }
        )
        
        assert response.status_code == 200

