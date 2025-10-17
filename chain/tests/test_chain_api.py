
from unittest.mock import patch
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_api import app, QuestionRequest, QuestionCheckRequest, KeywordCheckRequest
from prompts.promt_manager import PromptKey

client = TestClient(app)

class TestQuestionRequest:

    def test_question_request_default_values(self):

        request = QuestionRequest(question="What is Python?")
        
        assert request.question == "What is Python?"
        assert request.prompt_key == PromptKey.SOURCE
        assert request.show_sources is True
        assert request.user_roles == ["General"]
        
    def test_question_request_custom_values(self):

        request = QuestionRequest(
            question="What is AI?",
            prompt_key=PromptKey.SUMMARY,
            used_model="gpt-4",
            show_sources=False,
            user_roles=["IT & Technology", "Research & Development"]
        )
        
        assert request.question == "What is AI?"
        assert request.prompt_key == PromptKey.SUMMARY
        assert request.used_model == "gpt-4"
        assert request.show_sources is False
        assert len(request.user_roles) == 2
        
    def test_question_request_string_prompt_key(self):

        request = QuestionRequest(
            question="Test?",
            prompt_key="rag_source"
        )
        
        assert request.prompt_key == "rag_source"

class TestQuestionCheckRequest:

    def test_question_check_request_default_values(self):

        request = QuestionCheckRequest(
            question="What is Python?",
            answer="Python is a programming language",
            relevant_documents=["doc1.pdf", "doc2.pdf"]
        )
        
        assert request.question == "What is Python?"
        assert request.answer == "Python is a programming language"
        assert len(request.relevant_documents) == 2
        assert request.prompt_key == PromptKey.CHECK
        assert request.question_prompt_key == PromptKey.SOURCE
        
    def test_question_check_request_custom_values(self):

        request = QuestionCheckRequest(
            question="What is AI?",
            answer="AI is artificial intelligence",
            relevant_documents=["doc1.pdf"],
            prompt_key=PromptKey.CHECK,
            used_model="gpt-4",
            question_prompt_key=PromptKey.SUMMARY
        )
        
        assert request.used_model == "gpt-4"
        assert request.question_prompt_key == PromptKey.SUMMARY

class TestKeywordCheckRequest:

    def test_keyword_check_request_default_values(self):

        request = KeywordCheckRequest(
            question="What is Python?",
            answer="Python is a programming language"
        )
        
        assert request.question == "What is Python?"
        assert request.answer == "Python is a programming language"
        assert request.expected_keywords == []
        assert request.threshold == 0.6
        
    def test_keyword_check_request_custom_values(self):

        request = KeywordCheckRequest(
            question="What is Python?",
            answer="Python is a programming language",
            expected_keywords=["python", "programming", "language"],
            threshold=0.8
        )
        
        assert len(request.expected_keywords) == 3
        assert request.threshold == 0.8

class TestCallLLMEndpoint:

    @patch('chain_api.call_llm')
    def test_call_llm_endpoint_success(self, mock_call_llm):

        mock_call_llm.return_value = (
            "Python is a high-level programming language.",
            [{"source": "python.pdf", "content": "Python info"}],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "What is Python?",
                "prompt_key": "rag_source",
                "show_sources": True,
                "user_roles": ["General"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "relevant_documents" in data
        assert "prompt_key" in data
        assert "Python" in data["result"]
        
        mock_call_llm.assert_called_once()
        
    @patch('chain_api.call_llm')
    def test_call_llm_endpoint_with_enum_prompt_key(self, mock_call_llm):

        mock_call_llm.return_value = (
            "Summary text",
            [],
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
    def test_call_llm_endpoint_multiple_roles(self, mock_call_llm):

        mock_call_llm.return_value = (
            "Answer",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "What is the budget?",
                "user_roles": ["Finance & Controlling", "IT & Technology"]
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_call_llm.call_args
        assert "Finance & Controlling" in call_args[1]['user_roles']
        assert "IT & Technology" in call_args[1]['user_roles']
        
    @patch('chain_api.call_llm')
    def test_call_llm_endpoint_show_sources_false(self, mock_call_llm):

        mock_call_llm.return_value = (
            "Answer without sources",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Test question",
                "show_sources": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["relevant_documents"], list)
        
    @patch('chain_api.call_llm')
    def test_call_llm_endpoint_custom_model(self, mock_call_llm):

        mock_call_llm.return_value = (
            "Custom model answer",
            [],
            PromptKey.SOURCE
        )
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Test question",
                "used_model": "gpt-4"
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_call_llm.call_args
        assert call_args[1]['used_model'] == "gpt-4"
        
    @patch('chain_api.call_llm')
    def test_call_llm_endpoint_exception_handling(self, mock_call_llm):

        mock_call_llm.side_effect = Exception("LLM service unavailable")
        
        response = client.post(
            "/call_llm",
            json={
                "question": "Test question"
            }
        )
        
        assert response.status_code == 500
        assert "LLM service unavailable" in response.json()["detail"]
        
    def test_call_llm_endpoint_invalid_request(self):

        response = client.post(
            "/call_llm",
            json={
                "show_sources": True
            }
        )
        
        assert response.status_code == 422

class TestCheckAnswerEndpoint:

    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_endpoint_success(self, mock_check_answer):

        mock_check_answer.return_value = "The answer is correct and comprehensive."
        
        response = client.post(
            "/check_answer",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language",
                "relevant_documents": ["doc1.pdf", "doc2.pdf"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert "correct" in data["evaluation"].lower()
        
        mock_check_answer.assert_called_once()
        
    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_endpoint_with_custom_prompts(self, mock_check_answer):

        mock_check_answer.return_value = "Good answer"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Test?",
                "answer": "Test answer",
                "relevant_documents": [],
                "prompt_key": "rag_check",
                "question_prompt_key": "rag_summary"
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_check_answer.call_args
        assert call_args[1]['prompt'] == PromptKey.CHECK
        assert call_args[1]['question_prompt_key'] == PromptKey.SUMMARY
        
    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_endpoint_custom_model(self, mock_check_answer):

        mock_check_answer.return_value = "Evaluation result"
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Test?",
                "answer": "Test answer",
                "relevant_documents": [],
                "used_model": "gpt-4"
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_check_answer.call_args
        assert call_args[1]['used_model'] == "gpt-4"
        
    @patch('chain_api.check_answer_with_second_llm')
    def test_check_answer_endpoint_exception_handling(self, mock_check_answer):

        mock_check_answer.side_effect = Exception("Check service failed")
        
        response = client.post(
            "/check_answer",
            json={
                "question": "Test?",
                "answer": "Test answer",
                "relevant_documents": []
            }
        )
        
        assert response.status_code == 500
        assert "Check service failed" in response.json()["detail"]
        
    def test_check_answer_endpoint_invalid_request(self):

        response = client.post(
            "/check_answer",
            json={
                "question": "Test?",
            }
        )
        
        assert response.status_code == 422

class TestCheckKeywordsEndpoint:

    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_endpoint_success(self, mock_check_keywords):

        mock_check_keywords.return_value = {
            "evaluation": "yes",
            "present_keywords": ["python", "programming"],
            "missing_keywords": [],
            "total_keywords": 2,
            "ratio": 1.0
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "What is Python?",
                "answer": "Python is a programming language",
                "expected_keywords": ["python", "programming"],
                "threshold": 0.8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "evaluation" in data
        assert data["evaluation"] == "yes"
        assert data["ratio"] == 1.0
        
        mock_check_keywords.assert_called_once()
        
    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_endpoint_partial_match(self, mock_check_keywords):

        mock_check_keywords.return_value = {
            "evaluation": "partial",
            "present_keywords": ["python"],
            "missing_keywords": ["java", "javascript"],
            "total_keywords": 3,
            "ratio": 0.33
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "What are programming languages?",
                "answer": "Python is a language",
                "expected_keywords": ["python", "java", "javascript"],
                "threshold": 0.5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["evaluation"] == "partial"
        assert len(data["missing_keywords"]) == 2
        
    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_endpoint_default_values(self, mock_check_keywords):

        mock_check_keywords.return_value = {
            "evaluation": "undetermined",
            "present_keywords": [],
            "missing_keywords": [],
            "total_keywords": 0,
            "ratio": 1.0
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Test?",
                "answer": "Test answer"
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_check_keywords.call_args
        assert call_args[1]['expected_keywords'] == []
        assert call_args[1]['threshold'] == 0.6
        
    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_endpoint_custom_threshold(self, mock_check_keywords):

        mock_check_keywords.return_value = {
            "evaluation": "yes",
            "present_keywords": ["test"],
            "missing_keywords": [],
            "total_keywords": 1,
            "ratio": 1.0
        }
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Test?",
                "answer": "Test answer",
                "expected_keywords": ["test"],
                "threshold": 0.9
            }
        )
        
        assert response.status_code == 200
        
        call_args = mock_check_keywords.call_args
        assert call_args[1]['threshold'] == 0.9
        
    @patch('chain_api.check_answer_with_keywords')
    def test_check_keywords_endpoint_exception_handling(self, mock_check_keywords):

        mock_check_keywords.side_effect = Exception("Keyword check failed")
        
        response = client.post(
            "/check_keywords",
            json={
                "question": "Test?",
                "answer": "Test answer"
            }
        )
        
        assert response.status_code == 500
        assert "Keyword check failed" in response.json()["detail"]
        
    def test_check_keywords_endpoint_invalid_request(self):

        response = client.post(
            "/check_keywords",
            json={
                "question": "Test?"
            }
        )
        
        assert response.status_code == 422

class TestIntegration:

    @patch('chain_api.call_llm')
    @patch('chain_api.check_answer_with_keywords')
    def test_full_pipeline_llm_and_keyword_check(self, mock_check_keywords, mock_call_llm):

        mock_call_llm.return_value = (
            "Python is a programming language",
            [{"source": "doc.pdf"}],
            PromptKey.SOURCE
        )
        mock_check_keywords.return_value = {
            "evaluation": "yes",
            "present_keywords": ["python", "programming"],
            "missing_keywords": [],
            "total_keywords": 2,
            "ratio": 1.0
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
                "expected_keywords": ["python", "programming"]
            }
        )
        assert keyword_response.status_code == 200
        assert keyword_response.json()["evaluation"] == "yes"
        
    def test_api_app_exists(self):

        assert app is not None
        assert hasattr(app, 'post')
        
    def test_api_routes_exist(self):

        routes = [route.path for route in app.routes]
        
        assert "/call_llm" in routes
        assert "/check_answer" in routes
        assert "/check_keywords" in routes
        
    @patch('chain_api.call_llm')
    def test_concurrent_requests(self, mock_call_llm):

        mock_call_llm.return_value = ("Answer", [], PromptKey.SOURCE)
        
        responses = []
        for i in range(5):
            response = client.post(
                "/call_llm",
                json={"question": f"Question {i}"}
            )
            responses.append(response)
        
        assert all(r.status_code == 200 for r in responses)
