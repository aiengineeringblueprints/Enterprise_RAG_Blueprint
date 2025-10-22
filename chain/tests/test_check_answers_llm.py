
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_answers_llm import extract_keywords_from_text

class TestExtractKeywordsFromText:

    def test_extract_keywords_basic(self):

        text = "Python programming language machine learning algorithms"
        keywords = extract_keywords_from_text(text)
        
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords
        
    def test_extract_keywords_removes_stopwords(self):

        text = "The quick brown fox jumps over the lazy dog and cat"
        keywords = extract_keywords_from_text(text)
        
        assert "the" not in keywords
        assert "and" not in keywords
        assert "over" not in keywords
        
        assert "quick" in keywords
        assert "brown" in keywords
        assert "jumps" in keywords
        
    def test_extract_keywords_min_length(self):

        text = "AI ML API REST HTTP programming language"
        keywords = extract_keywords_from_text(text, min_length=4)
        
        assert "ai" not in keywords
        assert "ml" not in keywords
        assert "api" not in keywords
        
        assert "rest" in keywords
        assert "http" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        
    def test_extract_keywords_custom_min_length(self):

        text = "AI ML API programming"
        keywords = extract_keywords_from_text(text, min_length=2)
        
        assert "ai" in keywords
        assert "ml" in keywords
        assert "api" in keywords
        assert "programming" in keywords
        
    def test_extract_keywords_removes_punctuation(self):

        text = "Hello, world! This is a test. Python, Java, C++."
        keywords = extract_keywords_from_text(text)
        
        assert "hello" in keywords
        assert "world" in keywords
        assert "test" in keywords
        assert "python" in keywords
        assert "java" in keywords
        
        for keyword in keywords:
            assert "," not in keyword
            assert "." not in keyword
            assert "!" not in keyword
            
    def test_extract_keywords_case_insensitive(self):

        text = "Python JAVA javascript TypeScript"
        keywords = extract_keywords_from_text(text)
        
        assert "python" in keywords
        assert "java" in keywords
        assert "javascript" in keywords
        assert "typescript" in keywords
        
        assert "Python" not in keywords
        assert "JAVA" not in keywords
        
    def test_extract_keywords_unique(self):

        text = "python python python java java javascript"
        keywords = extract_keywords_from_text(text)
        
        assert keywords.count("python") == 1
        assert keywords.count("java") == 1
        assert keywords.count("javascript") == 1
        
    def test_extract_keywords_empty_string(self):

        keywords = extract_keywords_from_text("")
        assert keywords == []
        
    def test_extract_keywords_only_stopwords(self):

        text = "the and or but if"
        keywords = extract_keywords_from_text(text)
        assert len(keywords) == 0
        
    def test_extract_keywords_german_text(self):

        text = "Die Programmiersprache Python ist sehr beliebt für maschinelles Lernen"
        keywords = extract_keywords_from_text(text)
        
        assert "die" not in keywords
        assert "ist" not in keywords
        assert "für" not in keywords
        
        assert "programmiersprache" in keywords
        assert "python" in keywords
        assert "beliebt" in keywords
        assert "maschinelles" in keywords
        assert "lernen" in keywords
        
    def test_extract_keywords_mixed_language(self):

        text = "Python programming language and maschinelles Lernen sind wichtig"
        keywords = extract_keywords_from_text(text)
        
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        assert "maschinelles" in keywords
        assert "lernen" in keywords
        assert "wichtig" in keywords
        
    def test_extract_keywords_special_characters(self):

        text = "RAG-System API-Endpoint C++ C# .NET Framework"
        keywords = extract_keywords_from_text(text)
        
        assert any("rag" in k or "system" in k for k in keywords)
        assert any("api" in k or "endpoint" in k for k in keywords)
        assert any("framework" in k for k in keywords)
        
    def test_extract_keywords_numbers(self):

        text = "Python3 version 3.11 released in 2023"
        keywords = extract_keywords_from_text(text)
        
        assert "python3" in keywords or "python" in keywords
        assert "version" in keywords
        assert "released" in keywords

class TestCheckAnswerWithKeywords:

    @patch('check_answers_llm.extract_keywords_from_text')
    def test_check_answer_with_keywords_all_present(self, mock_extract):

        from check_answers_llm import check_answer_with_keywords
        
        mock_extract.return_value = ["python", "programming", "language"]
        
        question = "What is Python?"
        answer = "Python is a programming language"
        
        result = check_answer_with_keywords(question, answer)
        
        assert isinstance(result, dict)
        assert result["evaluation"] == "yes"
        assert len(result["present_keywords"]) == 3
        assert len(result["missing_keywords"]) == 0
        
    @patch('check_answers_llm.extract_keywords_from_text')
    def test_check_answer_with_keywords_missing(self, mock_extract):

        from check_answers_llm import check_answer_with_keywords
        
        mock_extract.return_value = ["python", "javascript", "typescript"]
        
        question = "What are Python, JavaScript, and TypeScript?"
        answer = "They are programming languages"
        
        result = check_answer_with_keywords(question, answer)
        
        assert isinstance(result, dict)
        assert result["evaluation"] == "no"
        assert len(result["missing_keywords"]) > 0
        
    @patch('check_answers_llm.extract_keywords_from_text')
    def test_check_answer_with_keywords_empty_keywords(self, mock_extract):

        from check_answers_llm import check_answer_with_keywords
        
        mock_extract.return_value = []
        
        question = "What is Python?"
        answer = "Python is a programming language"
        
        result = check_answer_with_keywords(question, answer)
        
        assert isinstance(result, dict)
        assert result["evaluation"] == "undetermined"
        assert result["total_keywords"] == 0
        
    def test_check_answer_with_keywords_with_threshold(self):

        from check_answers_llm import check_answer_with_keywords
        
        question = "What is Python?"
        answer = "Python is a language"
        expected_keywords = ["python", "programming", "language", "code"]
        
        result = check_answer_with_keywords(question, answer, expected_keywords, threshold=0.5)
        
        assert isinstance(result, dict)
        assert result["evaluation"] == "yes"
        assert result["ratio"] == 0.5

class TestCheckAnswerWithSecondLLM:

    @patch('check_answers_llm.rag_chain')
    @patch('check_answers_llm.load_chain')
    @patch('check_answers_llm.load_llm_model')
    @patch('check_answers_llm.ChatPromptTemplate')
    @patch('check_answers_llm.PromptManager')
    def test_check_answer_with_second_llm_basic(
        self, 
        mock_prompt_mgr,
        mock_chat_prompt,
        mock_load_llm,
        mock_load_chain,
        mock_rag_chain
    ):

        from check_answers_llm import check_answer_with_second_llm
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = Mock()
        mock_template.input_variables = ["context", "question"]
        mock_template.format_messages.return_value = "Formatted prompt"
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_load_chain.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Evaluation: The answer is correct", [])
        
        question = "What is Python?"
        answer = "Python is a programming language"
        relevant_docs = [Mock(page_content="Python documentation")]
        
        result = check_answer_with_second_llm(
            question=question,
            question_prompt_key=PromptKey.SOURCE,
            answer=answer,
            relevant_documents=relevant_docs,
            prompt=PromptKey.CHECK
        )
        
        assert isinstance(result, str)
        assert "Evaluation" in result or "correct" in result.lower()
        
        mock_prompt_mgr.assert_called()
        mock_load_llm.assert_called_once()
        mock_load_chain.assert_called_once()
        mock_rag_chain.assert_called_once()
        
    @patch('check_answers_llm.rag_chain')
    @patch('check_answers_llm.load_chain')
    @patch('check_answers_llm.load_llm_model')
    @patch('check_answers_llm.ChatPromptTemplate')
    @patch('check_answers_llm.PromptManager')
    def test_check_answer_with_custom_model(
        self,
        mock_prompt_mgr,
        mock_chat_prompt,
        mock_load_llm,
        mock_load_chain,
        mock_rag_chain
    ):

        from check_answers_llm import check_answer_with_second_llm
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}\nQuestion: {question}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = Mock()
        mock_template.input_variables = ["context", "question"]
        mock_template.format_messages.return_value = "Formatted prompt"
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_load_chain.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Custom model evaluation", [])
        
        custom_model = "gpt-4"
        result = check_answer_with_second_llm(
            question="Test question",
            question_prompt_key=PromptKey.SOURCE,
            answer="Test answer",
            relevant_documents=[],
            used_model=custom_model
        )
        
        mock_load_llm.assert_called_once_with(custom_model)
        
    @patch('check_answers_llm.rag_chain')
    @patch('check_answers_llm.load_chain')
    @patch('check_answers_llm.load_llm_model')
    @patch('check_answers_llm.ChatPromptTemplate')
    @patch('check_answers_llm.PromptManager')
    def test_check_answer_without_question_variable(
        self,
        mock_prompt_mgr,
        mock_chat_prompt,
        mock_load_llm,
        mock_load_chain,
        mock_rag_chain
    ):

        from check_answers_llm import check_answer_with_second_llm
        from prompts.promt_manager import PromptKey
        
        mock_prompt_instance = Mock()
        mock_prompt_instance.load_template.return_value = "Context: {context}"
        mock_prompt_mgr.return_value = mock_prompt_instance
        
        mock_template = Mock()
        mock_template.input_variables = ["context"]
        mock_template.format_messages.return_value = "Formatted prompt with context only"
        mock_chat_prompt.from_template.return_value = mock_template
        
        mock_llm = Mock()
        mock_load_llm.return_value = mock_llm
        
        mock_chain = Mock()
        mock_load_chain.return_value = mock_chain
        
        mock_rag_chain.return_value = ("Evaluation result", [])
        
        result = check_answer_with_second_llm(
            question="What is Python?",
            question_prompt_key=PromptKey.SOURCE,
            answer="Python is a language",
            relevant_documents=[Mock(page_content="Documentation")]
        )
        
        assert isinstance(result, str)
        mock_template.format_messages.assert_called()
        
        call_args = mock_template.format_messages.call_args
        assert "context" in call_args[1] or len(call_args[0]) > 0

class TestIntegration:

    def test_keyword_extraction_integration(self):

        text = """
        Python is a high-level, interpreted programming language with dynamic semantics.
        Its high-level built-in data structures, combined with dynamic typing and dynamic binding,
        make it very attractive for Rapid Application Development, as well as for use as a scripting
        or glue language to connect existing components together.
        """
        
        keywords = extract_keywords_from_text(text, min_length=4)
        
        assert "python" in keywords
        assert "programming" in keywords
        assert "language" in keywords
        assert "development" in keywords
        assert "scripting" in keywords
        
        assert "with" not in keywords
        assert "very" not in keywords
        assert "structures" in keywords
        
        assert all(len(k) >= 4 for k in keywords)
