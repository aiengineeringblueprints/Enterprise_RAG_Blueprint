
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import (
    combine_tag_bitmasks,
    is_admin_user,
    TAG_BITMASK_MAPPING
)

class TestCombineTagBitmasks:

    def test_single_tag(self):

        result = combine_tag_bitmasks(["General"])
        assert result == 1
        
    def test_multiple_tags(self):

        result = combine_tag_bitmasks(["General", "IT & Technology"])
        expected = 1 | 128
        assert result == expected
        
    def test_all_tags(self):

        all_tags = list(TAG_BITMASK_MAPPING.keys())
        result = combine_tag_bitmasks(all_tags)
        expected = sum(TAG_BITMASK_MAPPING.values())
        assert result == expected
        
    def test_empty_list(self):

        result = combine_tag_bitmasks([])
        assert result == 0
        
    def test_unknown_tag(self):

        result = combine_tag_bitmasks(["UnknownTag"])
        assert result == 0
        
    def test_mixed_known_unknown_tags(self):

        result = combine_tag_bitmasks(["General", "UnknownTag", "IT & Technology"])
        expected = 1 | 128
        assert result == expected
        
    def test_duplicate_tags(self):

        result1 = combine_tag_bitmasks(["General", "General"])
        result2 = combine_tag_bitmasks(["General"])
        assert result1 == result2 == 1
        
    def test_specific_role_combinations(self):

        result = combine_tag_bitmasks(["Marketing & Sales", "Finance & Controlling"])
        assert result == 4 | 16
        
        result = combine_tag_bitmasks(["Human Resources", "Legal & Compliance"])
        assert result == 32 | 64

class TestIsAdminUser:

    def test_admin_role_present(self):

        assert is_admin_user(["admin"]) is True
        
    def test_admin_with_other_roles(self):

        assert is_admin_user(["admin", "General", "IT & Technology"]) is True
        
    def test_no_admin_role(self):

        assert is_admin_user(["General", "IT & Technology"]) is False
        
    def test_empty_roles(self):

        assert is_admin_user([]) is False
        
    def test_case_sensitive(self):

        assert is_admin_user(["Admin"]) is False
        assert is_admin_user(["ADMIN"]) is False
        
    def test_admin_in_middle(self):

        assert is_admin_user(["General", "admin", "IT & Technology"]) is True

class TestCreateRetriever:

    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_create_retriever_with_existing_vectorstore(self, mock_embeddings, mock_chroma, mock_env_vars):

        from retriever import create_retriever
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        result = create_retriever(
            existing_vectorstore=mock_vectorstore,
            user_roles=["General"]
        )
        
        mock_vectorstore.as_retriever.assert_called_once()
        assert result == mock_retriever
        
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_create_retriever_admin_no_filter(self, mock_embeddings, mock_chroma, mock_env_vars):

        from retriever import create_retriever
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        result = create_retriever(
            existing_vectorstore=mock_vectorstore,
            user_roles=["admin"]
        )
        
        call_args = mock_vectorstore.as_retriever.call_args
        assert call_args[1]['search_type'] == 'similarity'
        assert 'filter' not in call_args[1]['search_kwargs']
        
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_create_retriever_general_user_no_filter(self, mock_embeddings, mock_chroma, mock_env_vars):

        from retriever import create_retriever
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        result = create_retriever(
            existing_vectorstore=mock_vectorstore,
            user_roles=["General"]
        )
        
        call_args = mock_vectorstore.as_retriever.call_args
        assert 'filter' in call_args[1]['search_kwargs']
        assert call_args[1]['search_kwargs']['filter'] == {'category_bitmask': 1}
        
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_create_retriever_specific_role_with_filter(self, mock_embeddings, mock_chroma, mock_env_vars):

        from retriever import create_retriever
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        
        result = create_retriever(
            existing_vectorstore=mock_vectorstore,
            user_roles=["IT & Technology", "Marketing & Sales"]
        )
        
        call_args = mock_vectorstore.as_retriever.call_args
        expected_bitmask = 128 | 4
        
        if 'filter' in call_args[1]['search_kwargs']:
            assert call_args[1]['search_kwargs']['filter']['category_bitmask'] == expected_bitmask
        
    def test_create_retriever_default_parameters(self, mock_env_vars):

        from retriever import create_retriever
        
        with patch('retriever.init_filtered_vectorstore') as mock_init:
            mock_retriever = Mock()
            mock_init.return_value = mock_retriever
            
            result = create_retriever()
            
            mock_init.assert_called_once()
            call_args = mock_init.call_args
            
            assert call_args[1]['returned_docs'] == 3
            assert call_args[1]['similarity_threshold'] == 0.5
            assert call_args[1]['user_roles'] == ["General"]

class TestInitFilteredVectorstore:

    @patch('retriever.os.makedirs')
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_init_creates_directory(self, mock_embeddings, mock_chroma, mock_makedirs, mock_env_vars):

        from retriever import init_filtered_vectorstore
        
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        init_filtered_vectorstore()
        
        mock_makedirs.assert_called_once()
        
    @patch('retriever.os.makedirs')
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    @patch.dict(os.environ, {"RETRIEVER_DISABLE_FILTER": "true"})
    def test_init_with_disabled_filter(self, mock_embeddings, mock_chroma, mock_makedirs):

        from retriever import init_filtered_vectorstore
        
        mock_vectorstore = Mock()
        mock_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_chroma.return_value = mock_vectorstore
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        result = init_filtered_vectorstore(user_roles=["IT & Technology"])
        
        mock_vectorstore.as_retriever.assert_called_once()
        call_args = mock_vectorstore.as_retriever.call_args
        assert 'filter' in call_args[1]['search_kwargs']
        assert call_args[1]['search_kwargs']['filter'] == {'category_bitmask': 128}
        
    @patch('retriever.os.makedirs')
    @patch('retriever.Chroma')
    @patch('retriever.OllamaEmbeddings')
    def test_reuse_existing_embeddings(self, mock_embeddings, mock_chroma, mock_makedirs, mock_env_vars):

        from retriever import init_filtered_vectorstore
        
        existing_embeddings = Mock()
        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore
        
        init_filtered_vectorstore(existing_embeddings=existing_embeddings)
        
        mock_embeddings.assert_not_called()
        
        mock_chroma.assert_called_once()
        call_args = mock_chroma.call_args
        assert call_args[1]['embedding_function'] == existing_embeddings

class TestTagBitmaskMapping:

    def test_all_tags_have_unique_bitmasks(self):

        bitmasks = list(TAG_BITMASK_MAPPING.values())
        assert len(bitmasks) == len(set(bitmasks))
        
    def test_bitmask_values_are_powers_of_two(self):

        for tag, bitmask in TAG_BITMASK_MAPPING.items():
            assert bitmask > 0 and (bitmask & (bitmask - 1)) == 0, f"{tag} has invalid bitmask {bitmask}"
            
    def test_expected_tags_present(self):

        expected_tags = [
            "General",
            "Research & Development",
            "Marketing & Sales",
            "Production & Manufacturing",
            "Finance & Controlling",
            "Human Resources",
            "Legal & Compliance",
            "IT & Technology",
            "Quality Management",
            "Project Documentation"
        ]
        
        for tag in expected_tags:
            assert tag in TAG_BITMASK_MAPPING, f"Tag '{tag}' missing from mapping"
            
    def test_bitmask_range(self):

        for tag, bitmask in TAG_BITMASK_MAPPING.items():
            assert 1 <= bitmask <= 512, f"{tag} bitmask {bitmask} out of expected range"
