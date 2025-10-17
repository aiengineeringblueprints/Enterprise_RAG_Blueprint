
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tag_management import (
    TAG_BITMASK_MAPPING,
    get_tag_bitmask,
    combine_tag_bitmasks,
    is_admin_user
)

class TestTagBitmaskMapping:

    def test_all_tags_have_unique_bitmasks(self):

        bitmasks = list(TAG_BITMASK_MAPPING.values())
        assert len(bitmasks) == len(set(bitmasks))
        
    def test_bitmask_values_are_powers_of_two(self):

        for tag, bitmask in TAG_BITMASK_MAPPING.items():
            assert bitmask > 0
            assert (bitmask & (bitmask - 1)) == 0
            
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
            assert tag in TAG_BITMASK_MAPPING
            
    def test_general_tag_is_first_bit(self):

        assert TAG_BITMASK_MAPPING["General"] == 1
        
    def test_bitmask_range(self):

        for bitmask in TAG_BITMASK_MAPPING.values():
            assert 1 <= bitmask <= 1024

class TestGetTagBitmask:

    def test_get_general_tag(self):

        assert get_tag_bitmask("General") == 1
        
    def test_get_research_development_tag(self):

        assert get_tag_bitmask("Research & Development") == 2
        
    def test_get_it_technology_tag(self):

        assert get_tag_bitmask("IT & Technology") == 128
        
    def test_get_unknown_tag_returns_zero(self):

        assert get_tag_bitmask("Unknown Tag") == 0
        
    def test_get_empty_string_tag(self):

        assert get_tag_bitmask("") == 0
        
    def test_get_none_tag(self):

        assert get_tag_bitmask(None) == 0
        
    def test_case_sensitive_tag(self):

        assert get_tag_bitmask("general") == 0
        assert get_tag_bitmask("General") == 1
        
    def test_all_valid_tags(self):

        for tag in TAG_BITMASK_MAPPING.keys():
            bitmask = get_tag_bitmask(tag)
            assert bitmask > 0
            assert bitmask == TAG_BITMASK_MAPPING[tag]

class TestCombineTagBitmasks:

    def test_single_tag(self):

        result = combine_tag_bitmasks(["General"])
        assert result == 1
        
    def test_multiple_tags(self):

        result = combine_tag_bitmasks(["General", "Research & Development"])
        assert result == 3
        
    def test_all_tags(self):

        all_tags = list(TAG_BITMASK_MAPPING.keys())
        result = combine_tag_bitmasks(all_tags)
        expected = sum(TAG_BITMASK_MAPPING.values())
        assert result == expected
        
    def test_empty_list(self):

        result = combine_tag_bitmasks([])
        assert result == 0
        
    def test_unknown_tag_ignored(self):

        result = combine_tag_bitmasks(["General", "Unknown Tag"])
        assert result == 1
        
    def test_mixed_known_unknown_tags(self):

        result = combine_tag_bitmasks([
            "General",
            "Unknown Tag",
            "IT & Technology",
            "Another Unknown"
        ])
        assert result == 129
        
    def test_duplicate_tags(self):

        result1 = combine_tag_bitmasks(["General", "General"])
        result2 = combine_tag_bitmasks(["General"])
        assert result1 == result2 == 1
        
    def test_order_independence(self):

        result1 = combine_tag_bitmasks(["General", "IT & Technology", "Finance & Controlling"])
        result2 = combine_tag_bitmasks(["Finance & Controlling", "General", "IT & Technology"])
        assert result1 == result2
        
    def test_specific_role_combinations(self):

        result = combine_tag_bitmasks(["Finance & Controlling", "IT & Technology"])
        assert result == 144
        
        result = combine_tag_bitmasks(["Human Resources", "Legal & Compliance"])
        assert result == 96

class TestIsAdminUser:

    def test_admin_role_present(self):

        assert is_admin_user(["admin"]) is True
        
    def test_admin_with_other_roles(self):

        assert is_admin_user(["admin", "user", "manager"]) is True
        
    def test_no_admin_role(self):

        assert is_admin_user(["user", "manager", "viewer"]) is False
        
    def test_empty_roles(self):

        assert is_admin_user([]) is False
        
    def test_case_sensitive(self):

        assert is_admin_user(["Admin"]) is False
        assert is_admin_user(["ADMIN"]) is False
        assert is_admin_user(["admin"]) is True
        
    def test_admin_in_middle(self):

        roles = ["user", "admin", "viewer"]
        assert is_admin_user(roles) is True
        
    def test_admin_substring_not_matched(self):

        assert is_admin_user(["administrator"]) is False
        assert is_admin_user(["admin_user"]) is False
        
    def test_none_roles(self):

        with pytest.raises(TypeError):
            is_admin_user(None)

class TestIntegration:

    def test_real_world_document_tagging(self):

        tags = ["Finance & Controlling", "Legal & Compliance", "IT & Technology"]
        combined = combine_tag_bitmasks(tags)
        
        assert combined & get_tag_bitmask("Finance & Controlling") != 0
        assert combined & get_tag_bitmask("Legal & Compliance") != 0
        assert combined & get_tag_bitmask("IT & Technology") != 0
        
    def test_filter_by_bitmask(self):

        user_bitmask = combine_tag_bitmasks(["Finance & Controlling", "IT & Technology"])
        
        doc_bitmask = get_tag_bitmask("Finance & Controlling")
        
        assert (user_bitmask & doc_bitmask) != 0
        
    def test_general_accessible_by_all(self):

        general_mask = get_tag_bitmask("General")
        
        assert general_mask == 1
        
        for tag in TAG_BITMASK_MAPPING.keys():
            tag_mask = get_tag_bitmask(tag)
            assert tag_mask > 0
            
    def test_bitmask_reconstruction(self):

        original_tags = ["General", "IT & Technology", "Quality Management"]
        combined = combine_tag_bitmasks(original_tags)
        
        found_tags = []
        for tag, bitmask in TAG_BITMASK_MAPPING.items():
            if combined & bitmask:
                found_tags.append(tag)
        
        assert set(found_tags) == set(original_tags)
