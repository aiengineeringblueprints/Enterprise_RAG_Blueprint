"""
Module for document tag and role management.
This module is used by both loader and chain services to ensure consistent tagging.
"""

# Tag-to-bitmask mapping for efficient filtering
TAG_BITMASK_MAPPING = {
    "General": 1 << 0,                          # Bit 0: 1      (binary: 0000000001)
    "Research & Development": 1 << 1,           # Bit 1: 2      (binary: 0000000010)
    "Marketing & Sales": 1 << 2,                # Bit 2: 4      (binary: 0000000100)
    "Production & Manufacturing": 1 << 3,       # Bit 3: 8      (binary: 0000001000)
    "Finance & Controlling": 1 << 4,            # Bit 4: 16     (binary: 0000010000)
    "Human Resources": 1 << 5,                  # Bit 5: 32     (binary: 0000100000)
    "Legal & Compliance": 1 << 6,               # Bit 6: 64     (binary: 0001000000)
    "IT & Technology": 1 << 7,                  # Bit 7: 128    (binary: 0010000000)
    "Quality Management": 1 << 8,               # Bit 8: 256    (binary: 0100000000)
    "Project Documentation": 1 << 9             # Bit 9: 512    (binary: 1000000000)
}

def get_tag_bitmask(tag: str) -> int:
    """
    Converts a tag name into a bitmask value.
    
    Args:
        tag (str): The tag name
        
    Returns:
        int: The corresponding bitmask value or 0 if tag is unknown
    """
    return TAG_BITMASK_MAPPING.get(tag, 0)

def combine_tag_bitmasks(tags: list) -> int:
    """
    Combines multiple tags into a single bitmask.
    
    Args:
        tags (list): List of tag names
        
    Returns:
        int: Combined bitmask value
    """
    combined_mask = 0
    for tag in tags:
        combined_mask |= get_tag_bitmask(tag)
    return combined_mask

def is_admin_user(user_roles: list) -> bool:
    """
    Checks if a user has admin rights.
    
    Args:
        user_roles (list): List of user roles
        
    Returns:
        bool: True if the user is admin, False otherwise
    """
    return "admin" in user_roles
