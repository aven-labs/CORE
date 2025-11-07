"""
Tag Manager - stores and retrieves tags by user_id in Supabase
"""
from typing import List
from brain_core.sup_extractor import supabase_service


class TagManager:
    """Manages tags in Supabase storage"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.table_name = "user_tags"

    def save_tag(self, tag: str):
        """Save a tag for this user"""
        supabase_service.save_tag(self.user_id, tag, self.table_name)

    def save_tags(self, tags: List[str]):
        """Save multiple tags for this user"""
        supabase_service.save_tags(self.user_id, tags, self.table_name)

    def get_tags(self) -> List[str]:
        """Get all unique tags for this user"""
        return supabase_service.get_tags(self.user_id, self.table_name)

    def clear_tags(self):
        """Clear all tags for this user"""
        supabase_service.clear_tags(self.user_id, self.table_name)

    def tag_exists(self, tag: str) -> bool:
        """Check if a tag already exists for this user"""
        return supabase_service.tag_exists(self.user_id, tag, self.table_name)
