"""
STM (Short-Term Memory) - Supabase storage for conversation history
"""
from typing import List, Dict, Any
from brain_core.sup_extractor import supabase_service


class STM:
    """Supabase storage for messages"""

    def __init__(self):
        self.table_name = "conversation_history"

    def save_messages(self, user_id: str, messages: List[Dict[str, Any]]):
        """Save chat messages to Supabase safely."""
        supabase_service.save_messages(user_id, messages, self.table_name)

    def get_messages(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get messages from Supabase"""
        return supabase_service.get_messages(user_id, limit, self.table_name)

    def clear_messages(self, user_id: str):
        """Clear messages for a user"""
        supabase_service.clear_messages(user_id, self.table_name)
