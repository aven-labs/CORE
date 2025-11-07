"""
Centralized Supabase operations extractor.

This module provides a unified interface for all Supabase database operations
used across the application, including:
- Conversation history (STM)
- Tag management
- Agent management
- Authentication
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseService:
    """Centralized Supabase service for all database operations."""
    
    _instance: Optional['SupabaseService'] = None
    _client: Optional[Client] = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super(SupabaseService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize Supabase client if not already initialized."""
        if self._client is None:
            self._client = self._init_client()
    
    def _init_client(self) -> Client:
        """Initialize Supabase client with service role key (bypasses RLS)."""
        supabase_url = os.getenv("SUPABASE_URL")
        # Prefer service role key for server-side operations (bypasses RLS)
        service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        supabase_key = service_role_key or anon_key
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set in environment variables")
        
        if service_role_key:
            logger.info("Using SUPABASE_SERVICE_ROLE_KEY (bypasses RLS)")
        else:
            logger.warning("Using SUPABASE_ANON_KEY (RLS may block queries)")
        
        return create_client(supabase_url, supabase_key)
    
    @property
    def client(self) -> Client:
        """Get the Supabase client."""
        if self._client is None:
            self._client = self._init_client()
        return self._client
    
    # ============================================================================
    # Conversation History (STM) Operations
    # ============================================================================
    
    def save_messages(self, user_id: str, messages: List[Dict[str, Any]], table_name: str = "conversation_history"):
        """Save chat messages to Supabase safely."""
        if not messages:
            return
        try:
            for message in messages:
                # Prepare message data
                message_data = {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "name": message.get("name")
                }
                
                # Prepare insert data
                insert_data = {
                    "user_id": user_id,
                    "message_data": message_data
                }
                
                # Get timestamp from message if provided
                timestamp = message.get("timestamp")
                if timestamp:
                    # If timestamp is a datetime object, convert to ISO format string
                    if isinstance(timestamp, datetime):
                        insert_data["timestamp"] = timestamp.isoformat()
                    elif isinstance(timestamp, str):
                        # Already a string, use as-is
                        insert_data["timestamp"] = timestamp
                # If timestamp is None, let Supabase use DEFAULT CURRENT_TIMESTAMP
                
                # Insert into Supabase
                self.client.table(table_name).insert(insert_data).execute()
        except Exception as e:
            logger.error(f"Error saving messages to Supabase for user {user_id}: {e}", exc_info=True)
    
    def get_messages(self, user_id: str, limit: int = 100, table_name: str = "conversation_history") -> List[Dict[str, Any]]:
        """Get messages from Supabase"""
        try:
            # Query Supabase table
            response = self.client.table(table_name)\
                .select("message_data, timestamp")\
                .eq("user_id", user_id)\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .execute()
            
            # Convert response to list of messages
            messages = []
            for row in response.data:
                message_data = row.get("message_data", {})
                timestamp = row.get("timestamp")
                
                # Combine message_data with timestamp
                message = {
                    **message_data,
                    "timestamp": timestamp
                }
                messages.append(message)
            
            # Reverse to get chronological order (oldest first)
            return messages[::-1]
        except Exception as e:
            logger.error(f"Error getting messages from Supabase for user {user_id}: {e}", exc_info=True)
            return []
    
    def clear_messages(self, user_id: str, table_name: str = "conversation_history"):
        """Clear messages for a user"""
        try:
            self.client.table(table_name)\
                .delete()\
                .eq("user_id", user_id)\
                .execute()
            logger.debug(f"Cleared messages for user: {user_id}")
        except Exception as e:
            logger.error(f"Error clearing messages from Supabase for user {user_id}: {e}", exc_info=True)
    
    # ============================================================================
    # Tag Operations
    # ============================================================================
    
    def save_tag(self, user_id: str, tag: str, table_name: str = "user_tags"):
        """Save a tag for this user"""
        if not tag:
            return
        
        # Check if tag already exists
        if self.tag_exists(user_id, tag, table_name):
            return
        
        try:
            self.client.table(table_name).insert({
                "user_id": user_id,
                "tag": tag
            }).execute()
        except Exception as e:
            logger.error(f"Error saving tag '{tag}' to Supabase for user {user_id}: {e}", exc_info=True)
    
    def save_tags(self, user_id: str, tags: List[str], table_name: str = "user_tags"):
        """Save multiple tags for this user"""
        for tag in tags:
            self.save_tag(user_id, tag, table_name)
    
    def get_tags(self, user_id: str, table_name: str = "user_tags") -> List[str]:
        """Get all unique tags for this user"""
        try:
            response = self.client.table(table_name)\
                .select("tag")\
                .eq("user_id", user_id)\
                .order("tag")\
                .execute()
            
            # Extract unique tags
            tags = set()
            for row in response.data:
                tag = row.get("tag")
                if tag:
                    tags.add(tag)
            
            return sorted(list(tags))
        except Exception as e:
            logger.error(f"Error getting tags from Supabase for user {user_id}: {e}", exc_info=True)
            return []
    
    def clear_tags(self, user_id: str, table_name: str = "user_tags"):
        """Clear all tags for this user"""
        try:
            self.client.table(table_name)\
                .delete()\
                .eq("user_id", user_id)\
                .execute()
            logger.debug(f"Cleared tags for user: {user_id}")
        except Exception as e:
            logger.error(f"Error clearing tags from Supabase for user {user_id}: {e}", exc_info=True)
    
    def tag_exists(self, user_id: str, tag: str, table_name: str = "user_tags") -> bool:
        """Check if a tag already exists for this user"""
        try:
            response = self.client.table(table_name)\
                .select("id")\
                .eq("user_id", user_id)\
                .eq("tag", tag)\
                .limit(1)\
                .execute()
            
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Error checking tag existence in Supabase for user {user_id}: {e}", exc_info=True)
            return False
    
    # ============================================================================
    # Agent Operations
    # ============================================================================
    
    def get_user_active_agents(self, user_id: str) -> List[Dict[str, Any]]:
        """Fetch all active agents for a user from Supabase.
        
        Args:
            user_id: The user ID to fetch agents for
        """
        try:
            # Use the default client which already uses service role key
            # This bypasses RLS for all operations
            client = self.client
            
            # Get active installed agents for the user
            installed_response = client.table("installed_agents").select(
                "id, agent_id, is_active"
            ).eq("user_id", user_id).eq("is_active", True).execute()
            
            if not installed_response.data:
                logger.debug(f"No active agents found for user: {user_id}")
                return []
            
            # Extract agent_ids
            agent_ids = [str(row["agent_id"]) for row in installed_response.data]
            installed_agent_map = {str(row["agent_id"]): str(row["id"]) for row in installed_response.data}

            # Query agents table to get full details including access_url
            agents_response = self.client.table("agents").select(
                "id, name, is_offline, access_url"
            ).in_("id", agent_ids).execute()
            
            # Build agent list
            agents = []
            for agent_row in agents_response.data:
                agent_id = str(agent_row["id"])
                agent_info = {
                    "id": agent_id,  # Use agent_id as the tool ID
                    "name": agent_row.get("name", "Unknown Agent"),
                    "local": agent_row.get("is_offline", False),  # is_offline determines if local
                    "access_url": agent_row.get("access_url", ""),
                    "installed_agent_id": installed_agent_map.get(agent_id, "")  # Keep installed_agents.id for reference
                }
                agents.append(agent_info)
            
            logger.debug(f"Loaded {len(agents)} active agent(s) for user: {user_id}")
            return agents
        
        except Exception as e:
            logger.error(f"Error fetching user agents from Supabase for user {user_id}: {e}", exc_info=True)
            return []
    
    def update_agent_last_used(self, installed_agent_id: str):
        """Update the last_used_at timestamp for an installed agent."""
        try:
            self.client.table("installed_agents").update({
                "last_used_at": datetime.now().isoformat()
            }).eq("id", installed_agent_id).execute()
        except Exception as e:
            # Silently fail - this is not critical
            logger.warning(f"Could not update last_used_at for agent {installed_agent_id}: {e}")
    
    # ============================================================================
    # Authentication Operations
    # ============================================================================
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify JWT token with Supabase and return user details
        
        Args:
            token: JWT token string
            
        Returns:
            dict: User details if token is valid, None otherwise
        """
        try:
            # Set the auth token and get user details
            # The get_user method verifies the token automatically
            response = self.client.auth.get_user(token)
            
            if response and hasattr(response, 'user') and response.user:
                user_data = {
                    "id": response.user.id,
                    "email": response.user.email,
                    "email_verified": response.user.email_confirmed_at is not None,
                    "created_at": response.user.created_at,
                    "user_metadata": response.user.user_metadata or {},
                    "app_metadata": response.user.app_metadata or {}
                }
                return user_data
            return None
            
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}", exc_info=True)
            return None


# Global singleton instance
supabase_service = SupabaseService()

