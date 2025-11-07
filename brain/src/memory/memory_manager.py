import asyncio
import threading
from typing import List, Dict, Any

from autogen_core.models import AssistantMessage, UserMessage
from memory.stm import STM
from memory.ltm import MemoryService


class MemoryManager:
    """Manages conversation memory by passing directly to STM for SQL storage"""

    def __init__(self, user_id: str = None):
        self.user_id = user_id

        # Initialize STM (SQL Memory) for persistent storage
        try:
            self.stm = STM()
        except Exception as e:
            print(f"Warning: Could not initialize STM storage: {e}")
            self.stm = None

        # Initialize LTM (Long-Term Memory) placeholder
        try:
            self.ltm = MemoryService(user_id=self.user_id)
        except Exception as e:
            print(f"Warning: Could not initialize LTM: {e}")
            self.ltm = None

    def add_messages(self, messages: List[Dict[str, Any]]):
        """Pass messages to STM (last 10) and LTM (if exceeds)"""
        if not messages:
            return
        messages_to_store = messages if isinstance(
            messages, list) else [messages]

        if not self.stm or not self.user_id:
            return

        try:
            current_messages = self.stm.get_messages(
                self.user_id, limit=100) if self.stm else []
            all_messages = current_messages + messages_to_store
            if len(all_messages) >= 30:
                ltm_messages = all_messages[:15]
                stm_messages = all_messages[-15:]

                if self.ltm and ltm_messages:
                    print(f"✓ Moved first 10 messages to LTM, kept last 10 in STM")

                    async def run_ltm_task():
                        try:
                            await self.ltm.process_conversation(ltm_messages)
                            self.stm.clear_messages(self.user_id)
                            self.stm.save_messages(self.user_id, stm_messages)
                        except Exception as e:
                            print(f"⚠️ LTM processing failed: {e}")

                    def start_background_ltm():
                        try:
                            asyncio.run(run_ltm_task())
                        except Exception as e:
                            print(f"⚠️ Background LTM thread error: {e}")
                    threading.Thread(
                        target=start_background_ltm, daemon=True).start()
            else:
                self.stm.save_messages(self.user_id, messages_to_store)

        except Exception as e:
            print(f"Warning: Could not save: {e}")

    def search_memories(self, user_message: str, top_k: int = 10) -> Dict:
        """Search memories using the MemoryRetriever"""
        if not self.ltm or not self.user_id:
            return []
        return self.ltm.search_memories(user_message, top_k=top_k)

    def get_messages(self, message: str) -> List[Dict[str, Any]]:
        """Get messages directly from STM (SQL storage)"""
        if self.stm and self.user_id:
            try:
                stm_messages = self.stm.get_messages(self.user_id)
                context_string = self.search_memories(message)
                messages = []
                for message in stm_messages:
                    if message.get("role") == "user":
                        messages.append(UserMessage(
                            content=message.get("content"), source="user"))
                    else:
                        messages.append(AssistantMessage(content=message.get(
                            "content"), source=message.get("role")))
                return context_string, messages
            except Exception as e:
                print(f"Warning: Could not get messages: {e}")
        return "", []

    def export_to_excel(self, output_path: str):
        """Get messages directly from LTM (Long-Term Memory)"""
        self.ltm.export_to_excel(output_path)
    
    def delete_all(self) -> bool:
        """
        Delete all user data from both STM and LTM:
        - STM: Conversation history from Supabase
        - LTM: FAISS vectors, Neo4j graph, and Supabase tags
        
        Returns:
            True if all deletions were successful, False otherwise
        """
        success_count = 0
        total_operations = 2
        
        print(f"\n🗑️  Starting complete deletion for user: {self.user_id}")
        print("=" * 60)
        
        # 1. Delete STM (Short-Term Memory) - conversation history
        if self.stm and self.user_id:
            try:
                self.stm.clear_messages(self.user_id)
                print("✓ STM conversation history deleted successfully")
                success_count += 1
            except Exception as e:
                print(f"❌ Error deleting STM data: {e}")
        else:
            print("⚠️  STM not initialized, skipping STM deletion")
            total_operations -= 1
        
        # 2. Delete LTM (Long-Term Memory) - vectors, graph, tags
        if self.ltm:
            try:
                if self.ltm.delete_all():
                    print("✓ LTM data deleted successfully")
                    success_count += 1
                else:
                    print("❌ Failed to delete some LTM data")
            except Exception as e:
                print(f"❌ Error deleting LTM data: {e}")
        else:
            print("⚠️  LTM not initialized, skipping LTM deletion")
            total_operations -= 1
        
        print("=" * 60)
        if success_count == total_operations and total_operations > 0:
            print(f"✅ Successfully deleted all data for user: {self.user_id}")
            return True
        else:
            print(f"⚠️  Deleted {success_count}/{total_operations} memory systems for user: {self.user_id}")
            return False
