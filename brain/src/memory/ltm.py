from typing import List, Dict, Any
from memory.ltm_core.tag_manager import TagManager
from agents.memory_agent import MemoryAgent, TaggedMemories
from memory.ltm_core.vector_manager import FlattenedMemory, VectorManager
from memory.ltm_core.neo4j_db import GraphManager
from memory.ltm_core.retriever import MemoryRetriever


class MemoryService:
    """
    High-level service to extract, tag, and store memories across:
      • Supabase (for tags / metadata)
      • LanceDB with Azure Blob (for vector retrieval)
      • Neo4j (for relationships and semantic linking)
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.tag_manager = TagManager(user_id=user_id)
        self.memory_agent = MemoryAgent(user_id=self.user_id)
        self.vector_manager = VectorManager(user_id=self.user_id)
        self.graph_manager = GraphManager()
        self.retriever = MemoryRetriever(user_id=self.user_id)

    # ──────────────────────────────────────────────
    # 2️⃣ Main Processing Pipeline
    # ──────────────────────────────────────────────
    async def process_conversation(
        self, conversation: List[Dict[str, Any]]
    ):
        """
        Full memory pipeline:
          1. Extract memories (via LLM agent)
          2. Store tags in Supabase
          3. Store embeddings in LanceDB (Azure Blob)
          4. Create graph relationships in Neo4j
        """
        tagged_memories = await self._extract_memories(conversation)
        if not tagged_memories:
            print(f"⚠️ No memories extracted for user: {self.user_id}")
            return {}
        self._store_tags(tagged_memories)
        flattened_memories = self._store_vectors(tagged_memories)
        self._store_graph(flattened_memories)

    def search_memories(self, user_message: str, top_k: int = 5) -> Dict:
        """Search memories using the MemoryRetriever"""
        if not self.retriever or not self.user_id:
            return []
        return self.retriever.search_memories(user_message, top_k=top_k)
    # ──────────────────────────────────────────────
    # 3️⃣ Extract Memories (via LLM Agent)
    # ──────────────────────────────────────────────

    async def _extract_memories(self, conversation: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Use the MemoryAgent to convert chat into structured tagged memories."""
        try:
            tagged_memories = await self.memory_agent.extract_memories(conversation)
            return tagged_memories or {}
        except Exception as e:
            print(f"❌ Error extracting memories: {e}")
            return {}

    # ──────────────────────────────────────────────
    # 4️⃣ Store Tags (SQL)
    # ──────────────────────────────────────────────
    def _store_tags(self, tagged_memories: TaggedMemories) -> None:
        """Store unique tags in SQL via TagManager."""
        try:
            tags = list(tagged_memories.keys())
            if tags:
                self.tag_manager.save_tags(tags)
                # print(f"✓ Tags saved in SQL for user: {self.user_id}")
        except Exception as e:
            print(f"❌ Error storing tags: {e}")

    # ──────────────────────────────────────────────
    # 5️⃣ Store Vectors (LanceDB + Azure Blob)
    # ──────────────────────────────────────────────
    def _store_vectors(self, tagged_memories: TaggedMemories) -> List[FlattenedMemory]:
        """Store embeddings and metadata in LanceDB with Azure Blob storage."""
        flattened_memories: List[FlattenedMemory] = []
        try:
            flattened_memories = self.vector_manager.add_memories(tagged_memories)
            print(
                f"✓ Stored long-term memory in LanceDB (Azure Blob) for user: {self.user_id}")
        except Exception as e:
            print(f"❌ Error storing vectors: {e}")
        return flattened_memories
    # ──────────────────────────────────────────────
    # 6️⃣ Store Relationships (Neo4j)
    # ──────────────────────────────────────────────

    def _store_graph(self, flattened_memories: List[FlattenedMemory]) -> None:
        """Store relationships between user, tags, and entities in Neo4j."""
        try:
            self.graph_manager.store_memory_graph(
                self.user_id, flattened_memories)
            print(f"✓ Graph relationships updated for user: {self.user_id}")
        except Exception as e:
            print(f"❌ Error storing graph data: {e}")

    def export_to_excel(self, output_path: str):
        """Get messages directly from LTM (Long-Term Memory)"""
        self.vector_manager.export_to_excel(output_path)
    
    # ──────────────────────────────────────────────
    # 7️⃣ Delete All User Data
    # ──────────────────────────────────────────────
    def delete_all(self) -> bool:
        """
        Delete all user data across all storage systems:
        - FAISS vector data (index and metadata files)
        - Neo4j graph data (memories, relationships, entities, tags)
        - Supabase tags
        
        Returns:
            True if all deletions were successful, False otherwise
        """
        success_count = 0
        total_operations = 3
        
        print(f"\n🗑️  Starting deletion of all data for user: {self.user_id}")
        print("=" * 60)
        
        # 1. Delete FAISS vector data
        try:
            if self.vector_manager.delete_all():
                print("✓ FAISS data deleted successfully")
                success_count += 1
            else:
                print("❌ Failed to delete FAISS data")
        except Exception as e:
            print(f"❌ Error deleting FAISS data: {e}")
        
        # 2. Delete Neo4j graph data
        try:
            if self.graph_manager.delete_all(self.user_id):
                print("✓ Neo4j graph data deleted successfully")
                success_count += 1
            else:
                print("❌ Failed to delete Neo4j graph data")
        except Exception as e:
            print(f"❌ Error deleting Neo4j graph data: {e}")
        
        # 3. Delete Supabase tags
        try:
            self.tag_manager.clear_tags()
            print("✓ Supabase tags deleted successfully")
            success_count += 1
        except Exception as e:
            print(f"❌ Error deleting Supabase tags: {e}")
        
        print("=" * 60)
        if success_count == total_operations:
            print(f"✅ Successfully deleted all data for user: {self.user_id}")
            return True
        else:
            print(f"⚠️  Deleted {success_count}/{total_operations} data sources for user: {self.user_id}")
            return False
