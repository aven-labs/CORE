from typing import List, Dict

from autogen_core.models import AssistantMessage
from memory.ltm_core.vector_manager import VectorManager
from memory.ltm_core.neo4j_db import GraphManager

class MemoryRetriever:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_manager = VectorManager(user_id=user_id)
        self.graph_manager = GraphManager()

    def search_memories(self, user_message: str, top_k: int = 5) -> Dict:
        """
        Retrieve relevant memories based on user message using both FAISS and Neo4j.
        """
        # ───────────────────────────────────────────────
        # STEP 1: Search in FAISS
        # ───────────────────────────────────────────────
        similar_memories = self.vector_manager.search(user_message, top_k=top_k)
        # Example format: [{"id": "uuid", "summary": "...", "score": 0.78}]
        memory_ids = [m["id"] for m in similar_memories]
        # ───────────────────────────────────────────────
        # STEP 2: Expand via Neo4j relationships
        # ───────────────────────────────────────────────
        related_memories_ids = self.graph_manager.get_related_memory_ids(memory_ids)

        # ───────────────────────────────────────────────
        # STEP 3: Combine and deduplicate
        # ───────────────────────────────────────────────
        all_memories = {m["id"]: m for m in similar_memories}
        for r in related_memories_ids:
            if r not in all_memories.keys():
                all_memories[r] = self.get_memory_by_id(r)

        # ───────────────────────────────────────────────
        # STEP 4: Sort by FAISS similarity or importance
        # ───────────────────────────────────────────────
        # Join all memories into a single context string
        sorted_memories = sorted(
            all_memories.values(),
            key=lambda x: x.get("score", 0) * x.get("importance", 1),
            reverse=True
        )
        # Create single context string from summaries
        context_string = "\n".join(m.get("summary", "") for m in sorted_memories if m.get("summary"))
        return context_string

    # ───────────────────────────────────────────────
    # Direct Memory Searches
    # ───────────────────────────────────────────────
    def get_memory_by_id(self, memory_id: str) -> Dict:
        """
        Retrieve a specific memory by its ID.
        """
        memory = self.vector_manager.get_memory_by_id(memory_id)
        if memory:
            return {
                "status": "found",
                "memory": memory
            }
        return {
            "status": "not_found",
            "message": f"Memory with ID {memory_id} not found"
        }