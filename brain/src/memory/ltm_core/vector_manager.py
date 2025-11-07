import os
import time
import math
import pickle
import shutil
import faiss
import numpy as np
from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from agents.memory_agent import TaggedMemories


# ──────────────────────────────────────────────
# Flattened Memory Representation
# ──────────────────────────────────────────────
class FlattenedMemory(BaseModel):
    id: str = Field(default="")
    summary: str = Field(default="")
    tag: str = Field(default="untagged")
    importance: float = Field(default=0.1, ge=0.0, le=1.0)
    confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    entities: List[str] = Field(default_factory=list)
    user_id: str = Field(default="")
    last_accessed: float = Field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "summary": self.summary,
            "tag": self.tag,
            "importance": self.importance,
            "confidence": self.confidence,
            "entities": self.entities,
            "user_id": self.user_id,
            "last_accessed": self.last_accessed,
        }

    @classmethod
    def from_memory_item(cls, memory_item: Any, tag: str, user_id: str):
        """Convert memory dict/object into FlattenedMemory."""
        get = lambda k, default=None: (
            getattr(memory_item, k, None)
            or (memory_item.get(k, default) if isinstance(memory_item, dict) else default)
        )

        # Get timestamp and convert to unix timestamp (float)
        timestamp = get("timestamp", None)
        if timestamp is None:
            timestamp_float = time.time()
        elif isinstance(timestamp, str):
            # Parse ISO-8601 string to unix timestamp
            from datetime import datetime as dt
            dt_obj = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp_float = dt_obj.timestamp()
        elif isinstance(timestamp, (int, float)):
            timestamp_float = float(timestamp)
        else:
            timestamp_float = time.time()

        return cls(
            id=get("id", ""),
            summary=get("summary", ""),
            tag=tag,
            importance=get("importance", 0.1),
            confidence=get("confidence", 0.1),
            entities=get("entities", []),
            user_id=user_id,
            last_accessed=timestamp_float,
        )


# ──────────────────────────────────────────────
# Vector Manager (FAISS + Azure)
# ──────────────────────────────────────────────
class VectorManager:
    SIMILARITY_THRESHOLDS = {
        "duplicate": 0.90,
        "merge": 0.80,
    }

    def __init__(self, user_id: str, base_path="data/faiss"):
        self.user_id = user_id

        # Azure setup
        self.client = AzureOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version="2024-05-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.embedding_model = "text-embedding-3-large"

        # Paths
        self.base_path = os.path.join(base_path, user_id)
        os.makedirs(self.base_path, exist_ok=True)
        self.index_path = os.path.join(self.base_path, "ltm_faiss.index")
        self.meta_path = os.path.join(self.base_path, "ltm_meta.pkl")

        # Load existing data
        self.index = self._load_index()
        self.metadata = self._load_metadata()

    # ──────────────────────────────
    # Internal persistence helpers
    # ──────────────────────────────
    def _load_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
        return None

    def _load_metadata(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                return pickle.load(f)
        return []

    def _save_index(self):
        if self.index:
            faiss.write_index(self.index, self.index_path)

    def _save_metadata(self):
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _l2_to_similarity(self, distance: float, max_distance: float = 1000.0) -> float:
        normalized = min(distance / max_distance, 1.0)
        return 1.0 - normalized

    # ──────────────────────────────
    # Embeddings
    # ──────────────────────────────
    def embed_query(self, query):
        try:
            response = self.client.embeddings.create(model=self.embedding_model, input=query)
            if isinstance(query, list):
                embeddings = np.array([d.embedding for d in response.data]).astype("float32")
            else:
                embeddings = np.array([response.data[0].embedding]).astype("float32")
            return embeddings
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None

    # ──────────────────────────────
    # Importance & Confidence models
    # ──────────────────────────────
    def _update_importance(self, memory: FlattenedMemory, recalled: bool = True, alpha: float = 0.08):
        """Increase importance with diminishing returns."""
        imp = memory.get("importance", 0.1)
        now = time.time()

        # Nonlinear reinforcement: harder to reach 1
        if recalled:
            imp += alpha * (1 - imp) ** 2  # slows down near 1
        # Robust handling of last_accessed: ensure it's a float timestamp
        last_accessed = memory.get("last_accessed", now)
        try:
            last_accessed = float(last_accessed)
        except (TypeError, ValueError):
            last_accessed = now
        # Optional: slow decay over time
        days_since = (now - last_accessed) / 86400
        decay_rate = 0.005
        imp *= math.exp(-decay_rate * days_since)
        memory["importance"] = min(0.98, max(0.05, imp))
        memory["last_accessed"] = now
        return memory

    def _update_confidence(self, memory: FlattenedMemory, match_score: float = 0.8, confirmed: bool = True):
        """Adjust confidence gradually based on verification & decay."""
        now = time.time()
        conf = memory.get("confidence", 0.1)
        last = memory.get("last_accessed", now)

        # Ensure last is a float timestamp, handle string or other types robustly
        try:
            last = float(last)
        except (TypeError, ValueError):
            last = now

        dt_days = (now - last) / 86400

        # Natural decay
        conf *= math.exp(-0.005 * dt_days)

        # Reinforce with evidence
        if confirmed:
            conf += (1 - conf) * 0.05 * match_score
        else:
            conf *= (1 - 0.05 * (1 - match_score))  # penalize mismatch

        conf = max(0.05, min(0.98, conf))
        memory["confidence"] = conf
        memory["last_accessed"] = now
        return memory

    # ──────────────────────────────
    # Core memory operations
    # ──────────────────────────────
    def add_memories(self, tagged_memories: TaggedMemories) -> List[FlattenedMemory]:
        flattened: List[FlattenedMemory] = []
        for tag, memories in tagged_memories.items():
            for m in memories:
                flattened_mem = FlattenedMemory.from_memory_item(m, tag, self.user_id)
                flattened.append(flattened_mem)

        if not flattened:
            print("⚠️ No memories to store.")
            return []

        texts = [m.summary for m in flattened]
        embeddings = self.embed_query(texts)
        if embeddings is None:
            print("❌ Failed to embed memories.")
            return []

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        new_memories, merged, skipped = [], 0, 0

        for flat_mem, embedding in zip(flattened, embeddings):
            if len(self.metadata) > 0:
                distances, indices = self.index.search(embedding.reshape(1, -1), k=1)
                distance = distances[0][0]
                similarity = self._l2_to_similarity(distance)

                if similarity >= self.SIMILARITY_THRESHOLDS["duplicate"]:
                    skipped += 1
                    continue
                elif similarity >= self.SIMILARITY_THRESHOLDS["merge"]:
                    idx = indices[0][0]
                    existing = self.metadata[idx]
                    existing = self._update_importance(existing)
                    existing = self._update_confidence(existing, similarity)
                    existing["entities"] = list(
                        set(existing.get("entities", []) + flat_mem.entities)
                    )
                    merged += 1
                    continue

            new_memories.append((flat_mem, embedding))
        if new_memories:
            new_embs = np.array([e for _, e in new_memories]).astype("float32")
            self.index.add(new_embs)
            self.metadata.extend([m.to_dict() for m, _ in new_memories])
            self._save_index()
            self._save_metadata()
            print(f"✓ Stored {len(new_memories)} new memories ({merged} merged, {skipped} skipped)")
        else:
            self._save_metadata()
        return [m for m, _ in new_memories]

    # ──────────────────────────────
    # Search & Reinforcement
    # ──────────────────────────────
    def search(self, query: str, top_k: int = 10):
        if not self.index or not self.metadata:
            return []

        embeddings = self.embed_query(query)
        if embeddings is None:
            return []

        distances, indices = self.index.search(embeddings, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            mem = self.metadata[idx]
            similarity = self._l2_to_similarity(dist)
            mem = self._update_importance(mem)
            mem = self._update_confidence(mem, similarity)
            results.append({**mem, "similarity": similarity, "index": idx})

        self._save_metadata()
        return results

    def get_memory_by_id(self, memory_id: str):
        for idx, memory in enumerate(self.metadata):
            if memory.get("id") == memory_id:
                self.metadata[idx] = self._update_importance(memory)
                self.metadata[idx] = self._update_confidence(memory)
                self._save_metadata()
                return {**memory, "index": idx}
        return None

    # ──────────────────────────────────────────────
    # Export to Excel
    # ──────────────────────────────────────────────
    def export_to_excel(self, output_path: str = None) -> str:
        """
        Export all memories to an Excel file.

        Args:
            output_path: Path to save the Excel file. If None, saves to data/exports/{user_id}_memories.xlsx

        Returns:
            Path to the exported Excel file, or None if failed
        """
        import os

        if not self.metadata:
            print(f"⚠️ No memories to export for user {self.user_id}")
            return None

        # Determine output path and ensure directory exists
        try:
            if output_path is None:
                output_path = os.path.join(output_path, f"{self.user_id}_memories.xlsx")
            else:
                # Ensure directory exists for given output_path
                output_dir = os.path.dirname(os.path.abspath(output_path))
                os.makedirs(output_dir, exist_ok=True)

            # Convert metadata to DataFrame
            df_data = []
            for memory in self.metadata:
                df_data.append({
                    "ID": memory.get("id", ""),
                    "Summary": memory.get("summary", ""),
                    "Tag": memory.get("tag", ""),
                    "Importance": memory.get("importance", 0.0),
                    "Confidence": memory.get("confidence", 0.0),
                    "Entities": ", ".join(memory.get("entities", [])),
                    "Last Accessed": self._unix_to_iso(memory.get("last_accessed", time.time())),
                    "User ID": memory.get("user_id", "")
                })

            df = pd.DataFrame(df_data)

            # Write to Excel with formatting
            try:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Memories', index=False)

                    # Format the worksheet
                    worksheet = writer.sheets['Memories']

                    # Set column widths
                    worksheet.column_dimensions['A'].width = 36  # ID (UUID)
                    worksheet.column_dimensions['B'].width = 50  # Summary
                    worksheet.column_dimensions['C'].width = 15  # Tag
                    worksheet.column_dimensions['D'].width = 12  # Importance
                    worksheet.column_dimensions['E'].width = 12  # Confidence
                    worksheet.column_dimensions['F'].width = 30  # Entities
                    worksheet.column_dimensions['G'].width = 25  # Last Accessed
                    worksheet.column_dimensions['H'].width = 15  # User ID

                    # Freeze header row
                    worksheet.freeze_panes = 'A2'
            except Exception as ex:
                print(f"❌ Error writing Excel file: {ex}")
                return None

            # Double-check if file really exists
            if os.path.exists(output_path):
                print(f"✓ Exported {len(df_data)} memories to {output_path}")
                return output_path
            else:
                print(f"❌ Excel file was not created at: {output_path}")
                return None

        except Exception as e:
            print(f"❌ Excel export failed: {e}")
            return None
    
    def _unix_to_iso(self, unix_timestamp: float) -> str:
        """Convert unix timestamp to ISO-8601 string."""
        try:
            return datetime.fromtimestamp(unix_timestamp).isoformat()
        except:
            return ""
    
    def delete_all(self) -> bool:
        """
        Delete all FAISS data for this user_id.
        Removes the index file, metadata file, and the entire user directory.
        
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Clear in-memory data
            self.index = None
            self.metadata = []
            
            # Delete index file if it exists
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
                print(f"✓ Deleted FAISS index: {self.index_path}")
            
            # Delete metadata file if it exists
            if os.path.exists(self.meta_path):
                os.remove(self.meta_path)
                print(f"✓ Deleted metadata file: {self.meta_path}")
            
            # Delete the entire user directory if it exists and is empty
            if os.path.exists(self.base_path):
                try:
                    # Try to remove directory (will only work if empty)
                    os.rmdir(self.base_path)
                    print(f"✓ Deleted user directory: {self.base_path}")
                except OSError:
                    # Directory not empty or other error, try to remove all contents
                    shutil.rmtree(self.base_path)
                    print(f"✓ Deleted user directory and all contents: {self.base_path}")
            
            print(f"✓ Successfully deleted all FAISS data for user: {self.user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting FAISS data for user {self.user_id}: {e}")
            import traceback
            traceback.print_exc()
            return False