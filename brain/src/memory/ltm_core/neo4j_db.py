from typing import List
from neo4j import GraphDatabase
import os
import dotenv

from memory.ltm_core.vector_manager import FlattenedMemory

dotenv.load_dotenv()

class GraphManager:
    """Handles Neo4j graph storage and retrieval for relationships, entities, and tags."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    # ───────────────────────────────────────────────
    # CHECK EXISTING MEMORIES
    # ───────────────────────────────────────────────
    def get_existing_memory_ids(self, memory_ids: List[str]) -> set:
        """Get which memory IDs from the given list already exist in Neo4j."""
        if not memory_ids:
            return set()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Memory)
                WHERE m.id IN $memory_ids
                RETURN COLLECT(m.id) as existing_ids
            """, memory_ids=memory_ids)
            record = result.single()
            return set(record["existing_ids"]) if record and record["existing_ids"] else set()

    # ───────────────────────────────────────────────
    # CREATE NODES & RELATIONSHIPS
    # ───────────────────────────────────────────────
    def store_memory_graph(self, user_id: str, memories: List[FlattenedMemory]):
        """
        Store or update memory relationships in Neo4j.
        Each memory is connected to:
        - a User
        - its Tag
        - its mentioned Entities
        
        Accepts:
        - Flattened format: [memories with 'tag' and 'id' fields]
        
        Skips memories that already exist in the database.
        """
        if not memories:
            return

        # Get list of existing memory IDs to skip
        memory_ids = [m.id for m in memories]
        existing_ids = self.get_existing_memory_ids(memory_ids)
        
        # Filter out existing memories
        new_memories = [m for m in memories if m.id not in existing_ids]
        skipped_count = len(memories) - len(new_memories)
        
        if skipped_count > 0:
            print(f"⊘ Skipped {skipped_count} existing memories in Neo4j")

        if not new_memories:
            return

        with self.driver.session() as session:
            # Make sure user node exists once
            session.run("""
                MERGE (u:User {id: $user_id})
                SET u.name = $user_id
            """, user_id=user_id)

            # Then loop memories
            for m in new_memories:
                session.run("""
                    MATCH (u:User {id: $user_id})
                    MERGE (t:Tag {name: $tag})
                    MERGE (m:Memory {id: $memory_id})
                    SET m.id = $memory_id
                    MERGE (u)-[:HAS_MEMORY]->(m)
                    MERGE (m)-[:BELONGS_TO_TAG]->(t)
                """, 
                user_id=user_id,
                tag=m.tag,
                memory_id=m.id,)

                for e in m.entities:
                    if e:
                        session.run("""
                            MATCH (u:User {id: $user_id})
                            MATCH (m:Memory {id: $memory_id})
                            MERGE (ent:Entity {name: $entity})
                            MERGE (u)-[:HAS_ENTITY]->(ent)
                            MERGE (m)-[:MENTIONS]->(ent)
                        """, user_id=user_id, memory_id=m.id, entity=e)


    # ───────────────────────────────────────────────
    # FETCH RELATED TAGS/ENTITIES on MEMORY IDs
    # ───────────────────────────────────────────────
    def get_related_memory_ids(self, memory_ids: list, limit: int = 20):
        """
        Given a list of memory IDs, find related memory IDs via shared entities or tags.
        Used to expand search contextually.
        """
        if not memory_ids:
            return []

        related_ids = set()
        with self.driver.session() as session:
            # Find memories connected via shared entities
            ent_result = session.run("""
                MATCH (m:Memory)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(related:Memory)
                WHERE m.id IN $memory_ids AND related.id <> m.id
                RETURN DISTINCT related.id AS memory_id
                LIMIT $limit
            """, memory_ids=memory_ids, limit=limit)
            related_ids.update([r["memory_id"] for r in ent_result])

            # Find memories connected via shared tags
            tag_result = session.run("""
                MATCH (m:Memory)-[:BELONGS_TO_TAG]->(t:Tag)<-[:BELONGS_TO_TAG]-(related:Memory)
                WHERE m.id IN $memory_ids AND related.id <> m.id
                RETURN DISTINCT related.id AS memory_id
                LIMIT $limit
            """, memory_ids=memory_ids, limit=limit)
            related_ids.update([r["memory_id"] for r in tag_result])

        return list(related_ids)
    
    # ───────────────────────────────────────────────
    # DELETE ALL DATA FOR USER
    # ───────────────────────────────────────────────
    def delete_all(self, user_id: str) -> bool:
        """
        Delete all Neo4j data for a specific user_id.
        Removes:
        - All Memory nodes connected to the user
        - All relationships from the User node (HAS_MEMORY, HAS_ENTITY)
        - The User node itself
        - Orphaned Tags and Entities that are only connected to this user
        
        Args:
            user_id: The user ID to delete all data for
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # First, get count of memories to delete for logging
                count_result = session.run("""
                    MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                    RETURN COUNT(m) as memory_count
                """, user_id=user_id)
                record = count_result.single()
                memory_count = record["memory_count"] if record else 0
                
                # Delete all Memory nodes and their relationships
                # This will cascade delete relationships to Tags and Entities
                session.run("""
                    MATCH (u:User {id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                    DETACH DELETE m
                """, user_id=user_id)
                
                if memory_count > 0:
                    print(f"✓ Deleted {memory_count} Memory nodes for user: {user_id}")
                
                # Delete all HAS_ENTITY relationships from User
                entity_rel_result = session.run("""
                    MATCH (u:User {id: $user_id})-[r:HAS_ENTITY]->(e:Entity)
                    DELETE r
                    RETURN COUNT(r) as rel_count
                """, user_id=user_id)
                entity_rel_record = entity_rel_result.single()
                entity_rel_count = entity_rel_record["rel_count"] if entity_rel_record else 0
                
                if entity_rel_count > 0:
                    print(f"✓ Deleted {entity_rel_count} HAS_ENTITY relationships for user: {user_id}")
                
                # Delete orphaned Tags (Tags with no remaining Memory connections)
                orphaned_tags_result = session.run("""
                    MATCH (t:Tag)
                    WHERE NOT (t)<-[:BELONGS_TO_TAG]-(:Memory)
                    DELETE t
                    RETURN COUNT(t) as tag_count
                """)
                orphaned_tags_record = orphaned_tags_result.single()
                orphaned_tags_count = orphaned_tags_record["tag_count"] if orphaned_tags_record else 0
                
                if orphaned_tags_count > 0:
                    print(f"✓ Deleted {orphaned_tags_count} orphaned Tag nodes")
                
                # Delete orphaned Entities (Entities with no remaining Memory or User connections)
                orphaned_entities_result = session.run("""
                    MATCH (e:Entity)
                    WHERE NOT (e)<-[:MENTIONS]-(:Memory)
                       AND NOT (e)<-[:HAS_ENTITY]-(:User)
                    DELETE e
                    RETURN COUNT(e) as entity_count
                """)
                orphaned_entities_record = orphaned_entities_result.single()
                orphaned_entities_count = orphaned_entities_record["entity_count"] if orphaned_entities_record else 0
                
                if orphaned_entities_count > 0:
                    print(f"✓ Deleted {orphaned_entities_count} orphaned Entity nodes")
                
                # Finally, delete the User node itself
                user_delete_result = session.run("""
                    MATCH (u:User {id: $user_id})
                    DELETE u
                    RETURN COUNT(u) as user_count
                """, user_id=user_id)
                user_delete_record = user_delete_result.single()
                user_count = user_delete_record["user_count"] if user_delete_record else 0
                
                if user_count > 0:
                    print(f"✓ Deleted User node for user: {user_id}")
                
                print(f"✓ Successfully deleted all Neo4j data for user: {user_id}")
                return True
                
        except Exception as e:
            print(f"❌ Error deleting Neo4j data for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return True