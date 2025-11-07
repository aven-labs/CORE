"""
Memory Agent - Extracts and organizes memories from conversations
"""
from datetime import datetime
import json
from typing import List, Dict, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from brain_core.config import Config
from memory.ltm_core.tag_manager import TagManager
from pydantic import BaseModel, Field
from typing import Dict, List
from uuid import uuid4

# Output is like this:
#
# {
#   "emotion": [
#     {
#       "id": "550e8400-e29b-41d4-a716-446655440000",
#       "summary": "User values work-life balance",
#       "importance": 0.8,
#       "confidence": 0.9,
#       "entities": ["work", "balance"],
#       "timestamp": "2024-01-01T00:00:00Z"
#     }
#   ],
#   "habit": [
#     {
#       "id": "550e8400-e29b-41d4-a716-446655440001",
#       "summary": "User regularly takes breaks",
#       "importance": 0.7,
#       "confidence": 0.85,
#       "entities": ["breaks", "routine"],
#       "timestamp": "2024-01-01T00:00:00Z"
#     }
#   ]
# }

class MemoryItem(BaseModel):
    """Represents one atomic behavioral or factual memory."""
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the memory (UUID)")
    summary: str = Field(..., description="A single, normalized behavioral or emotional insight about the user.")
    importance: float = Field(..., ge=0.0, le=1.0, description="How central or defining this memory is to the user's identity.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence that this insight is true or well-supported by context.")
    entities: List[str] = Field(default_factory=list, description="Key entities or concepts related to the memory.")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO-8601 string representing when the memory was extracted or last updated.")
TaggedMemories = Dict[str, List[MemoryItem]]
    


class MemoryAgent(AssistantAgent):
    """Analyzes conversations and extracts tagged memories."""

    def __init__(self, user_id: str = "default", **kwargs):
        """
        Initialize MemoryAgent.

        Args:
            user_id: User ID for memory isolation
            **kwargs: Additional arguments to pass to AssistantAgent
        """
        self.user_id = user_id
        self.tag_manager = TagManager(user_id=user_id)
        


        super().__init__(
            name="memory_organizer",
            description="Extracts and organizes memories from conversations",
            model_client=Config.model_client(index=0, json_output=True,  structured_output=True),
            system_message=self._get_system_message(),
            **kwargs
        )

    def _get_system_message(self) -> str:
        """Generate system message for memory extraction."""
        existing_tags = self.tag_manager.get_tags()
        tags_str = ", ".join(existing_tags) if existing_tags else "None yet"

        return f"""
        You are a BEHAVIORAL MEMORY AGENT analyzing human conversation as an impartial observer.
        Purpose:
        - Observe the conversation from a third-person perspective.
        - Filter out small talk, filler, or irrelevant text.
        - Identify *behavioral truths* about the user — their emotions, values, routines, habits, work style, and social tendencies.
        - Capture what the conversation *reveals about who the user is* rather than what was literally said.

        Guidelines:
        - Write concise and objective behavioral insights.
        - Avoid quoting the conversation directly.
        - Focus on personality, emotions, and actions that seem consistent or repeated.
        - Infer confidently but not speculatively — only include high-probability traits or behaviors.
        - Keep the voice neutral and observational (like a behavioral scientist).
        - Exclude commentary, reasoning, or references to the conversation itself.

        JSON Output Format (REQUIRED):
        Return a JSON object where each key is a tag and each value is an array of memory objects.
        Example:
        {{
          "emotion": [
            {{"id": "550e8400-e29b-41d4-a716-446655440000", "summary": "User values work-life balance", "importance": 0.8, "confidence": 0.9, "entities": ["work", "balance"], "timestamp": "2024-01-01T00:00:00Z"}}
          ],
          "habit": [
            {{"id": "550e8400-e29b-41d4-a716-446655440001", "summary": "User regularly takes breaks", "importance": 0.7, "confidence": 0.85, "entities": ["breaks", "routine"], "timestamp": "2024-01-01T00:00:00Z"}}
          ]
        }}

        Each memory object must have:
        - id: Unique UUID string identifier (string, e.g., "550e8400-e29b-41d4-a716-446655440000")
        - summary: A single-sentence normalized insight (string)
        - importance: How central to user identity (number 0.0-0.3)
        - confidence: How well-supported by conversation (number 0.0-1.0)
        - entities: List of key concepts (array of strings)
        - timestamp: ISO-8601 format (string) - will be auto-generated if not provided

        Current known tags for user {self.user_id}: {tags_str}

        IMPORTANT: Return ONLY valid JSON with no explanations or extra text.
        """

    async def extract_memories(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract tagged memories from conversation history.

        Args:
            conversation_history: List of message dictionaries

        Returns:
            Dictionary with tags as keys and list of memories as values
        """
        try:
            # Format conversation for analysis
            conversation_text = self._format_conversation(conversation_history)
            
            # Create extraction prompt
            prompt = (
                "Analyze this conversation and extract memories grouped by tags.\n"
                "Return ONLY valid JSON without extra text.\n\n"
                f"CONVERSATION:\n{conversation_text}\n\n"
                "Extract memories now:"
            )
            response = await self.on_messages(
                messages=[TextMessage(content=prompt, source="user", created_at=datetime.now())],
                cancellation_token=CancellationToken()
            )
            # Parse JSON response
            memories = self._parse_response(response.chat_message.content)

            if memories:
                print(f"✓ Extracted {sum(len(v) for v in memories.values())} memories")

            return memories

        except Exception as e:
            print(f"❌ Error extracting memories: {e}")
            return {}

    def _format_conversation(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Format conversation history into readable text."""
        formatted = []
        for msg in conversation_history:
            if isinstance(msg, dict):
                role = msg.get("name", msg.get("role", "user"))
                content = msg.get("content", "")
                if content:
                    formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _parse_response(self, response: str) -> Dict[str, List[Dict[str, Any]]]:
        """Parse LLM response to extract JSON memories."""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                # Extract memories field if it exists, otherwise treat as direct memories dict
                if isinstance(data, dict):
                    if "memories" in data:
                        memories = data["memories"]
                    else:
                        memories = data
                    
                    # Add timestamp and id to each memory if missing
                    current_timestamp = datetime.now().isoformat()
                    for tag, memory_list in memories.items():
                        if isinstance(memory_list, list):
                            for memory in memory_list:
                                if isinstance(memory, dict):
                                    memory["id"] = str(uuid4())
                                    memory["timestamp"] = current_timestamp
                    
                    return memories

        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing JSON: {e}")

        return {}
