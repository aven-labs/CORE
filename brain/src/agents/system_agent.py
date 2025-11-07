from typing import List, Dict, Any, Optional
from autogen_agentchat.agents import AssistantAgent
from brain_core.config import Config


class SystemAgent(AssistantAgent):
    """System Agent that handles actual system-level executions when triggered."""

    def __init__(self, name: str = "system_agent", tools: Optional[List[Dict[str, Any]]] = None, **kwargs):
        """
        Initialize SystemAgent.

        Args:
            name: Agent name
            tools: List of available tools in format [{"name": "", "id": "", "local": True|False}]
            **kwargs: Additional arguments to pass to AssistantAgent
        """
        self.tools = tools or []
        super().__init__(
            name=name,
            description="Executes real system-level commands when instructed by other agents.",
            model_client=Config.model_client(index=0, json_output=True),
            system_message=self._get_system_message(),
            **kwargs
        )

    def _get_system_message(self) -> str:
        """Define minimal, execution-focused behavior for the System Agent."""
        import json
        tools_json = json.dumps(self.tools, indent=2) if self.tools else "[]"

        return f"""You are the System Agent. Your role is to execute system-level actions based on user instructions.

Available Tools:
{tools_json}

CRITICAL RESPONSE FORMAT:
You MUST respond ONLY with a JSON object in this exact format:
{{ "action": "", "id": "" }}

Rules:
- "action": what needs to be done with clear request.
- "id": The tool ID to use (must match one of the available tool IDs)
- Return ONLY the JSON object, nothing else
- No additional text, explanations, or formatting
- If no matching action is found, return: {{ "action": "", "id": "" }}

Examples:
- User: "search for files"
  Response: {{ "action": "open my last christmas presentation", "id": "tool_123" }}

- User: "send email"
  Response: {{ "action": "send an email to john@example.com containing the subject 'Hello' and the body 'How are you?'", "id": "tool_456" }}

Remember: Return ONLY the JSON object in the format {{ "action": "", "id": "" }}"""
