import os
import sys
import warnings

# Set up paths before importing any src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
brain_src_path = os.path.join(current_dir, 'src')
brain_src_path = os.path.abspath(brain_src_path)
if brain_src_path not in sys.path:
    sys.path.insert(1, brain_src_path)

from src.brain_core.config import Config
from src.orchestration.orchestrator import AgentOrchestrator


warnings.filterwarnings(
    "ignore",
    message="Resolved model mismatch",
    category=UserWarning,
    module="autogen_agentchat.agents._assistant_agent"
)


class SimpleChatApp:
    """Simple chat application"""

    def __init__(self):
        self.orchestrator = None
        self.user_id = "dhruv"
        self._initialize()

    def _initialize(self):
        """Initialize the application"""
        try:
            Config.validate()
            print("✓ Configuration validated")

            self.orchestrator = AgentOrchestrator(user_id=self.user_id)
            print("✓ Agent orchestrator initialized")

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            sys.exit(1)

    def chat(self, message: str):
        """Run a simple chat conversation"""
        self.orchestrator.start_chat(message)


def main():
    app = SimpleChatApp()

    while True:
        try:
            message = input("You: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                print("Returning to menu...\n")
                break
            if message:
                app.chat(message)

        except KeyboardInterrupt:
            print("\n\nReturning to menu...\n")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
