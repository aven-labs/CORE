"""
Chat processing service for the Aven Speech API
"""
import logging
import json
from flask import request, jsonify, Response, stream_with_context
from brain.src.orchestration.orchestrator import AgentOrchestrator
from brain.src.memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)



class ChatService:
    """Service class for chat processing operations"""

    def process_chat_stream(self, user_id: str, access_token: str = None):
        """Process chat messages with streaming response"""
        try:
            data = request.get_json()
            if not data:
                logger.warning(f"Empty request data from user: {user_id}")
                return jsonify({"error": "No data provided"}), 400
            
            if 'message' not in data:
                logger.warning(f"Missing message field from user: {user_id}")
                return jsonify({"error": "No message provided"}), 400
            
            message = data['message']
            device_id = data.get('device_id', 'unknown')
            
            if not message or not message.strip():
                logger.warning(f"Empty message from user: {user_id}")
                return jsonify({"error": "Message cannot be empty"}), 400
            
            logger.info(f"Processing chat stream for user: {user_id}, message length: {len(message)}")
            
            try:
                orchestrator = AgentOrchestrator(user_id=user_id, access_token=access_token, device_id=device_id)
            except Exception as e:
                logger.error(f"Failed to initialize orchestrator for user {user_id}: {e}", exc_info=True)
                return jsonify({"error": "Failed to initialize chat service"}), 500
            
            def generate():
                """Generator function that yields streaming chunks"""
                try:
                    chunk_count = 0
                    for chunk in orchestrator.start_chat_stream(message):
                        chunk_count += 1
                        # Format as Server-Sent Events (SSE)
                        chunk_data = json.dumps({
                            "chunk": chunk,
                            "done": False
                        })
                        yield f"data: {chunk_data}\n\n"

                    logger.debug(f"Stream completed for user {user_id}, total chunks: {chunk_count}")
                    
                    # Send final done message
                    done_data = json.dumps({
                        "chunk": "",
                        "done": True
                    })
                    yield f"data: {done_data}\n\n"
                except Exception as e:
                    logger.error(f"Error in chat stream generator for user {user_id}: {e}", exc_info=True)
                    error_data = json.dumps({
                        "error": "Streaming error occurred",
                        "done": True
                    })
                    yield f"data: {error_data}\n\n"
            
            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive'
                }
            )
        except Exception as e:
            logger.error(f"Chat stream processing failed for user {user_id}: {e}", exc_info=True)
            return jsonify({"error": "Streaming chat failed"}), 500
    
    def clear_history(self, user_id: str):
        """Clear all chat history and memory data for a user"""
        try:
            logger.info(f"Clearing history for user: {user_id}")
            memory_manager = MemoryManager(user_id=user_id)
            success = memory_manager.delete_all()
            
            if success:
                logger.info(f"Successfully cleared history for user: {user_id}")
                return jsonify({
                    "success": True,
                    "message": "All chat history and memory data cleared successfully"
                }), 200
            else:
                logger.warning(f"Partial history clear for user: {user_id}")
                return jsonify({
                    "success": False,
                    "message": "Some data may not have been cleared completely"
                }), 207  # 207 Multi-Status
            
        except Exception as e:
            logger.error(f"Failed to clear history for user {user_id}: {e}", exc_info=True)
            return jsonify({
                "error": "Failed to clear history"
            }), 500
