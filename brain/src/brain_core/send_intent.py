"""
Function to send intent/action messages to devices via WebSocket relay server.
"""
import asyncio
import logging
import websockets
import json
import uuid
from typing import Optional, Dict, Any
from brain_core.config import Config

logger = logging.getLogger(__name__)


async def send_intent(device_id: str, action: str, data: Optional[Dict[str, Any]] = None, 
                     wait_for_response: bool = False, timeout: float = 10.0,
                     auth_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Send an intent/action message to a target device via the relay server.
    
    Args:
        device_id: Target device ID to send the message to
        action: Action/intent type (e.g., "search_file", "execute_command", etc.)
        data: Optional additional data payload to include in the message
        wait_for_response: If True, wait for and return the response from the device
        timeout: Timeout in seconds when waiting for response (default: 10.0)
        auth_token: Optional Supabase authentication token for relay server auth
    
    Returns:
        Optional[Dict]: Response from the device if wait_for_response=True, None otherwise
        
    Example:
        >>> # Send action without waiting for response
        >>> await send_intent("device_123", "search_file", {"query": "invoice"})
        
        >>> # Send action with authentication
        >>> response = await send_intent(
        ...     "device_123", 
        ...     "execute_command", 
        ...     {"command": "ls -la"},
        ...     wait_for_response=True,
        ...     auth_token="your_supabase_token"
        ... )
    """
    relay_url = Config.RELAY_SERVER_URL
    sender_id = "system_sender"  # ID for this sender
    
    logger.debug(f"Sending intent to device {device_id} with action '{action}' via relay: {relay_url}")
    
    try:
        async with websockets.connect(relay_url) as ws:
            # Register as sender with optional auth token
            registration = {"device_id": sender_id}
            if auth_token:
                registration["token"] = auth_token
            await ws.send(json.dumps(registration))
            
            # Wait for connection confirmation
            confirmation = await asyncio.wait_for(ws.recv(), timeout=5.0)
            conf_data = json.loads(confirmation)
            if conf_data.get("type") != "connected":
                logger.error(f"Failed to connect to relay server: {conf_data}")
                raise ConnectionError("Failed to connect to relay server")
            
            # Prepare message
            request_id = str(uuid.uuid4())
            message = {
                "device_id": sender_id,
                "target_id": device_id,
                "action": action,
                "request_id": request_id,
                "type": "intent",
                "data": data or {}
            }
            
            # Send message
            await ws.send(json.dumps(message))
            logger.debug(f"Sent intent message to device {device_id}, waiting for response: {wait_for_response}")
            
            # Wait for response if requested
            if wait_for_response:
                try:
                    while True:
                        response_msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        response_data = json.loads(response_msg)
                        
                        # Check if this is the response to our request
                        if (response_data.get("type") == "response" and 
                            response_data.get("request_id") == request_id):
                            logger.debug(f"Received response from device {device_id}")
                            return response_data.get("data")
                        
                        # Ignore other messages (like server_status)
                except asyncio.TimeoutError:
                    logger.warning(f"No response from device {device_id} within {timeout}s")
                    raise TimeoutError(f"No response from {device_id} within {timeout}s")
            
            return None
            
    except websockets.exceptions.ConnectionClosed:
        logger.error(f"Connection to relay server closed while sending intent to device {device_id}")
        raise ConnectionError("Connection to relay server closed")
    except (TimeoutError, ConnectionError) as e:
        # Re-raise these as-is
        raise
    except Exception as e:
        logger.error(f"Failed to send intent to device {device_id}: {e}", exc_info=True)
        raise ConnectionError(f"Failed to send intent: {str(e)}")
