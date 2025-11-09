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


def _is_connection_open(ws) -> bool:
    """Check if WebSocket connection is open (works for both client and server)."""
    if ws is None:
        return False
    # For client connections, close_code is None when open
    if hasattr(ws, 'close_code'):
        return ws.close_code is None
    # For server connections, check closed attribute
    if hasattr(ws, 'closed'):
        return not ws.closed
    return True


async def send_intent(
    device_id: str,
    action: str,
    data: Optional[Dict[str, Any]] = None,
    wait_for_response: bool = False,
    timeout: float = 10.0,
    auth_token: Optional[str] = None,
    retry_on_not_connected: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Send an intent/action message to a target device via the relay server.
    
    Args:
        device_id: Target device ID to send the message to
        action: Action/intent type (e.g., "search_file", "execute_command", etc.)
        data: Optional additional data payload to include in the message
        wait_for_response: If True, wait for and return the response from the device
        timeout: Timeout in seconds when waiting for response (default: 10.0)
        auth_token: Optional Supabase authentication token for relay server auth
        retry_on_not_connected: If True, retry if device is not connected (default: True)
        max_retries: Maximum number of retries if device not connected (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
    
    Returns:
        Optional[Dict]: Response from the device if wait_for_response=True, None otherwise
        
    Raises:
        ConnectionError: If relay server is unavailable or device is not connected
        TimeoutError: If response timeout exceeded
    """
    relay_url = Config.RELAY_SERVER_URL
    sender_id = "system_sender"
    
    logger.debug(f"Sending intent to device {device_id} with action '{action}' via relay: {relay_url}")
    
    ws = None
    
    for attempt in range(max_retries + 1):
        try:
            # Connect to relay server
            ws = await websockets.connect(
                relay_url,
                ping_interval=None,
                close_timeout=5.0
            )
            
            try:
                # Register as sender
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
                
                logger.debug(f"Successfully connected to relay server as {sender_id}")
                
                # Prepare message
                request_id = str(uuid.uuid4())
                message = {
                    "target_id": device_id,
                    "action": action,
                    "request_id": request_id,
                    "type": "intent",
                    "data": data or {}
                }
                
                # Send message
                await ws.send(json.dumps(message))
                logger.debug(f"Sent intent message to device {device_id}, request_id: {request_id}, wait_for_response: {wait_for_response}")
                
                # If not waiting for response, return immediately
                if not wait_for_response:
                    return None
                
                # Wait for response with timeout
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    remaining = timeout - elapsed
                    
                    if remaining <= 0:
                        logger.warning(f"No response from device {device_id} within {timeout}s")
                        raise TimeoutError(f"No response from {device_id} within {timeout}s")
                    
                    try:
                        # Wait for message with remaining timeout
                        response_msg = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                        response_data = json.loads(response_msg)
                        
                        # Handle pong messages (server ping responses)
                        if response_data.get("type") == "pong" or response_data.get("action") == "pong":
                            logger.debug("Received pong from server")
                            continue
                        
                        # Ignore server control messages
                        if response_data.get("type") in ["connected", "server_status"]:
                            logger.debug(f"Ignoring server message: {response_data.get('type')}")
                            continue
                        
                        # Handle error messages
                        if response_data.get("type") == "error":
                            error_msg = response_data.get("error", "Unknown error")
                            error_code = response_data.get("code", 500)
                            
                            # If device not connected (404), this is retryable
                            if error_code == 404 and "not connected" in error_msg.lower():
                                if retry_on_not_connected and attempt < max_retries:
                                    logger.warning(f"Device not connected (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                                    raise ConnectionError(f"Device not connected: {error_msg}")
                                else:
                                    raise ConnectionError(f"Device {device_id} is not connected: {error_msg}")
                            
                            # Other errors - don't retry
                            logger.error(f"Received error from relay server: {error_msg} (code: {error_code})")
                            raise ConnectionError(f"Relay server error: {error_msg}")
                        
                        # Check if this is the response to our request
                        if (response_data.get("type") == "response" and 
                            response_data.get("request_id") == request_id):
                            logger.debug(f"Received response from device {device_id}")
                            return response_data.get("data")
                        
                        logger.debug(f"Received unexpected message type: {response_data.get('type')}")
                        
                    except asyncio.TimeoutError:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed >= timeout:
                            raise TimeoutError(f"No response from {device_id} within {timeout}s")
                        continue
                
            finally:
                # Close connection when done
                if _is_connection_open(ws):
                    try:
                        await ws.close()
                    except Exception as e:
                        logger.debug(f"Error closing connection: {e}")
            
            # Success - exit retry loop
            return None
                
        except ConnectionError as e:
            # Retry on "not connected" errors
            if "not connected" in str(e).lower() and retry_on_not_connected and attempt < max_retries:
                logger.warning(f"Device not connected (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                continue
            # Other connection errors or exhausted retries - raise
            raise
            
        except (websockets.exceptions.ConnectionClosed, websockets.exceptions.InvalidStatusCode) as e:
            # Network/protocol errors - retry if we have attempts left
            if attempt < max_retries:
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries + 1}), retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                continue
            # Exhausted retries
            raise ConnectionError(f"Failed to connect to relay server: {e}")
            
        except TimeoutError:
            # Response timeout - don't retry
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error sending intent to {device_id}: {e}", exc_info=True)
            raise ConnectionError(f"Failed to send intent: {str(e)}")
            
        finally:
            # Ensure connection is closed on any exception
            if ws and _is_connection_open(ws):
                try:
                    await ws.close()
                except Exception:
                    pass
    
    # Should not reach here, but failsafe
    raise ConnectionError(f"Failed to send intent to device {device_id} after {max_retries + 1} attempts")