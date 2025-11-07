"""
Supabase authentication service for token verification and user details
"""
import logging
from functools import wraps
from flask import request, jsonify
from brain_core.sup_extractor import supabase_service

logger = logging.getLogger(__name__)


class AuthService:
    """Service class for Supabase authentication operations"""
    
    def verify_token(self, token: str) -> dict:
        """
        Verify JWT token with Supabase and return user details
        
        Args:
            token: JWT token string
            
        Returns:
            dict: User details if token is valid, None otherwise
        """
        return supabase_service.verify_token(token)
    
    def extract_token_from_header(self) -> str:
        """
        Extract token from Authorization header without verification.
        
        Returns:
            str: Token string if found, None otherwise
        """
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return None
        
        # Extract token (supports "Bearer <token>" or just "<token>")
        if auth_header.startswith('Bearer '):
            token = auth_header.split('Bearer ')[1]
        else:
            token = auth_header
        
        return token if token else None
    
    def get_user_from_request(self) -> dict:
        """
        Extract and verify token from request headers
        
        Returns:
            dict: User details if token is valid, None otherwise
        """
        token = self.extract_token_from_header()
        
        if not token:
            return None
        
        return self.verify_token(token)


# Global auth service instance
auth_service = AuthService()


def require_auth(f):
    """
    Decorator to require authentication for a route
    
    Usage:
        @app.route('/protected')
        @require_auth
        def protected_route():
            user = request.user  # Access verified user details
            return jsonify({"message": f"Hello {user['email']}"})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            user_data = auth_service.get_user_from_request()
            
            if not user_data:
                logger.warning(f"Unauthorized access attempt to {request.path}")
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Invalid or missing authentication token"
                }), 401
            
            # Extract and store token for use in route handlers
            token = auth_service.extract_token_from_header()
            
            # Attach user data and token to request for use in route handlers
            request.user = user_data
            request.user_id = user_data['id']
            request.access_token = token
            
            logger.debug(f"Authenticated user {user_data['id']} accessing {request.path}")
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in authentication decorator: {e}", exc_info=True)
            return jsonify({
                "error": "Internal server error",
                "message": "Authentication check failed"
            }), 500
    
    return decorated_function

