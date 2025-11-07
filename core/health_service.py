"""
Health check service for the Aven Speech API
"""
from flask import jsonify


def health_check():
    """Health check endpoint handler"""
    return jsonify({"status": "healthy", "service": "Aven Speech API"})
