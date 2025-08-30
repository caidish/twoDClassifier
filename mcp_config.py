#!/usr/bin/env python3
"""
MCP Server Configuration
Centralized configuration for the 2D Material Classification MCP Server
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

class MCPConfig:
    """Configuration class for MCP Server"""
    
    # Server settings
    DEFAULT_HOST = "0.0.0.0"
    DEFAULT_PORT = 8000
    
    # Protocol settings
    MCP_PROTOCOL_VERSION = "2025-06-18"
    
    # File handling
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    MAX_IMAGE_SIZE_MB = 50
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    # Model settings
    MODEL_CACHE_SIZE = 5  # Maximum number of models to keep in memory
    
    # Logging
    LOG_LEVEL = "INFO"
    
    # CORS settings
    CORS_ORIGINS = ["*"]  # In production, specify actual origins
    
    # History settings
    MAX_PREDICTION_HISTORY = 1000
    
    @classmethod
    def get_server_config(cls) -> Dict[str, Any]:
        """Get server configuration dictionary"""
        return {
            "host": os.getenv("MCP_HOST", cls.DEFAULT_HOST),
            "port": int(os.getenv("MCP_PORT", cls.DEFAULT_PORT)),
            "reload": os.getenv("MCP_RELOAD", "false").lower() == "true",
            "log_level": os.getenv("MCP_LOG_LEVEL", cls.LOG_LEVEL).lower()
        }
    
    @classmethod
    def get_capabilities(cls) -> Dict[str, Any]:
        """Get MCP server capabilities"""
        return {
            "protocol": "mcp",
            "version": cls.MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": True,
                "resources": False,
                "prompts": False
            },
            "server_info": {
                "name": "2D Material Classifier",
                "version": "1.0.0",
                "description": "MCP server for 2D material flake quality classification"
            }
        }
    
    @classmethod
    def validate_paths(cls) -> None:
        """Validate required paths exist"""
        DATA_DIR.mkdir(exist_ok=True)
        
        if not MODELS_DIR.exists():
            raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")
        
        if not SRC_DIR.exists():
            raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

# Environment-based configuration
PRODUCTION = os.getenv("MCP_ENVIRONMENT", "development") == "production"

if PRODUCTION:
    MCPConfig.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    MCPConfig.LOG_LEVEL = "WARNING"