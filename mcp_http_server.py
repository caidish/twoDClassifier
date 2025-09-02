#!/usr/bin/env python3
"""
MCP Server for 2D Material Classification
Enables remote language models to upload images and classify 2D materials
"""

import asyncio
import json
import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import base64
from io import BytesIO

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add src to path for imports
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

from model_loader import load_graphene_model, list_available_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server for 2D Material Classification"""
    
    def __init__(self):
        self.app = FastAPI(title="2D Material Classifier MCP Server", version="1.0.0")
        self.data_dir = current_dir / "data"
        self.models_dir = current_dir / "models"
        self.loaded_models: Dict[str, Any] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        
        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_mcp_routes()
        self._setup_tool_routes()
    
    def _setup_mcp_routes(self):
        """Setup MCP protocol routes"""
        
        @self.app.get("/.well-known/mcp-capabilities")
        async def get_capabilities():
            """Return MCP server capabilities"""
            return {
                "protocol": "mcp",
                "version": "2025-06-18",
                "capabilities": {
                    "tools": True,
                    "resources": False,
                    "prompts": False
                },
                "tools": [
                    {
                        "name": "upload_image",
                        "description": "Upload an image for 2D material classification",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "image_data": {
                                    "type": "string",
                                    "description": "Base64 encoded image data"
                                },
                                "filename": {
                                    "type": "string",
                                    "description": "Original filename"
                                }
                            },
                            "required": ["image_data", "filename"]
                        }
                    },
                    {
                        "name": "list_models",
                        "description": "Get list of available 2D material classification models",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    },
                    {
                        "name": "predict_flake_quality",
                        "description": "Predict 2D material flake quality using specified model",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "model_name": {
                                    "type": "string",
                                    "description": "Name of the model to use for prediction"
                                },
                                "image_filename": {
                                    "type": "string",
                                    "description": "Filename of the uploaded image"
                                }
                            },
                            "required": ["model_name", "image_filename"]
                        }
                    },
                    {
                        "name": "get_prediction_history",
                        "description": "Get history of previous predictions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "limit": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return",
                                    "default": 10
                                }
                            }
                        }
                    }
                ]
            }
    
    def _setup_tool_routes(self):
        """Setup MCP tool execution routes"""
        
        @self.app.post("/mcp/tools/upload_image")
        async def upload_image(image_data: str = Form(...), filename: str = Form(...)):
            """Upload and save image to data directory"""
            try:
                # Decode base64 image data
                image_bytes = base64.b64decode(image_data)
                
                # Generate unique filename
                file_ext = Path(filename).suffix
                unique_filename = f"{uuid.uuid4()}{file_ext}"
                file_path = self.data_dir / unique_filename
                
                # Save image
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                
                logger.info(f"Image uploaded: {unique_filename}")
                
                return {
                    "success": True,
                    "filename": unique_filename,
                    "original_filename": filename,
                    "size": len(image_bytes),
                    "path": str(file_path)
                }
                
            except Exception as e:
                logger.error(f"Image upload failed: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")
        
        @self.app.get("/mcp/tools/list_models")
        async def list_models_endpoint():
            """Get list of available models"""
            try:
                models = list_available_models()
                return {
                    "success": True,
                    "models": models,
                    "count": len(models)
                }
            except Exception as e:
                logger.error(f"Failed to list models: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
        
        @self.app.post("/mcp/tools/predict_flake_quality")
        async def predict_flake_quality(model_name: str = Form(...), image_filename: str = Form(...)):
            """Predict flake quality using specified model and image"""
            try:
                # Validate image exists
                image_path = self.data_dir / image_filename
                if not image_path.exists():
                    raise HTTPException(status_code=404, detail=f"Image not found: {image_filename}")
                
                # Load model if not already loaded
                if model_name not in self.loaded_models:
                    logger.info(f"Loading model: {model_name}")
                    self.loaded_models[model_name] = load_graphene_model(model_name)
                
                classifier = self.loaded_models[model_name]
                
                # Make prediction
                prediction = classifier.predict_single_image(str(image_path))
                probabilities = classifier.predict_single_image(str(image_path), return_probabilities=True)
                
                # Prepare result
                result = {
                    "success": True,
                    "model_name": model_name,
                    "image_filename": image_filename,
                    "prediction": int(prediction),
                    "quality": "Good Quality" if prediction == 1 else "Bad Quality",
                    "confidence": {
                        "bad_quality": float(probabilities[0][0]),
                        "good_quality": float(probabilities[0][1])
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add to history
                self.prediction_history.append(result.copy())
                
                logger.info(f"Prediction completed: {model_name} on {image_filename} -> {result['quality']}")
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.get("/mcp/tools/get_prediction_history")
        async def get_prediction_history(limit: int = 10):
            """Get prediction history"""
            try:
                # Return most recent predictions
                recent_history = self.prediction_history[-limit:] if limit > 0 else self.prediction_history
                return {
                    "success": True,
                    "history": recent_history,
                    "total_predictions": len(self.prediction_history)
                }
            except Exception as e:
                logger.error(f"Failed to get history: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "loaded_models": list(self.loaded_models.keys()),
                "data_directory": str(self.data_dir),
                "models_directory": str(self.models_dir)
            }

def create_server() -> MCPServer:
    """Create and configure MCP server instance"""
    return MCPServer()

def main():
    """Run MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="2D Material Classifier MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Create server instance
    server = create_server()
    
    logger.info(f"Starting 2D Material Classifier MCP Server on {args.host}:{args.port}")
    
    # Run server
    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()