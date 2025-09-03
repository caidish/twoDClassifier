#!/usr/bin/env python3
"""
Compact FastMCP Server for 2D Material Classification
Compatible with MCP clients via HTTP or stdio transport
"""

# IMPORTANT: Monkey patch MCP message size limit BEFORE any imports
# This must happen before FastMCP or MCP modules are imported
import sys

# Pre-load and patch the MCP module to increase message size limit
# This supports 16MB files (which become ~21MB when Base64 encoded)
try:
    import mcp.server.streamable_http
    mcp.server.streamable_http.MAXIMUM_MESSAGE_SIZE = 25 * 1024 * 1024  # 25MB
    print(f"Increased MCP message size limit to {25}MB for large file support", file=sys.stderr)
except ImportError:
    print("Warning: Could not import mcp.server.streamable_http for size limit patch", file=sys.stderr)

import base64
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastmcp import FastMCP

# Add src to path for imports  
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

from model_loader import load_graphene_model, list_available_models

# Initialize FastMCP server
mcp = FastMCP("2D Material Classifier")

# Server state
data_dir = current_dir / "data"
models_dir = current_dir / "models"
loaded_models: Dict[str, Any] = {}
prediction_history: List[Dict[str, Any]] = []

# Ensure data directory exists
data_dir.mkdir(exist_ok=True)


@mcp.tool()
def list_models() -> Dict[str, Any]:
    """Get list of available 2D material classification models"""
    try:
        models = list_available_models()
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def upload_image(image_data: str, filename: str) -> Dict[str, Any]:
    """Upload an image for 2D material classification
    
    Args:
        image_data: Base64 encoded image data
        filename: Original filename
    """
    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        
        # Keep original filename, ensure uniqueness
        target_path = data_dir / filename
        final_filename = filename
        
        if target_path.exists():
            # Add counter to make unique
            base, ext = os.path.splitext(filename)
            counter = 1
            while target_path.exists():
                final_filename = f"{base}_{counter}{ext}"
                target_path = data_dir / final_filename
                counter += 1
        
        # Save image with preserved filename
        with open(target_path, "wb") as f:
            f.write(image_bytes)
        
        return {
            "success": True,
            "filename": final_filename,
            "original_filename": filename,
            "size": len(image_bytes),
            "path": str(target_path)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Upload failed: {str(e)}"
        }


@mcp.tool()
def predict_flake_quality(model_name: str, image_filename: str) -> Dict[str, Any]:
    """Predict 2D material flake quality using specified model
    
    Args:
        model_name: Name of the model to use for prediction
        image_filename: Filename of the uploaded image
    """
    global loaded_models, prediction_history
    
    try:
        # Validate image exists
        image_path = data_dir / image_filename
        if not image_path.exists():
            return {
                "success": False,
                "error": f"Image not found: {image_filename}"
            }
        
        # Load model if not already loaded
        if model_name not in loaded_models:
            loaded_models[model_name] = load_graphene_model(model_name)
        
        classifier = loaded_models[model_name]
        
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
        prediction_history.append(result.copy())
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }


@mcp.tool()
def get_prediction_history(limit: int = 10) -> Dict[str, Any]:
    """Get history of previous predictions
    
    Args:
        limit: Maximum number of results to return (default: 10)
    """
    try:
        # Return most recent predictions
        recent_history = prediction_history[-limit:] if limit > 0 else prediction_history
        return {
            "success": True,
            "history": recent_history,
            "total_predictions": len(prediction_history)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get history: {str(e)}"
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastMCP Server for 2D Material Classification")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--transport", choices=["http", "stdio"], default="http", 
                       help="Transport type (default: http)")
    
    args = parser.parse_args()
    
    if args.transport == "http":
        print(f"Starting FastMCP server with HTTP transport on {args.host}:{args.port}")
        print(f"Server will be accessible at: http://{args.host}:{args.port}/mcp/")
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        print("Starting FastMCP server with stdio transport")
        mcp.run(transport="stdio")