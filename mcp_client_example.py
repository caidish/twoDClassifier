#!/usr/bin/env python3
"""
MCP Client Example
Demonstrates how to interact with the 2D Material Classification MCP Server
"""

import asyncio
import httpx
import base64
import json
from pathlib import Path

class MCPClient:
    """Simple MCP client for testing the 2D Material Classification server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def get_capabilities(self) -> dict:
        """Get server capabilities"""
        response = await self.client.get(f"{self.base_url}/.well-known/mcp-capabilities")
        response.raise_for_status()
        return response.json()
    
    async def upload_image(self, image_path: str) -> dict:
        """Upload an image to the server"""
        image_file = Path(image_path)
        
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read and encode image
        with open(image_file, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Upload image
        data = {
            "image_data": image_data,
            "filename": image_file.name
        }
        
        response = await self.client.post(
            f"{self.base_url}/mcp/tools/upload_image",
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> dict:
        """Get list of available models"""
        response = await self.client.get(f"{self.base_url}/mcp/tools/list_models")
        response.raise_for_status()
        return response.json()
    
    async def predict_flake_quality(self, model_name: str, image_filename: str) -> dict:
        """Predict flake quality"""
        data = {
            "model_name": model_name,
            "image_filename": image_filename
        }
        
        response = await self.client.post(
            f"{self.base_url}/mcp/tools/predict_flake_quality",
            data=data
        )
        response.raise_for_status()
        return response.json()
    
    async def get_prediction_history(self, limit: int = 10) -> dict:
        """Get prediction history"""
        params = {"limit": limit}
        response = await self.client.get(
            f"{self.base_url}/mcp/tools/get_prediction_history",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    async def health_check(self) -> dict:
        """Check server health"""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

async def demo_workflow():
    """Demonstrate complete MCP client workflow"""
    print("=== 2D Material Classification MCP Client Demo ===\n")
    
    async with MCPClient() as client:
        try:
            # Check server health
            print("1. Checking server health...")
            health = await client.health_check()
            print(f"   Server status: {health['status']}")
            print(f"   Loaded models: {health.get('loaded_models', [])}")
            print()
            
            # Get capabilities
            print("2. Getting server capabilities...")
            capabilities = await client.get_capabilities()
            tools = capabilities.get("tools", [])
            print(f"   Available tools: {[tool['name'] for tool in tools]}")
            print()
            
            # List available models
            print("3. Listing available models...")
            models_response = await client.list_models()
            models = models_response["models"]
            print(f"   Found {len(models)} models: {models}")
            print()
            
            # Upload test image (assuming Data1.jpg exists)
            test_image = "data/Data1.jpg"
            if Path(test_image).exists():
                print(f"4. Uploading test image: {test_image}")
                upload_result = await client.upload_image(test_image)
                uploaded_filename = upload_result["filename"]
                print(f"   Image uploaded as: {uploaded_filename}")
                print(f"   File size: {upload_result['size']} bytes")
                print()
                
                # Make prediction with first available model
                if models:
                    model_name = models[0]
                    print(f"5. Making prediction with model: {model_name}")
                    prediction = await client.predict_flake_quality(model_name, uploaded_filename)
                    print(f"   Prediction: {prediction['quality']}")
                    print(f"   Confidence - Good: {prediction['confidence']['good_quality']:.1%}")
                    print(f"   Confidence - Bad: {prediction['confidence']['bad_quality']:.1%}")
                    print()
                
                # Get prediction history
                print("6. Getting prediction history...")
                history = await client.get_prediction_history(limit=5)
                print(f"   Total predictions: {history['total_predictions']}")
                for i, pred in enumerate(history['history'][-3:], 1):
                    print(f"   {i}. {pred['model_name']}: {pred['quality']} ({pred['timestamp']})")
                print()
            
            else:
                print(f"4. Test image not found: {test_image}")
                print("   Skipping upload and prediction demo")
                print()
            
            print("Demo completed successfully!")
            
        except Exception as e:
            print(f"Demo failed: {str(e)}")

async def test_single_prediction(image_path: str, model_name: str = None):
    """Test single prediction workflow"""
    async with MCPClient() as client:
        try:
            # Get available models if none specified
            if not model_name:
                models_response = await client.list_models()
                models = models_response["models"]
                if not models:
                    print("No models available")
                    return
                model_name = models[0]
                print(f"Using model: {model_name}")
            
            # Upload image
            print(f"Uploading image: {image_path}")
            upload_result = await client.upload_image(image_path)
            uploaded_filename = upload_result["filename"]
            
            # Make prediction
            print("Making prediction...")
            prediction = await client.predict_flake_quality(model_name, uploaded_filename)
            
            print(f"\nResults:")
            print(f"Model: {prediction['model_name']}")
            print(f"Image: {prediction['image_filename']}")
            print(f"Quality: {prediction['quality']}")
            print(f"Good Quality Confidence: {prediction['confidence']['good_quality']:.1%}")
            print(f"Bad Quality Confidence: {prediction['confidence']['bad_quality']:.1%}")
            
        except Exception as e:
            print(f"Prediction failed: {str(e)}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="2D Material Classification MCP Client")
    parser.add_argument("--demo", action="store_true", help="Run complete demo workflow")
    parser.add_argument("--image", help="Image path for single prediction")
    parser.add_argument("--model", help="Model name for prediction")
    parser.add_argument("--server", default="http://localhost:8000", help="MCP server URL")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(demo_workflow())
    elif args.image:
        asyncio.run(test_single_prediction(args.image, args.model))
    else:
        print("Use --demo for full workflow or --image <path> for single prediction")
        parser.print_help()

if __name__ == "__main__":
    main()