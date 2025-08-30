#!/usr/bin/env python3
"""Simple MCP server test without async"""

import requests
import base64
import json
from pathlib import Path

def test_mcp_server():
    base_url = "http://127.0.0.1:8001"
    
    print("=== Simple MCP Server Test ===\n")
    
    # Test health
    print("1. Testing health...")
    response = requests.get(f"{base_url}/health")
    if response.status_code == 200:
        print(f"   ✓ Server healthy: {response.json()['status']}")
    else:
        print(f"   ✗ Health check failed: {response.status_code}")
        return
    
    # Test capabilities
    print("2. Testing capabilities...")
    response = requests.get(f"{base_url}/.well-known/mcp-capabilities")
    if response.status_code == 200:
        capabilities = response.json()
        print(f"   ✓ Protocol: {capabilities['protocol']} v{capabilities['version']}")
        print(f"   ✓ Tools: {len(capabilities['tools'])} available")
    else:
        print(f"   ✗ Capabilities failed: {response.status_code}")
        return
    
    # Test models list
    print("3. Testing models list...")
    response = requests.get(f"{base_url}/mcp/tools/list_models")
    if response.status_code == 200:
        models_data = response.json()
        models = models_data['models']
        print(f"   ✓ Found {len(models)} models: {models[:3]}...")
    else:
        print(f"   ✗ Models list failed: {response.status_code}")
        return
    
    # Test image upload
    test_image_path = Path("data/Data1.jpg")
    if test_image_path.exists():
        print(f"4. Testing image upload: {test_image_path}")
        
        # Read and encode image
        with open(test_image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Upload image
        data = {
            "image_data": image_data,
            "filename": test_image_path.name
        }
        
        response = requests.post(f"{base_url}/mcp/tools/upload_image", data=data)
        if response.status_code == 200:
            upload_result = response.json()
            uploaded_filename = upload_result['filename']
            print(f"   ✓ Image uploaded as: {uploaded_filename}")
            print(f"   ✓ File size: {upload_result['size']} bytes")
            
            # Test prediction
            print("5. Testing prediction...")
            model_name = models[0] if models else "graphene_1"
            
            pred_data = {
                "model_name": model_name,
                "image_filename": uploaded_filename
            }
            
            response = requests.post(f"{base_url}/mcp/tools/predict_flake_quality", data=pred_data)
            if response.status_code == 200:
                prediction = response.json()
                print(f"   ✓ Model: {prediction['model_name']}")
                print(f"   ✓ Quality: {prediction['quality']}")
                print(f"   ✓ Good confidence: {prediction['confidence']['good_quality']:.1%}")
                print(f"   ✓ Bad confidence: {prediction['confidence']['bad_quality']:.1%}")
                
                # Test history
                print("6. Testing prediction history...")
                response = requests.get(f"{base_url}/mcp/tools/get_prediction_history?limit=3")
                if response.status_code == 200:
                    history = response.json()
                    print(f"   ✓ Total predictions: {history['total_predictions']}")
                    for i, pred in enumerate(history['history'][-2:], 1):
                        print(f"   ✓ {i}. {pred['model_name']}: {pred['quality']}")
                else:
                    print(f"   ✗ History failed: {response.status_code}")
                
            else:
                print(f"   ✗ Prediction failed: {response.status_code}")
                print(f"       Error: {response.text}")
        
        else:
            print(f"   ✗ Image upload failed: {response.status_code}")
            print(f"       Error: {response.text}")
    
    else:
        print(f"4. Test image not found: {test_image_path}")
        print("   Skipping upload and prediction tests")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_mcp_server()