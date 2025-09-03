# MCP Server Usage Guide

## Overview

The 2D Material Classification MCP (Model Context Protocol) Server enables remote language models to:

- Upload images via IP address and port connection
- Select from 10+ available neural network models  
- Extract flake quality information from 2D material images
- Get prediction history and model information

## Quick Start

### 1. Installation

```bash
# Install additional MCP dependencies
source venv/bin/activate
pip install fastapi 'uvicorn[standard]' python-multipart httpx requests

# Verify core dependencies are installed
pip install -r requirements.txt
```

### 2. Start MCP Server

```bash
# Basic server start (localhost:8000)
python mcp_http_server.py

# Custom host and port
python mcp_http_server.py --host 0.0.0.0 --port 8001

# Development mode with auto-reload
python mcp_http_server.py --host 127.0.0.1 --port 8001 --reload
```

### 3. Test Server Connection

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test MCP capabilities
curl http://localhost:8000/.well-known/mcp-capabilities

# Run comprehensive test
python test_mcp_simple.py
```

## MCP Server Capabilities

### Available Tools

1. **`upload_image`**: Upload images for classification (supports up to 16MB files via Base64 encoding)
2. **`list_models`**: Get available model list  
3. **`predict_flake_quality`**: Run 2D material quality prediction
4. **`get_prediction_history`**: Retrieve prediction history

### Supported Models

The server provides access to 10+ pretrained models:

- **Graphene Models**: `graphene_1`, `graphene_2`, `Graphene1234`, `Graphene_1to7`, `Graphene_Shuwen`
- **hBN Models**: `hBN`, `hBN_monolayer`, `hBN_2to4nm`, `hBN_3to6layers`, `hBN_20x_dot`

## Remote Language Model Integration

### MCP Protocol Details

- **Protocol**: MCP v2025-06-18
- **Transport**: HTTP with JSON-RPC 2.0
- **Authentication**: None (add OAuth 2.1 for production)
- **CORS**: Enabled for all origins (configure for production)

### Connection Workflow

1. **Connect**: Remote LM connects to `http://<server-ip>:<port>`
2. **Discover**: GET `/.well-known/mcp-capabilities` for available tools
3. **Upload**: POST `/mcp/tools/upload_image` with base64 image data
4. **Select**: Choose model from `/mcp/tools/list_models` response
5. **Predict**: POST `/mcp/tools/predict_flake_quality` with model and image
6. **Results**: Receive quality classification and confidence scores

## API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-30T15:56:33.474121",
  "loaded_models": ["hBN_monolayer"],
  "data_directory": "/path/to/data",
  "models_directory": "/path/to/models"
}
```

### MCP Capabilities
```http
GET /.well-known/mcp-capabilities
```
**Response:**
```json
{
  "protocol": "mcp",
  "version": "2025-06-18",
  "capabilities": {"tools": true, "resources": false, "prompts": false},
  "tools": [...]
}
```

### Image Upload
```http
POST /mcp/tools/upload_image
Content-Type: application/x-www-form-urlencoded

image_data=<base64-encoded-image>
filename=<original-filename>
```
**Response:**
```json
{
  "success": true,
  "filename": "uuid-generated-filename.jpg",
  "original_filename": "Data1.jpg",
  "size": 6207776,
  "path": "/path/to/saved/image"
}
```

### List Models
```http
GET /mcp/tools/list_models
```
**Response:**
```json
{
  "success": true,
  "models": ["hBN_monolayer", "graphene_1", ...],
  "count": 10
}
```

### Predict Flake Quality
```http
POST /mcp/tools/predict_flake_quality
Content-Type: application/x-www-form-urlencoded

model_name=<model-name>
image_filename=<uploaded-filename>
```
**Response:**
```json
{
  "success": true,
  "model_name": "hBN_monolayer",
  "image_filename": "uuid-generated-filename.jpg",
  "prediction": 1,
  "quality": "Good Quality",
  "confidence": {
    "bad_quality": 0.099,
    "good_quality": 0.901
  },
  "timestamp": "2025-08-30T15:58:00.000000"
}
```

### Get Prediction History
```http
GET /mcp/tools/get_prediction_history?limit=10
```
**Response:**
```json
{
  "success": true,
  "history": [...],
  "total_predictions": 1
}
```

## Example Usage

### Python Client Example

```python
import requests
import base64

# Server connection
server_url = "http://192.168.1.100:8001"

# Upload image
with open("my_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

upload_response = requests.post(
    f"{server_url}/mcp/tools/upload_image",
    data={"image_data": image_data, "filename": "my_image.jpg"}
)
uploaded_filename = upload_response.json()["filename"]

# Get available models
models_response = requests.get(f"{server_url}/mcp/tools/list_models")
models = models_response.json()["models"]

# Make prediction
prediction_response = requests.post(
    f"{server_url}/mcp/tools/predict_flake_quality",
    data={"model_name": models[0], "image_filename": uploaded_filename}
)

result = prediction_response.json()
print(f"Quality: {result['quality']}")
print(f"Confidence: {result['confidence']['good_quality']:.1%}")
```

### cURL Example

```bash
# Upload image (base64 encode first)
base64 -i my_image.jpg | tr -d '\n' > image_b64.txt

curl -X POST http://192.168.1.100:8001/mcp/tools/upload_image \
  -d "image_data=$(cat image_b64.txt)" \
  -d "filename=my_image.jpg"

# Make prediction
curl -X POST http://192.168.1.100:8001/mcp/tools/predict_flake_quality \
  -d "model_name=hBN_monolayer" \
  -d "image_filename=uploaded-uuid-filename.jpg"
```

## Configuration

### Environment Variables

```bash
# Server configuration
export MCP_HOST=0.0.0.0
export MCP_PORT=8001
export MCP_ENVIRONMENT=production

# Security configuration  
export CORS_ORIGINS=https://trusted-site.com,https://another-site.com
```

### Production Deployment

For production use:

1. **Security**: Add OAuth 2.1 authentication
2. **CORS**: Configure specific allowed origins
3. **HTTPS**: Use reverse proxy (nginx) with SSL certificates
4. **Monitoring**: Add logging and health monitoring
5. **Scaling**: Consider load balancing for high traffic

### File Storage

- **Images**: Uploaded to `data/` directory with original filenames (preserved with counter for duplicates)
- **File Size**: Supports images up to 16MB (increased from 2MB default limit)
- **Models**: Stored in `models/` directory (9 pretrained models)
- **History**: In-memory storage (add database for persistence)

## Troubleshooting

### Common Issues

1. **Server won't start**:
   - Check if port is already in use: `lsof -i :8001`
   - Verify Python environment: `source venv/bin/activate`

2. **Model loading errors**:
   - Ensure `models/` directory exists with `.json` and `.h5` files
   - Check model file permissions

3. **Image upload fails**:
   - Verify `data/` directory is writable
   - Check image file size limits (16MB max for Base64 uploads)
   - Ensure proper base64 encoding
   - For larger files, consider using file path upload methods

4. **Connection refused**:
   - Verify server is running: `curl http://localhost:8001/health`  
   - Check firewall settings for external connections
   - Confirm host/port binding matches client requests

### Debug Mode

```bash
# Enable debug logging
python mcp_http_server.py --host 0.0.0.0 --port 8001 --reload

# Test with verbose client
python test_mcp_simple.py
```

### Server Status

Check server status and loaded models:
```bash
curl http://localhost:8001/health | jq
```

This will show active models, directory paths, and server health status.