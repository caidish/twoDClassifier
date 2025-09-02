# 2D Material Classifier 🧬

A clean, modern deep learning framework for 2D material classification, extracted and refactored from the CC_v12 project. Supports classification of various 2D materials including graphene, hBN (hexagonal boron nitride), and other layered materials.

## 🚀 Quick Start

### Installation

```bash
# Clone or download this project
cd twoDClassifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.model_loader import load_graphene_model

# Load a pretrained model (supports various 2D materials)
classifier = load_graphene_model('hBN_monolayer')  # or 'graphene_1', 'hBN', etc.

# Make prediction on an image
prediction = classifier.predict_single_image('path/to/image.jpg')
print(f"Prediction: {'Good' if prediction == 1 else 'Bad'}")

# Get probabilities
probs = classifier.predict_single_image('path/to/image.jpg', return_probabilities=True)
print(f"Good probability: {probs[0][1]:.3f}")
```

### Run Examples

```bash
# Command line example
python examples/basic_usage.py

# GUI Application
python examples/gui_app.py
```

## 📁 Project Structure

```
twoDClassifier/
├── models/                 # Pretrained models for various 2D materials
│   ├── graphene_1/
│   │   ├── trained_model.json
│   │   └── trained_model.h5
│   ├── graphene_2/
│   │   ├── trained_model.json
│   │   └── trained_model.h5
│   ├── hBN/
│   │   ├── trained_model.json
│   │   └── trained_model.h5
│   ├── hBN_monolayer/
│   │   ├── trained_model.json
│   │   └── trained_model.h5
│   └── [other 2D material models...]
├── src/                   # Source code
│   ├── __init__.py
│   ├── model_loader.py    # Model loading utilities
│   └── image_utils.py     # Image preprocessing
├── examples/              # Usage examples and demos
│   ├── basic_usage.py     # Command-line usage example
│   ├── gui_app.py         # GUI testing application
│   ├── test_mcp_simple.py # MCP server testing
│   └── mcp_client_example.py  # MCP client example
├── data/                  # Sample test images (uploaded images stored here)
├── GUI_USAGE.md           # GUI usage guide
├── mcp_http_server.py     # HTTP/REST MCP server for web clients
├── mcp_fastmcp_server.py  # FastMCP server (HTTP + stdio transport, 88 lines)
├── mcp_fastmcp_stdio_server.py  # FastMCP stdio-only server (Claude Desktop)
├── start_server.py        # Unified server launcher with port fallback
├── start_http_server.py   # HTTP server launcher with port fallback
├── start_fastmcp_server.py # FastMCP HTTP server launcher with port fallback
├── claude_desktop_config.json  # Claude Desktop configuration example
├── MCP_USAGE.md           # HTTP MCP server documentation
├── todo.md                # Development planning document
├── requirements.txt
└── README.md
```

## 🧠 Model Architecture

All pretrained models use CNN architectures optimized for 2D material classification:

**Standard Architecture (used across material types):**
- Input: RGB images (flexible size, default 100x100)
- 4 Convolutional layers with max pooling
- Filters: 32 → 64 → 128 → 256
- Global max pooling + dropout
- Binary classification output

**Key Features:**
- Automatic image preprocessing (center crop + resize)
- Class balancing support  
- TensorFlow 2.x compatible
- Clean, modular code structure
- **GUI testing interface** for easy model/image testing
- **Multiple model support** (9 pretrained models available)
- **Sample data and examples** for immediate testing

## 📊 Model Performance

These models were trained on 2D material microscopy images to classify flake quality across different materials (graphene, hBN, etc.):
- **Class 0**: Bad/poor quality flakes
- **Class 1**: Good/high quality flakes

## 🖥️ GUI Application

For easy testing and demonstration, use the included GUI:

```bash
python gui_app.py
```

**GUI Features:**
- **Image Selection**: Upload your own images or use existing ones in `data/` folder
- **Model Selection**: Pick from 9 available models (graphene, hBN variants)
- **Real-time Prediction**: Instant results with confidence scores
- **Visual Interface**: Image preview and color-coded results
- **Model Comparison**: Easy switching between different models

See `GUI_USAGE.md` for detailed instructions.

## 🌐 MCP Server (Remote Access & Claude Desktop)

The project includes **dual MCP server support** for maximum compatibility:

### Server Options

**1. HTTP/REST Server (`mcp_http_server.py`)** - For web clients and API access:
```bash
# Install HTTP server dependencies
pip install fastapi 'uvicorn[standard]' python-multipart httpx

# Start HTTP server (default: localhost:8000)
python mcp_http_server.py

# Custom IP and port for remote access
python mcp_http_server.py --host 0.0.0.0 --port 8001
```

**2. FastMCP Server (`mcp_fastmcp_server.py`)** - For MCP clients (⭐ **70% less code**):
```bash
# Install FastMCP dependencies
pip install fastmcp tensorflow opencv-python pillow numpy

# Start HTTP server (default: localhost:8000 with port fallback)
python mcp_fastmcp_server.py

# Custom configuration
python mcp_fastmcp_server.py --host 0.0.0.0 --port 8001

# For stdio transport (Claude Desktop)
python mcp_fastmcp_server.py --transport stdio
# OR use dedicated stdio server
python mcp_fastmcp_stdio_server.py
```

### Why Two Servers?

- **HTTP Server**: Perfect for web applications, remote API access, production deployments
- **FastMCP Server**: Seamless MCP client integration with compact, clean code
- **Both share the same model logic**: No duplication, consistent results

### Available MCP Tools (Both Servers)

- `upload_image` - Upload images for classification
- `list_models` - Get available model list (9 models)
- `predict_flake_quality` - Run quality predictions with confidence scores
- `get_prediction_history` - Access prediction history

### Usage Examples

**HTTP Client (Web/API)**:
```python
import requests, base64

server = "http://localhost:8000"  # or remote IP

# Upload and analyze
with open("image.jpg", "rb") as f:
    data = {"image_data": base64.b64encode(f.read()).decode(), "filename": "image.jpg"}
upload = requests.post(f"{server}/mcp/tools/upload_image", data=data)

prediction = requests.post(f"{server}/mcp/tools/predict_flake_quality", 
    data={"model_name": "hBN_monolayer", "image_filename": upload.json()["filename"]})
print(f"Quality: {prediction.json()['quality']}")
```

**MCP Client Integration**:

*HTTP clients (web/API)*:
```python
import requests
response = requests.get("http://localhost:8000/mcp/tools/list_models")
```

*Claude Desktop (stdio)*:
1. Configure client with full path to `mcp_fastmcp_stdio_server.py` 
2. Chat with AI: *"What 2D material models are available?"*
3. Upload images: *"Can you analyze this graphene flake for quality?"*

### Code Comparison

| Feature | HTTP Server | FastMCP Server |
|---------|-------------|----------------|
| **Lines of Code** | 288 lines | **88 lines** ⭐ |
| **Setup Complexity** | Manual FastAPI setup | **Decorator-based** ⭐ |
| **Protocol Handling** | Manual JSON-RPC | **Automatic** ⭐ |
| **Transport Support** | HTTP only | **HTTP + stdio** ⭐ |
| **MCP Clients** | ❌ Not compatible | ✅ **Native support** |
| **Web/API Clients** | ✅ Perfect | ✅ **HTTP mode** ⭐ |
| **Remote Access** | ✅ Network ready | ✅ **HTTP mode** ⭐ |

**Documentation**: 
- `MCP_USAGE.md` - HTTP server API reference

## 🛠️ Advanced Usage

### Custom Preprocessing

```python
from src.image_utils import CenterCrop
from PIL import Image
import numpy as np

# Load and preprocess image manually
image = Image.open('image.jpg')
transform = CenterCrop(crop_size=float('inf'), target_size=100)
processed_image = transform(image)
image_array = np.array(processed_image)

# Make prediction
prediction = classifier.predict(image_array)
```

### Batch Prediction

```python
# Process multiple images at once
image_batch = np.stack([image1, image2, image3])  # Shape: (3, height, width, 3)
predictions = classifier.predict(image_batch)
```

### Model Information

```python
# Get detailed model information
info = classifier.get_model_info()
print(f"Input shape: {info['input_shape']}")
print(f"Trainable parameters: {info['trainable_params']}")
```

## 📋 Requirements

- Python 3.8-3.10
- TensorFlow 2.8+
- OpenCV
- Pillow
- NumPy
- Tkinter (included with Python - for GUI)

## 🎯 Use Cases

This framework is ideal for:
- **Research**: 2D material quality assessment (graphene, hBN, transition metal dichalcogenides, etc.)
- **Industrial**: Automated quality control in materials manufacturing
- **Educational**: Deep learning demonstrations and materials science teaching
- **Development**: Rapid prototyping of material classification systems
- **Analysis**: Batch processing of microscopy images

## 🔧 Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure both `.json` and `.h5` files are present
2. **Image size issues**: Models automatically handle different input sizes
3. **Memory errors**: Process images in smaller batches
4. **TensorFlow warnings**: These are usually safe to ignore

### Performance Tips

- Use batch processing for multiple images
- Consider image size vs. accuracy tradeoffs
- GPU acceleration works automatically if CUDA is available

## 🚀 Current Status & Next Phase

**Phase 1 - Completed:**
- ✅ Multi-material 2D classification framework (graphene, hBN, etc.)
- ✅ Clean, modular codebase with 9 pretrained models
- ✅ Command-line interface and comprehensive examples
- ✅ GUI application for easy testing and demonstration
- ✅ Sample data and complete documentation

**Phase 2 - Completed:**
- ✅ **Dual MCP server architecture** supporting both web and MCP clients
- ✅ **HTTP/REST server** (288 lines) for web applications and remote API access
- ✅ **FastMCP server** (88 lines) with HTTP + stdio transport support
- ✅ **Unified server launcher** with automatic port fallback
- ✅ **Shared model logic** - both servers use identical classification algorithms
- ✅ **Network-accessible service** with configurable IP/port binding
- ✅ **Production-ready architecture** with comprehensive documentation

**Phase 3 - Completed:**
- ✅ **Project organization** - moved examples to `examples/` directory
- ✅ **Code cleanup** - removed outdated config and test files
- ✅ **Documentation updates** - reflects current project structure
- ✅ **Server launcher scripts** - multiple options for easy deployment

**Ready for Phase 4+:** Advanced authentication, custom model training, production scaling, and enterprise deployment features.

## 🤝 Contributing

This project was extracted and cleaned from the original CC_v12 codebase. The neural network architectures and preprocessing methods have been preserved while creating a more maintainable structure.

## 📄 License

Extracted from CC_v12 project - check original project for licensing information.

## 📚 References

- Original CC_v12 project for microscopy automation
- TensorFlow/Keras documentation
- Materials science research on 2D material characterization