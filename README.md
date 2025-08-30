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
python gui_app.py
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
├── examples/              # Usage examples
│   └── basic_usage.py
├── data/                  # Sample test images
│   ├── Data1.jpg
│   └── Data2.jpg
├── gui_app.py             # GUI testing application
├── GUI_USAGE.md           # GUI usage guide
├── mcp_server.py          # MCP server for remote language models
├── mcp_config.py          # MCP server configuration
├── mcp_client_example.py  # MCP client example code
├── mcp_requirements.txt   # Additional MCP dependencies
├── test_mcp_simple.py     # MCP server testing script
├── MCP_USAGE.md           # MCP server documentation
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
- **Multiple model support** (10+ pretrained models available)
- **Sample data included** for immediate testing

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
- **Image Selection**: Choose from sample images in `data/` folder
- **Model Selection**: Pick from 10+ available models (graphene, hBN variants)
- **Real-time Prediction**: Instant results with confidence scores
- **Visual Interface**: Image preview and color-coded results
- **Model Comparison**: Easy switching between different models

See `GUI_USAGE.md` for detailed instructions.

## 🌐 MCP Server (Remote Access)

The project now includes a Model Context Protocol (MCP) server that enables remote language models to access the 2D material classification system via network connections.

### Quick Start

```bash
# Install MCP dependencies
pip install fastapi 'uvicorn[standard]' python-multipart httpx

# Start MCP server (default: localhost:8000)
python mcp_server.py

# Custom IP and port for remote access
python mcp_server.py --host 0.0.0.0 --port 8001
```

### Remote Language Model Integration

Remote LMs can connect via HTTP to:
1. **Upload images**: POST base64-encoded images to server
2. **Select models**: Choose from 10+ available neural networks
3. **Get predictions**: Receive quality classifications with confidence scores
4. **Access history**: Retrieve previous prediction results

**Connection URL**: `http://<server-ip>:<port>`

**Available MCP Tools**:
- `upload_image` - Upload images for classification
- `list_models` - Get available model list
- `predict_flake_quality` - Run quality predictions
- `get_prediction_history` - Access prediction history

### Example Usage

```python
# Remote client connection
import requests, base64

server = "http://192.168.1.100:8001"

# Upload image
with open("image.jpg", "rb") as f:
    data = {"image_data": base64.b64encode(f.read()).decode(), "filename": "image.jpg"}
upload = requests.post(f"{server}/mcp/tools/upload_image", data=data)

# Get prediction
prediction = requests.post(f"{server}/mcp/tools/predict_flake_quality", 
    data={"model_name": "hBN_monolayer", "image_filename": upload.json()["filename"]})
print(f"Quality: {prediction.json()['quality']}")
```

**Documentation**: See `MCP_USAGE.md` for complete API reference and setup instructions.

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
- ✅ Clean, modular codebase with 10+ pretrained models
- ✅ Command-line interface and comprehensive examples
- ✅ GUI application for easy testing and demonstration
- ✅ Sample data and complete documentation

**Phase 2 - Completed:**
- ✅ Model Context Protocol (MCP) server for remote language model integration
- ✅ HTTP API endpoints for image upload and prediction
- ✅ Network-accessible service with configurable IP/port binding
- ✅ RESTful API with JSON responses and comprehensive documentation
- ✅ Production-ready MCP server with health monitoring and logging

**Ready for Phase 3:** Advanced features, authentication, custom training, and production deployment scaling.

## 🤝 Contributing

This project was extracted and cleaned from the original CC_v12 codebase. The neural network architectures and preprocessing methods have been preserved while creating a more maintainable structure.

## 📄 License

Extracted from CC_v12 project - check original project for licensing information.

## 📚 References

- Original CC_v12 project for microscopy automation
- TensorFlow/Keras documentation
- Materials science research on 2D material characterization