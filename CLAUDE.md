# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installation and Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional MCP server dependencies
pip install fastapi 'uvicorn[standard]' python-multipart httpx requests
```

### Running Applications
```bash
# Command line example with sample data testing
python examples/basic_usage.py

# GUI application for interactive testing
python examples/gui_app.py

# MCP HTTP servers for remote language model access (Phase 2)
python mcp_http_server.py --host 0.0.0.0 --port 8001    # Traditional HTTP server
python mcp_fastmcp_server.py --host 0.0.0.0 --port 8002 # FastMCP HTTP server

# Unified server launcher (defaults to FastMCP HTTP)
python start_server.py                                  # Default: FastMCP HTTP
python start_server.py --type http --port 8000          # Traditional HTTP

# Test MCP server functionality
python examples/test_mcp_simple.py
```

### Testing
No formal test suite is configured. Testing is done via:
- Command line examples with sample data (data/Data1.jpg, data/Data2.jpg)
- GUI application for interactive testing with multiple models
- MCP server testing with `examples/test_mcp_simple.py` (Phase 2)
- Manual verification of model predictions

## Architecture

### Core Components

**Model Architecture**: CNN-based binary classifiers for 2D material quality assessment
- Input: RGB images (flexible size, typically 100x100)  
- Architecture: 4 Conv2D layers (32→64→128→256 filters) + Global Max Pooling + Dropout + Dense(2)
- Output: Binary classification (0=bad quality, 1=good quality flakes)

**Key Classes**:
- `GrapheneClassifier` (src/model_loader.py:13): Main wrapper for pretrained models
  - Handles model loading from JSON architecture + H5 weights
  - Provides prediction methods with automatic preprocessing
  - Extracts input size from model architecture
- `CenterCrop` (src/image_utils.py:17): Image preprocessing transform
  - Center crops to largest square, then resizes to target size
  - Used for consistent input preprocessing across models

**Model Storage**: Models are stored in `models/` directory with this structure:
```
models/
├── graphene_1/               # Basic graphene model
├── graphene_2/               # Alternative graphene model
├── hBN/                      # Hexagonal boron nitride
├── hBN_monolayer/           # Single-layer hBN
├── hBN_2to4nm/              # Multi-layer hBN
├── hBN_3to6layers/          # Thick hBN
├── hBN_20x_dot/             # hBN nanodots
├── Graphene1234/            # Graphene variant
├── Graphene_1to7/           # Multi-layer graphene

Each model directory contains:
├── trained_model.json       # Keras model architecture
└── trained_model.h5         # Trained weights
```

**Available Models**: 9 pretrained models supporting various 2D materials (graphene, hBN variants)

### Data Flow
1. Images are loaded via PIL and preprocessed with CenterCrop
2. Models expect RGB images normalized via Rescaling layer (scale=1/255)
3. Predictions return softmax probabilities or binary classifications
4. Class 0 = "Bad" quality flakes, Class 1 = "Good" quality flakes

### Usage Patterns
- Load models using `load_graphene_model(model_name)` (supports all 2D materials)
- Use `predict_single_image()` for individual files
- Use `predict()` for numpy arrays (single image or batches)
- Available models can be listed with `list_available_models()`

### GUI Application
**File**: `examples/gui_app.py` - Complete tkinter-based testing interface
- Image selection from `data/` folder (includes Data1.jpg, Data2.jpg samples)
- Model selection from all available models in `models/`
- Real-time prediction with confidence scores
- Threading for non-blocking predictions
- Visual image preview and color-coded results

**Usage**: `python examples/gui_app.py` (after activating venv)

### Project Status
**Phase 1 Complete**: Multi-material framework with GUI
- 9 pretrained models for various 2D materials
- Complete CLI and GUI interfaces  
- Sample data and comprehensive documentation
- Production-ready codebase structure

**Phase 2 Complete**: MCP server for remote language model integration
- Model Context Protocol (MCP) v2025-06-18 server implementation
- HTTP/JSON API endpoints for image upload and prediction
- Network-accessible service with configurable IP/port binding
- 4 MCP tools: upload_image, list_models, predict_flake_quality, get_prediction_history
- Comprehensive testing and documentation (MCP_USAGE.md)
- Ready for Phase 3: authentication, scaling, custom training

### Dependencies
**Core**: TensorFlow 2.8+, OpenCV, Pillow, NumPy, tkinter (GUI)
**MCP Server**: FastAPI, uvicorn, python-multipart, httpx, requests
**Optional**: matplotlib (for examples/visualization)  
**Python compatibility**: 3.8-3.10