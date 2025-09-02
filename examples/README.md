# Examples and Demos

This directory contains example scripts and demo applications for the 2D Material Classifier.

## Files

### Core Examples

**`basic_usage.py`** - Command-line usage example
- Shows how to load models and make predictions
- Demonstrates basic API usage
- Good starting point for understanding the framework

**`gui_app.py`** - GUI testing application  
- Complete tkinter-based interface for testing models
- Image upload and selection from data folder
- Real-time predictions with confidence scores
- Visual results with color-coded quality indicators

### MCP Server Examples

**`test_mcp_simple.py`** - MCP server testing script
- Tests all MCP server endpoints
- Validates image upload and prediction workflow
- Useful for debugging server issues

**`mcp_client_example.py`** - MCP client usage examples
- Shows how to connect to MCP servers
- Example HTTP requests and responses
- Client-side integration patterns

**`mcp_config.py`** - Configuration example (legacy)
- Shows server configuration patterns
- Mostly for reference (superseded by command-line args)

## Usage

```bash
# Run examples from project root directory
cd /path/to/twoDClassifier

# Command-line example
python examples/basic_usage.py

# GUI application
python examples/gui_app.py

# Test MCP server (start server first)
python start_server.py  # In another terminal
python examples/test_mcp_simple.py
```

## Requirements

All examples use the same dependencies as the main project:
- TensorFlow 2.8+
- OpenCV, Pillow, NumPy
- tkinter (for GUI)
- requests, httpx (for MCP client examples)

Make sure to activate the virtual environment and install requirements before running examples.