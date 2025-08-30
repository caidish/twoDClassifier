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
```

### Running Examples
```bash
# Run basic usage example
python examples/basic_usage.py
```

### Testing
No formal test suite is configured. Testing is done via the examples and manual verification of model predictions.

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
├── graphene_1/
│   ├── trained_model.json    # Keras model architecture
│   └── trained_model.h5      # Trained weights
├── graphene_2/
│   ├── trained_model.json
│   └── trained_model.h5
└── [other model directories...]
```

### Data Flow
1. Images are loaded via PIL and preprocessed with CenterCrop
2. Models expect RGB images normalized via Rescaling layer (scale=1/255)
3. Predictions return softmax probabilities or binary classifications
4. Class 0 = "Bad" quality flakes, Class 1 = "Good" quality flakes

### Usage Patterns
- Load models using `load_graphene_model(model_name)` 
- Use `predict_single_image()` for individual files
- Use `predict()` for numpy arrays (single image or batches)
- Available models can be listed with `list_available_models()`

### Dependencies
Core: TensorFlow 2.8+, OpenCV, Pillow, NumPy
Optional: matplotlib (for examples/visualization)
Python compatibility: 3.8-3.10