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

### Run Example

```bash
python examples/basic_usage.py
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
├── data/                  # Place your data here
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

## 📊 Model Performance

These models were trained on 2D material microscopy images to classify flake quality across different materials (graphene, hBN, etc.):
- **Class 0**: Bad/poor quality flakes
- **Class 1**: Good/high quality flakes

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

## 🎯 Use Cases

This framework is ideal for:
- 2D material quality assessment (graphene, hBN, transition metal dichalcogenides, etc.)
- Multi-material flake classification
- Microscopy image analysis
- Research in materials science and nanotechnology
- Educational deep learning projects

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

## 🤝 Contributing

This project was extracted and cleaned from the original CC_v12 codebase. The neural network architectures and preprocessing methods have been preserved while creating a more maintainable structure.

## 📄 License

Extracted from CC_v12 project - check original project for licensing information.

## 📚 References

- Original CC_v12 project for microscopy automation
- TensorFlow/Keras documentation
- Materials science research on 2D material characterization