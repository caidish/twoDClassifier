"""
Basic usage example for the 2D Material Classifier.

This example shows how to:
1. Load pretrained models for different 2D materials (graphene, hBN, etc.)
2. Make predictions on single images
3. Work with multiple material models
4. Test with sample data
"""

import os
import sys

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_loader import load_graphene_model, list_available_models
from image_utils import CenterCrop
from PIL import Image
import numpy as np


def main():
    """Main example function."""
    
    print("=== 2D Material Classifier Example ===\n")
    
    # 1. List available models
    print("Available models:")
    available_models = list_available_models()
    for model in available_models:
        print(f"  - {model}")
    print()
    
    if not available_models:
        print("No models found! Make sure the models directory contains trained models.")
        return
    
    # 2. Load a model
    model_name = available_models[0]  # Use the first available model
    print(f"Loading model: {model_name}")
    try:
        classifier = load_graphene_model(model_name)
        print("✓ Model loaded successfully!\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # 3. Get model information
    print("Model Information:")
    model_info = classifier.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 4. Example prediction on a dummy image
    print("Creating a dummy image for testing...")
    
    # Create a dummy RGB image (simulate a microscopy image)
    input_size = classifier.input_size
    dummy_image = np.random.randint(0, 256, (input_size, input_size, 3), dtype=np.uint8)
    
    print(f"Making prediction on {input_size}x{input_size} dummy image...")
    
    try:
        # Binary prediction
        prediction = classifier.predict(dummy_image)
        print(f"Binary prediction: {prediction} ({'Good' if prediction == 1 else 'Bad'})")
        
        # Probability prediction
        probabilities = classifier.predict(dummy_image, return_probabilities=True)
        print(f"Probabilities: Bad={probabilities[0][0]:.3f}, Good={probabilities[0][1]:.3f}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return
    
    print("\n=== Testing with sample data ===")
    
    # 5. Test with real images if available
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    sample_images = []
    if os.path.exists(data_dir):
        for img_file in ['Data1.jpg', 'Data2.jpg']:
            img_path = os.path.join(data_dir, img_file)
            if os.path.exists(img_path):
                sample_images.append(img_path)
    
    if sample_images:
        print(f"Found {len(sample_images)} sample image(s) in data/ directory")
        
        # Test with different models on the first sample image
        test_image = sample_images[0]
        print(f"\nTesting '{os.path.basename(test_image)}' with different models:")
        
        test_models = ['graphene_1', 'hBN_monolayer', 'hBN']  # Test a few models
        for model_name in test_models:
            if model_name in available_models:
                try:
                    print(f"\n--- Testing with {model_name} ---")
                    test_classifier = load_graphene_model(model_name)
                    prediction = test_classifier.predict_single_image(test_image)
                    probabilities = test_classifier.predict_single_image(test_image, return_probabilities=True)
                    
                    print(f"Prediction: {prediction} ({'Good' if prediction == 1 else 'Bad'})")
                    print(f"Probabilities: Bad={probabilities[0][0]:.3f}, Good={probabilities[0][1]:.3f}")
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
            else:
                print(f"Model {model_name} not available")
    else:
        print("No sample images found in data/ directory")
    
    print("\n=== Example completed successfully! ===")
    
    # 6. Show usage instructions
    print("\nTo use with your own images:")
    print("```python")
    print("classifier = load_graphene_model('hBN_monolayer')  # or any available model")
    print("prediction = classifier.predict_single_image('path/to/your/image.jpg')")
    print("print(f'Prediction: {\"Good\" if prediction == 1 else \"Bad\"}')")
    print("```")


if __name__ == "__main__":
    main()