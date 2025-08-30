"""
Model loading utilities for graphene classification.
Handles loading of pretrained TensorFlow models.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from typing import Optional


class GrapheneClassifier:
    """
    Wrapper class for graphene classification models.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the classifier with a pretrained model.
        
        Args:
            model_path: Path to directory containing trained_model.json and trained_model.h5
        """
        self.model_path = model_path
        self.model = None
        self.input_size = None
        self._load_model()
        self._extract_input_size()
    
    def _load_model(self):
        """Load the TensorFlow model from JSON and H5 files."""
        json_path = os.path.join(self.model_path, 'trained_model.json')
        weights_path = os.path.join(self.model_path, 'trained_model.h5')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Model architecture file not found: {json_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")
        
        try:
            with open(json_path, 'r') as f:
                model_json = f.read()
            
            self.model = model_from_json(model_json)
            self.model.load_weights(weights_path)
            print(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _extract_input_size(self):
        """Extract the expected input size from the model architecture."""
        try:
            # Try to get input shape from the model
            input_shape = self.model.input_shape
            if len(input_shape) >= 3:
                # Assuming square input images
                self.input_size = input_shape[1] if input_shape[1] is not None else 100
            else:
                self.input_size = 100  # Default fallback
                
        except Exception:
            self.input_size = 100  # Default fallback
            
        print(f"Model expects input size: {self.input_size}x{self.input_size}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "input_size": self.input_size,
            "trainable_params": self.model.count_params(),
            "layers": len(self.model.layers),
            "model_path": self.model_path
        }
    
    def predict(self, image_array, return_probabilities: bool = False):
        """
        Make a prediction on a single image or batch of images.
        
        Args:
            image_array: Preprocessed image array of shape (height, width, 3) or (batch, height, width, 3)
            return_probabilities: If True, return class probabilities instead of binary prediction
            
        Returns:
            Prediction(s) - binary classification (0=bad, 1=good) or probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure input is 4D (batch dimension)
        if len(image_array.shape) == 3:
            image_array = tf.expand_dims(image_array, 0)
        
        # Make prediction
        predictions = self.model.predict(image_array, verbose=0)
        probabilities = tf.nn.softmax(predictions)
        
        if return_probabilities:
            return probabilities.numpy()
        else:
            # Return binary predictions (class with highest probability)
            binary_predictions = tf.argmax(probabilities, axis=1).numpy()
            return binary_predictions if len(binary_predictions) > 1 else binary_predictions[0]
    
    def predict_single_image(self, image_path: str, return_probabilities: bool = False):
        """
        Convenient method to predict on a single image file.
        
        Args:
            image_path: Path to image file
            return_probabilities: If True, return class probabilities
            
        Returns:
            Prediction - binary classification or probabilities
        """
        from PIL import Image
        import numpy as np
        import sys
        import os
        
        # Add src directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from image_utils import CenterCrop
        
        # Load and preprocess image
        image = Image.open(image_path)
        transform = CenterCrop(crop_size=float('inf'), target_size=self.input_size)
        image_processed = transform(image)
        image_array = np.array(image_processed)
        
        return self.predict(image_array, return_probabilities)


def load_graphene_model(model_name: str = "graphene_1") -> GrapheneClassifier:
    """
    Load a pretrained 2D material classification model.
    
    Args:
        model_name: Name of the model (e.g., "graphene_1", "graphene_2", "hBN", "hBN_monolayer", etc.)
        
    Returns:
        GrapheneClassifier instance
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, "models", model_name)
    
    if not os.path.exists(model_path):
        available_models = []
        models_dir = os.path.join(project_root, "models")
        if os.path.exists(models_dir):
            available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        raise ValueError(f"Model '{model_name}' not found at {model_path}. Available models: {available_models}")
    
    return GrapheneClassifier(model_path)


def list_available_models() -> list:
    """
    List all available pretrained models.
    
    Returns:
        List of available model names
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_dir = os.path.join(project_root, "models")
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for item in os.listdir(models_dir):
        model_path = os.path.join(models_dir, item)
        if os.path.isdir(model_path):
            json_file = os.path.join(model_path, 'trained_model.json')
            h5_file = os.path.join(model_path, 'trained_model.h5')
            if os.path.exists(json_file) and os.path.exists(h5_file):
                models.append(item)
    
    return models