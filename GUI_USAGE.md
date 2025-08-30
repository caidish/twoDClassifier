# GUI Usage Guide

## Running the GUI Application

To start the 2D Material Classifier GUI:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch GUI
python gui_app.py
```

## Features

### 1. Image Selection
- **Select Image dropdown**: Choose from available images in the `data/` folder
- **Refresh Images button**: Reload the list of available images
- **Image Preview**: Shows a thumbnail of the selected image

### 2. Model Selection  
- **Select Model dropdown**: Choose from available models in the `models/` folder
- **Refresh Models button**: Reload the list of available models

Available models include:
- `graphene_1`, `graphene_2` - Graphene classification models
- `hBN`, `hBN_monolayer`, `hBN_2to4nm` - hBN (hexagonal boron nitride) models
- `Graphene1234`, `Graphene_1to7`, `Graphene_Shuwen` - Additional graphene variants
- `hBN_20x_dot`, `hBN_3to6layers` - Specialized hBN models

### 3. Prediction
- **Run Prediction button**: Executes the classification
- Results show:
  - **Prediction**: "Good Quality" or "Bad Quality" 
  - **Confidence**: Probability percentages for both classes
  - **Color coding**: Green for good quality, red for bad quality

### 4. Status Information
- Real-time status updates during model loading and prediction
- Progress indicator during processing
- Error messages if issues occur

## Sample Test Images

The `data/` folder contains sample images:
- `Data1.jpg` - Test image 1
- `Data2.jpg` - Test image 2

## Tips

1. **Model Loading**: Models are loaded on-demand and cached for subsequent predictions
2. **Threading**: Predictions run in background threads to keep the GUI responsive  
3. **Image Formats**: Supports JPG, PNG, BMP, TIFF image formats
4. **Multiple Tests**: You can easily switch between different model/image combinations

## Troubleshooting

- If the GUI doesn't start, ensure you have activated the virtual environment
- If models fail to load, check that the `models/` directory contains the required `.json` and `.h5` files
- If images don't appear, verify they are in the `data/` folder and have supported extensions