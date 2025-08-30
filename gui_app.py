#!/usr/bin/env python3
"""
Simple GUI for 2D Material Classifier Testing

This GUI application allows users to:
1. Select images from the data folder
2. Select models from the models folder
3. Run predictions and view results
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
from PIL import Image, ImageTk
import threading

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.model_loader import load_graphene_model, list_available_models
except ImportError:
    # Fallback for different import paths
    from model_loader import load_graphene_model, list_available_models


class TwoDMaterialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Material Classifier - Testing GUI")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Variables
        self.selected_image_path = tk.StringVar()
        self.selected_model = tk.StringVar()
        self.prediction_result = tk.StringVar()
        self.prediction_confidence = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        
        # Current classifier instance
        self.classifier = None
        
        # Set up the GUI
        self.setup_gui()
        self.refresh_data()
    
    def setup_gui(self):
        """Set up the main GUI layout."""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="2D Material Classifier", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Image selection section
        ttk.Label(main_frame, text="Select Image:", font=("Arial", 12, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=(10, 5))
        
        self.image_combo = ttk.Combobox(main_frame, textvariable=self.selected_image_path,
                                       state="readonly", width=50)
        self.image_combo.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.image_combo.bind('<<ComboboxSelected>>', self.on_image_selected)
        
        ttk.Button(main_frame, text="Refresh Images", 
                  command=self.refresh_images).grid(row=2, column=2, padx=(10, 0))
        
        # Image preview
        self.image_frame = ttk.LabelFrame(main_frame, text="Image Preview", padding="10")
        self.image_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        self.image_frame.columnconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="No image selected")
        self.image_label.grid(row=0, column=0)
        
        # Model selection section
        ttk.Label(main_frame, text="Select Model:", font=("Arial", 12, "bold")).grid(
            row=4, column=0, sticky=tk.W, pady=(20, 5))
        
        self.model_combo = ttk.Combobox(main_frame, textvariable=self.selected_model,
                                       state="readonly", width=50)
        self.model_combo.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(main_frame, text="Refresh Models", 
                  command=self.refresh_models).grid(row=5, column=2, padx=(10, 0))
        
        # Predict button
        self.predict_button = ttk.Button(main_frame, text="Run Prediction", 
                                        command=self.run_prediction, style="Accent.TButton")
        self.predict_button.grid(row=6, column=0, columnspan=3, pady=(20, 10))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        results_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        results_frame.columnconfigure(1, weight=1)
        
        ttk.Label(results_frame, text="Prediction:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.result_label = ttk.Label(results_frame, textvariable=self.prediction_result,
                                     font=("Arial", 12, "bold"))
        self.result_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(results_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(results_frame, textvariable=self.prediction_confidence).grid(row=1, column=1, sticky=tk.W)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
        status_frame.columnconfigure(0, weight=1)
        
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.status_text).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # Progress bar (initially hidden)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=0, column=2, sticky=tk.E, padx=(10, 0))
    
    def refresh_data(self):
        """Refresh both images and models."""
        self.refresh_images()
        self.refresh_models()
    
    def refresh_images(self):
        """Refresh the list of available images in the data folder."""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        images = []
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    images.append(file)
        
        self.image_combo['values'] = sorted(images)
        if images and not self.selected_image_path.get():
            self.selected_image_path.set(images[0])
            self.on_image_selected()
        
        self.status_text.set(f"Found {len(images)} images in data folder")
    
    def refresh_models(self):
        """Refresh the list of available models."""
        try:
            models = list_available_models()
            self.model_combo['values'] = sorted(models)
            if models and not self.selected_model.get():
                self.selected_model.set(models[0])
            
            self.status_text.set(f"Found {len(models)} available models")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def on_image_selected(self, event=None):
        """Handle image selection and show preview."""
        image_name = self.selected_image_path.get()
        if not image_name:
            return
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        image_path = os.path.join(data_dir, image_name)
        
        try:
            # Load and resize image for preview
            pil_image = Image.open(image_path)
            
            # Calculate size maintaining aspect ratio
            max_size = (300, 300)
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            self.status_text.set(f"Loaded image: {image_name} ({pil_image.size[0]}x{pil_image.size[1]})")
            
        except Exception as e:
            self.image_label.configure(image="", text=f"Error loading image: {str(e)}")
            self.image_label.image = None
    
    def run_prediction(self):
        """Run prediction on selected image with selected model."""
        image_name = self.selected_image_path.get()
        model_name = self.selected_model.get()
        
        if not image_name:
            messagebox.showwarning("Warning", "Please select an image")
            return
        
        if not model_name:
            messagebox.showwarning("Warning", "Please select a model")
            return
        
        # Disable predict button and show progress
        self.predict_button.configure(state='disabled')
        self.progress.start()
        self.status_text.set("Loading model and running prediction...")
        
        # Clear previous results
        self.prediction_result.set("")
        self.prediction_confidence.set("")
        
        # Run prediction in a separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._predict_thread, 
                                 args=(image_name, model_name))
        thread.daemon = True
        thread.start()
    
    def _predict_thread(self, image_name, model_name):
        """Run prediction in a separate thread."""
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            image_path = os.path.join(data_dir, image_name)
            
            # Load model (reuse if same model)
            if self.classifier is None or self.classifier.model_path.split(os.sep)[-1] != model_name:
                self.root.after(0, lambda: self.status_text.set(f"Loading model: {model_name}..."))
                self.classifier = load_graphene_model(model_name)
            
            # Run prediction
            self.root.after(0, lambda: self.status_text.set("Running prediction..."))
            prediction = self.classifier.predict_single_image(image_path)
            probabilities = self.classifier.predict_single_image(image_path, return_probabilities=True)
            
            # Update GUI with results
            result_text = "Good Quality" if prediction == 1 else "Bad Quality"
            confidence_text = f"Bad: {probabilities[0][0]:.1%}, Good: {probabilities[0][1]:.1%}"
            
            # Schedule GUI updates on main thread
            self.root.after(0, lambda: self._update_results(result_text, confidence_text))
            self.root.after(0, lambda: self.status_text.set("Prediction completed"))
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self.status_text.set("Prediction failed"))
        
        finally:
            # Re-enable predict button and hide progress
            self.root.after(0, lambda: self.predict_button.configure(state='normal'))
            self.root.after(0, lambda: self.progress.stop())
    
    def _update_results(self, result_text, confidence_text):
        """Update the results display."""
        self.prediction_result.set(result_text)
        self.prediction_confidence.set(confidence_text)
        
        # Color code the result
        if "Good" in result_text:
            self.result_label.configure(foreground="green")
        else:
            self.result_label.configure(foreground="red")


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    
    # Set up theme
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "light")
    except:
        pass  # Theme not available, use default
    
    app = TwoDMaterialGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()


if __name__ == "__main__":
    main()