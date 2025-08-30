"""
Image processing utilities for graphene classification.
Extracted and cleaned from the original CC_v12 project.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
import random

# Set OpenCV to handle large images
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 60).__str__()


class CenterCrop:
    """
    Crops out the center portion of an image and resizes to a target size.
    
    Attributes:
        crop_size: size of the region to crop from the image, the image is 
                  cropped to the largest possible square if crop_size > either dimension
        target_size: the cropped image is resized to this size
        interpolation: interpolation method for resizing
    """
    
    def __init__(self, crop_size: int, target_size: int):
        self.crop_size = crop_size
        self.target_size = target_size
        self.interpolation = Image.BILINEAR
        
    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        width_to_crop = max(w - self.crop_size, 0)
        height_to_crop = max(h - self.crop_size, 0)
        width_to_crop = width_to_crop if height_to_crop != 0 else max(w - h, 0)
        height_to_crop = height_to_crop if width_to_crop != 0 else max(h - w, 0)
        
        left = 0 + width_to_crop // 2
        top = 0 + height_to_crop // 2
        right = w - (width_to_crop - left)
        bottom = h - (height_to_crop - top)
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.target_size, self.target_size), self.interpolation)
        return image


def extract_color_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract color channels from BGR image.
    
    Args:
        image: BGR image as numpy array
        
    Returns:
        Tuple of (grayscale, blue, green, red) channels as uint8 arrays
    """
    b, g, r = cv2.split(image)
    k = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return k.astype("uint8"), b.astype("uint8"), g.astype("uint8"), r.astype("uint8")


def undersample(good_items: List, bad_items: List) -> Tuple[List, List]:
    """
    Create two lists of the same length by removing items from the longer list.
    
    Args:
        good_items: first list
        bad_items: second list
        
    Returns:
        Tuple of (good_items, bad_items) with equal length
    """
    n_good = len(good_items)
    n_bad = len(bad_items)
    
    if n_good > n_bad:
        return good_items[:n_bad], bad_items
    elif n_good < n_bad:
        return good_items, bad_items[:n_good]
    else:
        return good_items, bad_items


def oversample(good_items: List, bad_items: List) -> Tuple[List, List]:
    """
    Create two lists of the same length by randomly duplicating items from the shorter list.
    
    Args:
        good_items: first list
        bad_items: second list
        
    Returns:
        Tuple of (good_items, bad_items) with equal length
    """
    n_good = len(good_items)
    n_bad = len(bad_items)
    
    print(f"{n_good} good flakes")
    print(f"{n_bad} bad flakes")
    
    if n_good > n_bad:
        idxs = np.random.choice(range(n_bad), n_good - n_bad)
        bad_items = bad_items + [bad_items[i] for i in idxs]
        return good_items, bad_items
    elif n_good < n_bad:
        idxs = np.random.choice(range(n_good), n_bad - n_good)
        good_items = good_items + [good_items[i] for i in idxs]
        return good_items, bad_items
    else:
        return good_items, bad_items


def load_images_from_directory(images_dir: str, loader_transform: CenterCrop, 
                             f_split: float = 0.8, sampling_mode: str = 'over') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images from a directory with "Good" and "Bad" subdirectories into training and validation sets.
    
    Args:
        images_dir: directory containing Good/ and Bad/ subdirectories
        loader_transform: transform to apply to images after loading
        f_split: fraction of data to use for training
        sampling_mode: 'under', 'over', or None for class balancing
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    good_images_dir = os.path.join(images_dir, 'Good')
    bad_images_dir = os.path.join(images_dir, 'Bad')
    
    if not os.path.exists(good_images_dir) or not os.path.exists(bad_images_dir):
        raise ValueError(f"Directory must contain 'Good' and 'Bad' subdirectories. Found: {os.listdir(images_dir)}")
    
    good_images = os.listdir(good_images_dir)
    bad_images = os.listdir(bad_images_dir)
    random.shuffle(good_images)
    random.shuffle(bad_images)
    
    # Split into train/val
    n_train_good = int(np.floor(f_split * len(good_images)))
    good_images_train = good_images[:n_train_good]
    good_images_val = good_images[n_train_good:]
    
    n_train_bad = int(np.floor(f_split * len(bad_images)))
    bad_images_train = bad_images[:n_train_bad]
    bad_images_val = bad_images[n_train_bad:]
    
    # Apply sampling strategy
    if sampling_mode == 'under':
        good_images_train, bad_images_train = undersample(good_images_train, bad_images_train)
        good_images_val, bad_images_val = undersample(good_images_val, bad_images_val)
    elif sampling_mode == 'over':
        print("Training set:")
        good_images_train, bad_images_train = oversample(good_images_train, bad_images_train)
        print("Validation set:")
        good_images_val, bad_images_val = oversample(good_images_val, bad_images_val)
    
    # Load training images
    X_train = []
    y_train = []
    
    for image_name in good_images_train:
        image_path = os.path.join(good_images_dir, image_name)
        image = Image.open(image_path)
        image = loader_transform(image)
        X_train.append(np.array(image))
        y_train.append([1.0])
        
    for image_name in bad_images_train:
        image_path = os.path.join(bad_images_dir, image_name)
        image = Image.open(image_path)
        image = loader_transform(image)
        X_train.append(np.array(image))
        y_train.append([0.0])
    
    # Load validation images
    X_val = []
    y_val = []
    
    for image_name in good_images_val:
        image_path = os.path.join(good_images_dir, image_name)
        image = Image.open(image_path)
        image = loader_transform(image)
        X_val.append(np.array(image))
        y_val.append([1.0])
        
    for image_name in bad_images_val:
        image_path = os.path.join(bad_images_dir, image_name)
        image = Image.open(image_path)
        image = loader_transform(image)
        X_val.append(np.array(image))
        y_val.append([0.0])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype='float32')
    X_val = np.array(X_val)
    y_val = np.array(y_val, dtype='float32')
    
    return X_train, y_train, X_val, y_val