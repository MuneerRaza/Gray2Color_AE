import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

# Load configuration from config.json
def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# Load Dataset
def load_images(path, target_size=(128, 128), color_mode='rgb'):
    images = []
    for img_name in tqdm(os.listdir(path), desc=f"Loading {color_mode} images"):
        img_path = os.path.join(path, img_name)
        img = load_img(img_path, target_size=target_size, color_mode=color_mode)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# Load and preprocess images
def load_and_preprocess_images(config):
    gray_images = load_images(config['gray_path'], color_mode='grayscale', target_size=(128, 128)) / 255.0
    color_images = load_images(config['color_path'], color_mode='rgb', target_size=(128, 128)) / 255.0

    gray_images = gray_images.reshape(-1, 128, 128, 1)

    x_train, x_test, y_train, y_test = train_test_split(gray_images, color_images, test_size=config['test_size'], random_state=config['random_state'])
    
    return x_train, x_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    config = load_config()
    x_train, x_test, y_train, y_test = load_and_preprocess_images(config)
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")