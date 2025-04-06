import json
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img


# Load configuration from config.json
def load_config():
    with open('../config.json') as config_file:
        config = json.load(config_file)
    return config

# Load and preprocess a single image for inference
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img = img_to_array(img) / 255.0  # Normalize
    img = img.reshape(-1, 128, 128, 1)  # Reshape for model input
    return img


# Main inference function
def infer(model_path, image_path):
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Predict the colorized image
    predicted_image = model.predict(input_image)

    psnr = tf.image.psnr(predicted_image, input_image, max_val=1.0).numpy()
    ssim = tf.image.ssim(predicted_image, input_image, max_val=1.0).numpy()
    print(f"PSNR: {psnr}")
    print(f"SSIM: {ssim}")

    return predicted_image

if __name__ == "__main__":
    config = load_config()
    model_path = config['model_path']
    image_path = config['test_image_path']

    # Perform inference
    colorized_image = infer(model_path, image_path)

    save_path = config['output_image_path']
    # Save the colorized image
    colorized_image = (colorized_image[0] * 255).astype(np.uint8)
    tf.keras.preprocessing.image.save_img(save_path, colorized_image)
    print(f"Colorized image saved to {save_path}")