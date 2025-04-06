Sure! Here's an improved version of your README that refines the content and structure, making it clearer and more engaging for potential collaborators or users:

---

# Grayscale to Color Image Conversion Using Autoencoder (AE)

## Overview
This project implements a deep learning model to convert grayscale images to color images using an advanced Autoencoder architecture. The model is enhanced with Transformer blocks and skip connections, leveraging both convolutional and attention-based mechanisms to generate realistic colorized outputs. The model is trained on a dataset of grayscale images and their corresponding color images, aiming to learn the mapping between the two and produce high-quality colorized images.

## Project Structure

```plaintext
Gray2Color_AE
├── src
│   ├── dataloader.py       # Functions for loading and preprocessing images
│   ├── model.py            # Model architecture definition (Autoencoder with Transformer)
│   ├── train.py            # Training script for the model
│   ├── infer.py            # Inference script to generate colorized images
│   └── utils
│       └── __init__.py     # Utility functions (if any)
├── config.json             # Configuration settings for paths and parameters
├── requirements.txt        # List of required Python packages
├── README.md               # Project documentation
└── gray2color-ae.ipynb     # Jupyter notebook for original code and experimentation
```

## Installation
To get started with this project, make sure you have Python 3.x installed. Then, install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies, including TensorFlow, Keras, NumPy, and other libraries for training and running the model.

## Configuration
The project relies on the `config.json` file for managing dataset paths and model parameters. Update this file with the appropriate paths for your dataset and adjust any model parameters to suit your needs.

## Usage

### 1. **Data Loading and Preprocessing**
The `dataloader.py` script handles the loading and preprocessing of the dataset. The `load_images` function takes the paths from `config.json`, loads the grayscale and color images, resizes them to the target size, and normalizes the pixel values.

### 2. **Model Training**
To train the Autoencoder model, run the `train.py` script:

```bash
python src/train.py
```

This script will load the training and test data, define the Autoencoder model with Transformer blocks, and then train the model using Mean Squared Error loss and PSNR/SSIM metrics. The trained model will be saved to the path specified in `config.json`.

### 3. **Inference**
Once the model is trained, you can generate colorized images using the `infer.py` script. This script loads the trained model and predicts colorized outputs from grayscale images. It also evaluates the model using PSNR and SSIM metrics:

```bash
python src/infer.py
```

You can specify the path to the input grayscale image and the path to save the output colorized image in the script.

## Model Performance
The Autoencoder model has shown promising results in converting grayscale images to colorized versions. Below are some example metrics from the evaluation:

- **PSNR (Peak Signal-to-Noise Ratio):** 25.03 dB
- **SSIM (Structural Similarity Index):** 0.9418

These values indicate good image quality and high similarity between the colorized and original color images.

## Results
![image](https://github.com/user-attachments/assets/07cbf16d-9411-48df-b42f-bf8d5d402472)

![image](https://github.com/user-attachments/assets/3b2ccb5f-a467-45b6-9078-6acfb570540c)

## Acknowledgments
This project is inspired by recent advancements in deep learning for image processing, especially in the area of image colorization. It aims to contribute to the field by providing a simple yet effective implementation of an Autoencoder-based architecture augmented with Transformers for improved colorization quality.
