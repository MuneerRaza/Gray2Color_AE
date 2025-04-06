# Grayscale to Color Image Conversion Using Autoencoder (AE)

## Overview
This project implements a deep learning model to convert grayscale images to color images using an Autoencoder architecture enhanced with Transformer blocks and skip connections. The model is trained on a dataset of grayscale and corresponding color images, aiming to generate realistic colorized outputs.

## Project Structure
```
Gray2Color_AE
├── src
│   ├── dataloader.py       # Functions for loading and preprocessing images
│   ├── model.py            # Model architecture definition
│   ├── train.py            # Training code for the model
│   ├── infer.py            # Inference code for predicting colorized images
│   └── utils
│       └── __init__.py     # Utility functions (if any)
├── config.json             # Configuration settings for paths and parameters
├── README.md                # Project documentation
└── gray2color-ae.ipynb     # Original code and experiments
```

## Installation
To set up the project, ensure you have the required libraries installed. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Configuration
The project uses a `config.json` file to manage paths and parameters. Update this file with the appropriate paths for your dataset and any model parameters you wish to modify.

## Usage
1. **Data Loading and Preprocessing**: Use the `dataloader.py` file to load and preprocess your dataset. The `load_images` function will handle loading images from the specified paths in `config.json`.

2. **Model Training**: Run the `train.py` file to compile and train the Autoencoder model. This file imports the necessary functions from `dataloader.py` and `model.py`.

3. **Inference**: After training, use the `infer.py` file to load the trained model and predict colorized images from grayscale inputs. This file also evaluates the model's performance using PSNR and SSIM metrics.

## Model Performance
The Autoencoder model has shown effectiveness in generating realistic colorized images. Metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) are used to evaluate the quality of the generated images.

## Future Work
Potential improvements include experimenting with advanced Autoencoder architectures, incorporating adversarial loss for better colorization, and implementing data augmentation techniques to enhance model robustness.

## Acknowledgments
This project is inspired by recent advancements in deep learning for image processing and aims to contribute to the field of automatic image colorization.