import json
import tensorflow as tf
from keras.optimizers import Adam
from dataloader import load_images
from model import autoencoder_with_transformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load configuration from config.json
with open('../config.json') as config_file:
    config = json.load(config_file)

# Load and preprocess images
print("Loading and preprocessing images...")
gray_images = load_images(config['gray_path'], color_mode='grayscale', target_size=(128, 128)) / 255.0
color_images = load_images(config['color_path'], color_mode='rgb', target_size=(128, 128)) / 255.0

# Remove extra dimensions for grayscale images
gray_images = gray_images.reshape(-1, 128, 128, 1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(gray_images, color_images, test_size=config['test_size'], random_state=42)

# Instantiate and compile the model
print("Building and compiling the model...")
model = autoencoder_with_transformer(input_shape=(config["image_size"][0], config["image_size"][1], 1), num_classes=3)
model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mean_squared_error')

# Train the model
print("Training the model...")
history = model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=config['epochs'], validation_data=(x_test, y_test))
print("Training completed.")
# Save the trained model
model.save(config['model_path'])

# evaluate the model
print("Evaluating the model...")
psnr = []
ssim = []
# Evaluate model with progress bar
for i in tqdm(range(len(x_test)), desc="Evaluating images"):
    pred = model.predict(x_test[i].reshape(1, 128, 128, 1), verbose=0)
    psnr.append(tf.image.psnr(y_test[i], pred[0], max_val=1.0).numpy())
    ssim.append(tf.image.ssim(y_test[i], pred[0], max_val=1.0).numpy())
print(f"Average PSNR: {sum(psnr) / len(psnr)}")
print(f"Average SSIM: {sum(ssim) / len(ssim)}")

