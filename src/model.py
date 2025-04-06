import tensorflow as tf
from tensorflow.keras import layers, models

def autoencoder_with_transformer(input_shape=(128, 128, 1), num_classes=3):
    inputs = layers.Input(shape=input_shape)

    # Encoder (Contracting Path)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(2)(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(2)(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(2)(conv3)

    # Bottleneck with Transformer block
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    bottleneck = layers.Conv2D(512, 3, activation='relu', padding='same')(bottleneck)

    # Reshape to (batch_size, height * width, channels) for Transformer
    bottleneck_shape = bottleneck.shape
    bottleneck_flattened = layers.Reshape((-1, bottleneck_shape[-1]))(bottleneck)

    # Apply the Transformer block
    transformer = TransformerBlock(dim=512, heads=8, ff_dim=1024)(bottleneck_flattened)

    # Reshape back to the spatial dimensions
    transformer_reshaped = layers.Reshape((bottleneck_shape[1], bottleneck_shape[2], 512))(transformer)

    # Decoder (Expanding Path) with Skip Connections
    up3 = layers.UpSampling2D(2)(transformer_reshaped)
    concat3 = layers.Concatenate()([up3, conv3])
    deconv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat3)
    deconv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(deconv3)

    up2 = layers.UpSampling2D(2)(deconv3)
    concat2 = layers.Concatenate()([up2, conv2])
    deconv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
    deconv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(deconv2)

    up1 = layers.UpSampling2D(2)(deconv2)
    concat1 = layers.Concatenate()([up1, conv1])
    deconv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    deconv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(deconv1)

    # Output layer (Using Sigmoid activation for output in AE)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same')(deconv1)

    # Create model
    model = models.Model(inputs, outputs)
    
    return model

class TransformerBlock(layers.Layer):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=dim, dropout=dropout)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x):
        attn_output = self.attention(x, x)  # Self-attention
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2