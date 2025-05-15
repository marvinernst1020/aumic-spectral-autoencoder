#!/usr/bin/env python

"""
Collection of artificial neural networks
----------------------------------------------------------
2024
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
The models must to be compilated outside of these classes
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras as K
from tensorflow.keras.layers import (
    Conv1D, Reshape, Conv2D, Flatten, Dense, Lambda,
    Conv1DTranspose, Layer, MaxPooling2D, Input, Dropout,
    BatchNormalization, Activation, Concatenate, AveragePooling1D, UpSampling1D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import UnivariateSpline  # For spline-like activation functions
from spectrumnet.utils.logger_config import logger  # Ensures correct module import
from keras.saving import register_keras_serializable # For custom serialization
logger.info("VAE model file imported.")


@register_keras_serializable()
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(z_log_sigma) * epsilon


@register_keras_serializable()
class MCDropout(Layer):
    """
    Monte Carlo Dropout layer to enable stochastic behavior during inference.

    Args:
        rate (float): Dropout rate (probability of dropping a unit).
        is_disabled (bool, optional): If True, disables dropout (default: False).
        noise_shape (tuple, optional): Shape of noise mask (default: None).
        name (str, optional): Name of the layer.

    Methods:
        call(inputs, training): Applies dropout during training (and optionally inference).
        get_config(): Returns the layer's configuration dictionary.
    """

    def __init__(
            self, rate: float, is_disabled: bool = False,
            noise_shape: tuple = None, name: str = None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.is_disabled = is_disabled
        self.noise_shape = noise_shape
        logger.info(f"MCDropout initialized with rate={self.rate}, is_disabled={self.is_disabled}")

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Apply dropout unless disabled."""
        if self.is_disabled:
            return inputs
        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape)

    def get_config(self) -> dict:
        """Returns the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'is_disabled': self.is_disabled,
            'noise_shape': self.noise_shape,
        })
        return config

class SuperVAE:
    """
    Base class for Variational Autoencoders (VAEs).

    Args:
        n_inputs (int): Input data dimension.
        latent_dim (int): Latent space dimension.
        dropout (float): Dropout rate.
        mcdropout (bool): Enables Monte Carlo Dropout.
    """

    def __init__(self, latent_dim, dropout=0.2, actfn='tanh', mcdropout=True):
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.mcdropout = mcdropout
        self.actfn = actfn

        logger.info(f"TensorFlow Version: {tf.__version__}")
        logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")


    def mcdo_predict(self, testset, encoder, decoder, mc_dropout_num=50):
        """Monte Carlo Dropout prediction for VAEs."""
        predictions_enc = np.array([encoder(testset, training=True)[0] for _ in range(mc_dropout_num)])
        predictions_dec = np.array([decoder(predictions_enc[i], training=True) for i in range(mc_dropout_num)])

        return {
            'mean_encoder': np.mean(predictions_enc, axis=0),
            'std_encoder': np.std(predictions_enc, axis=0),
            'mean': np.mean(predictions_dec, axis=0),
            'std': np.std(predictions_dec, axis=0)
        }

    def load_model(self, model_name):
        """Loads a trained VAE model with necessary custom layers."""
        custom_objects = {'MCDropout': MCDropout, 'sampling': self.sampling}
        return tf.keras.models.load_model(model_name, custom_objects=custom_objects)

    def model_tf(self):
        """This method should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")


class SpecVaeCnn1D(SuperVAE):
    """Flexible 1D Convolutional Variational Autoencoder (VAE)"""

    def __init__(self, n_inputs, conv_layers=None, dense_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        # Flexible convolutional layer setup
        self.conv_layers = conv_layers if conv_layers is not None else [(128, 5), (256, 5)]

        # Flexible dense layer setup
        self.dense_layers = dense_layers if dense_layers is not None else [512, 256]

    def model_tf(self):
        """Builds the flexible 1D CNN VAE model dynamically based on user-defined layers."""
        logger.info("Building flexible 1D CNN VAE...")

        # ===== ENCODER =====
        inputs = K.Input(shape=(self.n_inputs,))
        x = Reshape((self.n_inputs, 1))(inputs)

        # Apply Conv + Pooling layers
        for filters, kernel_size in self.conv_layers:
            x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation=self.actfn)(x)
            x = AveragePooling1D(pool_size=5)(x)
            x = BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        shape_after_pooling = K.backend.int_shape(x)[1:]  # (timesteps, features)
        x = Flatten()(x)

        for neurons in self.dense_layers:
            x = Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        z_mean = Dense(self.latent_dim)(x)
        z_log_sigma = Dense(self.latent_dim)(x)
        
        # KL Divergence Loss Layer
        z_mean = KLDivergenceLayer(beta=1e-3)([z_mean, z_log_sigma])
        z = Lambda(sampling)([z_mean, z_log_sigma])

        encoder = K.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        logger.info(encoder.summary())

        # ===== DECODER =====
        latent_inputs = K.Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs

        for neurons in reversed(self.dense_layers):
            x = Dense(neurons, activation=self.actfn)(x)
            x = MCDropout(self.dropout)(x)

        x = Dense(np.prod(shape_after_pooling), activation=self.actfn)(x)
        x = Reshape(shape_after_pooling)(x)

        # Reverse convolutional layers
        for filters, kernel_size in reversed(self.conv_layers):
            x = UpSampling1D(size=5)(x)  # Mirror of AveragePooling1D(pool_size=5)
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=self.actfn, padding='same')(x)
            x = BatchNormalization()(x)
            x = MCDropout(self.dropout)(x)

        x = Conv1D(1, 3, activation=self.actfn, padding="same")(x)
        x = MCDropout(self.dropout)(x)
    
        x = Lambda(lambda t: resize_1d_tensor(t, self.n_inputs),
            output_shape=(self.n_inputs, 1))(x)
        decoder_outputs = Reshape((self.n_inputs,), name="decoder_output")(x)

        # decoder_outputs = Reshape((self.n_inputs,))(x)

        decoder = K.Model(latent_inputs, decoder_outputs, name='decoder')
        logger.info(decoder.summary())

        # Get latent outputs from encoder
        z_mean, z_log_sigma, z_sample = encoder(inputs)

        # Decoder output
        z_out = decoder(z_sample)

        # Final VAE model (encoder → decoder)
        vae = K.Model(inputs, z_out, name='vae')

        # Add KL divergence loss manually
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma))
        vae.add_loss(1e-3 * kl_loss)

        # Add reconstruction loss manually
        recon_loss = tf.reduce_mean(tf.square(inputs - z_out))
        vae.add_loss(recon_loss)

        # Save references for convenience
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder

        return vae, encoder, decoder

    
# Saving the models:
def save_all_models(vae, encoder, decoder, path_prefix):
    vae.save(f"{path_prefix}_fullmodel.keras")
    encoder.save(f"{path_prefix}_encoder.keras")
    decoder.save(f"{path_prefix}_decoder.keras")
    print(f"Models saved to {path_prefix}_*.keras")


@register_keras_serializable()
class KLDivergenceLayer(K.layers.Layer):
    """
    Custom Keras layer to compute and add the KL divergence loss.
    Compatible with Functional API and symbolic KerasTensors.
    """
    def __init__(self, beta=1.0, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.beta = beta  # scaling factor for KL loss

    def call(self, inputs):
        z_mean, z_log_sigma = inputs

        # Compute KL divergence loss
        kl = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        kl = tf.reduce_sum(kl, axis=-1)
        kl_loss = -0.5 * self.beta * tf.reduce_mean(kl)

        # Register KL loss with model
        self.add_loss(kl_loss)

        return z_mean  # or return inputs[0] if you only need it passed through

@register_keras_serializable()
class ReconstructionLossLayer(K.layers.Layer):
    def call(self, inputs):
        x_true, x_pred = inputs
        loss = tf.reduce_mean(tf.square(x_true - x_pred), axis=-1)
        self.add_loss(tf.reduce_mean(loss))
        return x_pred  # passthrough
    

def resize_1d_tensor(t, target_len):
    # Input: (batch, time, channels)
    t = tf.expand_dims(t, axis=1)  # → (batch, 1, time, channels)
    t = tf.image.resize(t, size=[1, target_len], method='bilinear')  # Valid 2D resize
    t = tf.squeeze(t, axis=1)  # → (batch, time, channels)
    return t

