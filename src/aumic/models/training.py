import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import numpy as np

def build_callbacks(beta_var, max_beta=1e-4, n_epochs=100, patience=20, warmup=True):
    if warmup:
        class KLAnnealing(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                new_beta = min(max_beta, max_beta * (epoch + 1) / n_epochs)
                beta_var.assign(new_beta)
                print(f"Epoch {epoch+1}: beta = {new_beta:.6f}")
        anneal_cb = KLAnnealing()
    else:
        class BetaAnnealingCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                new_beta = min(max_beta, max_beta * (epoch / n_epochs))
                beta_var.assign(new_beta)
                print(f"Epoch {epoch+1}: beta = {new_beta:.6f}")
        anneal_cb = BetaAnnealingCallback()

    return [
        anneal_cb,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=patience)
    ]

def compute_noise_weights(wavelengths, mask_lines, sigma=0.05, min_weight=0.2):
    weights = np.ones_like(wavelengths)
    for i, wl in enumerate(wavelengths):
        min_dist = np.min(np.abs(mask_lines - wl))
        decay = 1 - np.exp(-(min_dist ** 2) / (2 * sigma ** 2))
        weights[i] = min_weight + (1 - min_weight) * decay
    return weights

def augment_with_noise(X, factor=10, noise_std=0.05, 
                       wavelengths=None, mask_lines=None, 
                       sigma_protect=0.05, protect_mask_lines=False,
                       min_weight=0.2):
    augmented = [X]
    for _ in range(factor):
        if protect_mask_lines and wavelengths is not None and mask_lines is not None:
            weights = compute_noise_weights(wavelengths, mask_lines, sigma=sigma_protect, min_weight=min_weight)
            noise_matrix = np.random.normal(0, noise_std, size=X.shape)
            noise = noise_matrix * weights[None, :]
        else:
            noise = np.random.normal(0, noise_std, size=X.shape)
        augmented.append(X + noise)
    return np.vstack(augmented)


