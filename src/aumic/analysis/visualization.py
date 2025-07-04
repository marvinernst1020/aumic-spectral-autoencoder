import os
import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(wavelength, original, reconstructed, save_path=None, mask_lines=None):
    plt.figure(figsize=(10, 4))
    plt.plot(wavelength, original, label="Original", color="black")
    plt.plot(wavelength, reconstructed, label="Reconstructed", color="orange", alpha=0.8)

    if mask_lines is not None:
        for wl in mask_lines:
            if wavelength[0] <= wl <= wavelength[-1]:
                plt.axvline(wl, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.title("Original vs. Reconstructed Spectrum")
    plt.xlabel("Wavelength [\u00c5]")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.show()
    plt.close()


def plot_order_reconstructions(wave_data, original, reconstructed, valid_orders, i=0, save_dir=None, mask_lines=None):
    for j, order in enumerate(valid_orders):
        wavelength = wave_data[order]
        start = sum(wave_data[o].shape[0] for o in valid_orders[:j])
        end = start + wavelength.shape[0]

        orig = original[i, start:end]
        recon = reconstructed[i, start:end]

        plt.figure(figsize=(8, 3))
        plt.plot(wavelength, orig, label="Original", color="black")
        plt.plot(wavelength, recon, label="Reconstructed", color="orange", alpha=0.8)

        if mask_lines is not None:
            for wl in mask_lines:
                if wavelength[0] <= wl <= wavelength[-1]:
                    plt.axvline(wl, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

        plt.title(f"Order {order} – Spectrum {i}")
        plt.xlabel("Wavelength [Å]")
        plt.ylabel("Normalized Flux")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/order_{order}_spec_{i}.png")

        plt.show()
        plt.close()


def plot_training_loss(history, save_path=None):
    plt.plot(history.history['loss'], label='Train')
    plt.title("VAE Training Loss (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()