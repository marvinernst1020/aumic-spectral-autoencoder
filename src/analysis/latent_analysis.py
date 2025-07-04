import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from astropy.timeseries import LombScargle
import umap
import os

def plot_umap(z_mean, color_values, title, color_label, save_path=None):
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(z_mean)

    plt.figure(figsize=(6, 5))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=color_values, cmap="viridis")
    plt.title(title)
    plt.colorbar(label=color_label)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

def plot_pca(z_mean, color_values, title, color_label, save_path=None):
    latent_pca = PCA(n_components=2).fit_transform(z_mean)
    plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=color_values, cmap="viridis")
    plt.colorbar(label=color_label)
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

def plot_tsne(z_mean, color_values, title, color_label, save_path=None):
    tsne = TSNE(n_components=2, perplexity=min(30, len(z_mean) // 2))
    latent_tsne = tsne.fit_transform(z_mean)
    plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=color_values, cmap="viridis")
    plt.colorbar(label=color_label)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()

def plot_lomb_scargle(z_mean, time, output_dir, rotation_period=4.8367):
    frequencies = np.linspace(1/100, 0.5, 5000)
    rotation_freq = 1 / rotation_period
    os.makedirs(output_dir, exist_ok=True)

    for i in range(z_mean.shape[1]):
        y = z_mean[:, i] - np.mean(z_mean[:, i])
        ls = LombScargle(time, y)
        power = ls.power(frequencies)
        fap_01 = ls.false_alarm_level(0.001)
        fap_1 = ls.false_alarm_level(0.01)
        fap_10 = ls.false_alarm_level(0.1)
        best_freq = frequencies[np.argmax(power)]
        best_period = 1 / best_freq

        plt.figure(figsize=(8, 5))
        plt.plot(frequencies, power, label=f"Latent dim {i}", color='black')
        plt.axhline(fap_01, linestyle='--', color='gray', label='0.1% FAP')
        plt.axhline(fap_1, linestyle='--', color='gray', label='1% FAP')
        plt.axhline(fap_10, linestyle='--', color='gray', label='10% FAP')
        plt.axvline(rotation_freq, color='orange', linestyle='--', label='Rotation')
        plt.axvline(2 * rotation_freq, color='orange', linestyle='--', label='2x Rotation')
        plt.axvline(best_freq, color='red', linestyle='--', label=f'Peak: {best_freq:.3f}')
        plt.xlabel("Frequency [1/day]")
        plt.ylabel("Power")
        plt.title(f"Lomb-Scargle of Latent Dimension {i}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dimension_{i}.png"), dpi=150)
        plt.show()
        plt.close()
        print(f"Latent dim {i}: Best period = {best_period:.3f} days")

def phase_fold(time, signal, period):
    phase = (time % period) / period
    sorted_indices = np.argsort(phase)
    return phase[sorted_indices], signal[sorted_indices]

def plot_phase_folded_latent_dims(z_mean, bjd, output_dir, n_bins=100):
    os.makedirs(output_dir, exist_ok=True)
    frequencies = np.linspace(1/100, 0.5, 5000)

    for i in range(z_mean.shape[1]):
        signal = z_mean[:, i] - np.mean(z_mean[:, i])
        ls = LombScargle(bjd, signal)
        power = ls.power(frequencies)
        best_freq = frequencies[np.argmax(power)]
        best_period = 1 / best_freq

        phase, folded = phase_fold(bjd, signal, best_period)

        plt.figure(figsize=(8, 4))
        plt.plot(phase, folded, 'o', markersize=3, alpha=0.6)
        plt.xlabel(f"Phase (P = {best_period:.2f} d)")
        plt.ylabel(f"Latent dim {i}")
        plt.title(f"Phase-folded Latent Dimension {i}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"latent_dim_{i}_phasefold.png"), dpi=150)
        plt.show()
        plt.close()

    print(f"Phase-folded plots saved to: {output_dir}")
