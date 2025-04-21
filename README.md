# aumicAE: Spectral Variability Encoding of AU Microscopii

This project applies deep learning techniques to **high-resolution optical spectra** of the active M-dwarf star **AU Microscopii**, using data from the **CARMENES VIS_A channel**. The aim is to learn a compressed latent representation of the star's spectral variability — caused by magnetic activity — using convolutional autoencoders.

The project is inspired by the methodology from:
> Mas-Buitrago et al. (2024), *A&A 687, A205*  
> [https://doi.org/10.1051/0004-6361/202449865](https://doi.org/10.1051/0004-6361/202449865)

However, while that paper estimates stellar parameters across multiple stars, this project focuses on the **spectral time series of a single star**, using the latent space to analyze temporal variability due to surface activity.

---

## Project Goals

- Load and preprocess high-resolution AU Mic spectra (VIS_A arm)
- Normalize and stack consistent spectral orders
- Train 1D convolutional autoencoders (standard and denoising) to learn compressed representations
- Track changes in the latent space over time
- Correlate latent parameters with stellar activity (e.g., flares, line indices)

---

## Directory Structure

```plaintext
aumicAE/ 
├── notebooks/
│   ├── 01_explore_carvis.ipynb
│   ├── 02_prototype_autoencoder.ipynb
│   ├── 03_latent_analysis.ipynb
│   ├── 04_improve_autoencoder.ipynb
│   ├── 05_metadata_analysis.ipynb
│   ├── 06_SNR_mitigate.ipynb
│   └── 07_full_pipeline.ipynb
├── src/
│   ├── data/
│   │   └── preprocess.py          # load_spectrum, normalize_orders
│   ├── models/
│   │   └── autoencoder.py         # build_autoencoder(), build_denoising_ae()
│   ├── train.py                   # train_autoencoder(cfg), train_denoising(cfg)
│   └── utils/
│       └── visualization.py       # plot_latent_evolution, plot_denoising
├── configs/
│   └── base.yaml                  # orders: [25, 46, 47], latent_dim: 8, ...
├── data/
│   ├── carvis_visA/
│       └── vis_a_files.txt
├── outputs/
│   ├── latent/
│   ├── models/
│   ├── plots/
│   ├── logs/
│   └── preprocessed/
├── requirements.txt
└── README.md
```

Note: Large data files are stored in aumicAE1/ due to repository size limits.

---

## Environment Setup

- Python 3.11
- TensorFlow ≥ 2.13
- astropy, numpy, matplotlib, etc. (see *requirements.txt*)

---

## Status

- **Data Preparation**: Multi-order spectra (orders 25, 46, 47) loaded and normalized. SNR outliers filtered (±3σ), six low-quality observations excluded, final set: 95 spectra.

- **Autoencoder Architecture**: Conv1D autoencoder with modular components:
  - Optional dropout (encoder/decoder/latent)
  - BatchNormalization after first Conv1D
  - Latent normalization
  - Configurable depth and activation functions

- **Denoising & SNR Bias Mitigation**:
  - Gaussian noise injection (optional, tunable std)
  - SNR-weighted MSE loss
  - SNR decorrelation penalty (optional, partial, configurable α and dimension subset)
  - Latent vector normalization post-training
  - Early stopping based on reconstruction loss plateau

- **Training Pipeline**:
  - Fully modular with options for noise, decorrelation, weighting, dropout, activation, batch size, prefetch, epochs, and stopping criteria.
  - Batch size of 8 found optimal for stability and SNR decorrelation.

- **Evaluation**:
  - Residual latent–SNR correlations: greatly reduced (Pearson & Spearman)
  - UMAP projection: no longer structured by SNR
  - Latent–activity correlation: RV, FWHM, BIS, CONTRAST, HALPHA show interpretable structure
  - Regression on latent space to predict HALPHA shows promise (R² ≈ 0.093)

---

## Next Steps

- **Full Spectral Order Integration**: Expand training to additional orders beyond 25, 46, 47 to capture full activity information.

- **Pipeline Scaling**: Build end-to-end workflow: order selection → autoencoder training → decorrelation → activity prediction.

- **Decoder-Free Regression Models**: Freeze encoder, use latent space as input to predict activity indicators directly.

- **Architectural Refinement**:
  - Explore deeper encoder/decoder
  - Skip connections
  - More structured latent space constraints (e.g., disentanglement)

- **Resolution Experiments**: Investigate reduced-resolution spectra to test generalization and robustness.

- **Documentation & Reporting**:
  - Maintain visual diagnostics (correlation matrices, UMAPs, periodograms)
  - Continue weekly reporting and visual inspection of residual SNR traces

---

## Contact

Developed by Marvin Ernst  
Barcelona School of Economics – MSc in Data Science Methodology  
Project supervised by Dr. Manuel Perger

---
