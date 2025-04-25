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
│   ├── check_lines
│   ├── gif_frames
│   ├── outputs
│   ├── 01_explore_carvis.ipynb
│   ├── 02_prototype_autoencoder.ipynb
│   ├── 03_latent_analysis.ipynb
│   ├── 04_improve_autoencoder.ipynb
│   ├── 05_metadata_analysis.ipynb
│   ├── 06_SNR_mitigate.ipynb
│   ├── 07_ha_activity_lomb_analysis.ipynb
│   └── 08_full_pipeline.ipynb
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
│   │   ├── vis_a_files_all.txt  
│   │   └── vis_a_files.txt
│   └── CARM_VIS_M15.mas           # the mask
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

- **Data Preparation**: 
  - Multi-order spectra (orders 25, 46, 47) loaded and normalized.
  - Order 25 isolated (Hα region cropped).
  - SNR outliers filtered (±3σ) and six low-quality observations manually excluded (final set: 95 spectra).
  - Optional resolution reduction implemented for cropped region.

- **Autoencoder Architecture**:
  - Conv1D-based, flexible design:
    - Dropout layers (encoder/decoder/latent) optional and tunable.
    - BatchNormalization after first Conv1D.
    - Latent normalization (LayerNorm) optional.
    - Activity regularization (L1) configurable after each layer.
    - Contractive loss penalty (optional) on bottleneck layer.
    - Tunable activation functions and depth.

- **Denoising & SNR Bias Mitigation**:
  - Gaussian noise injection option (configurable std).
  - SNR-weighted MSE loss.
  - Latent–SNR decorrelation penalty (partial decorrelation configurable).
  - Latent space normalization.
  - Early stopping based on reconstruction loss.
  - Batch size optimization (8 found optimal for decorrelation and training stability).

- **Evaluation**:
  - Residual latent–SNR correlations strongly reduced.
  - UMAP projections no longer structured by SNR.
  - Lomb-Scargle periodograms applied:
    - Latents show strong signals close to the stellar rotation period.
    - False Alarm Probabilities (FAP) calculated.
    - Second harmonics checked.
  - HALPHA, RV, FWHM, BIS: independent periodogram analysis performed.
  - GIF animations generated to visualize temporal evolution of spectral reconstructions.

- **Regression on Latent Space**:
  - HALPHA prediction from latent vectors (R² ≈ 0.093, promising given data limits).

---

## Next Steps

- **Spectral Expansion**:
  - Extend analysis beyond order 25.
  - Include multiple orders while preserving focus on activity indicators (Hα, Ca II, etc.).

- **Resolution Experiments**:
  - Train models at reduced spectral resolution and assess impact on latent interpretability and robustness.

- **Architecture Improvements**:
  - Test deeper encoder/decoder structures.
  - Add skip connections to improve spectral reconstructions.
  - Explore structured latent regularization (e.g., disentanglement, variational techniques).

- **Activity Indicator Prediction**:
  - Build decoder-free regression models: 
    - Freeze encoder.
    - Train neural networks to predict RV, BIS, FWHM, CONTRAST, and HALPHA from latent vectors.
    - Evaluate generalization.

- **Lomb-Scargle Refinement**:
  - Refine periodogram analysis:
    - Focus on significant latent modes (FAP < 0.01).
    - Track stability across different architectures.

- **Documentation & Reporting**:
  - Update README and weekly reports.
  - Organize visual outputs (periodograms, UMAPs, reconstructions) into a clean analysis notebook.
  - Start preparing polished plots for final project reporting.

---

## Contact

Developed by Marvin Ernst  
Barcelona School of Economics – MSc in Data Science Methodology  
Project supervised by Dr. Manuel Perger

---
