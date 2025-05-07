# aumicAE: Spectral Variability Encoding of AU Microscopii

This project applies deep learning techniques to **high-resolution optical spectra** of the active M-dwarf star **AU Microscopii**, using data from the **CARMENES VIS_A channel**. The aim is to learn a compressed latent representation of the star's *spectral variability* — caused by *magnetic activity* — using convolutional autoencoders. The analysis focuses both on full-order spectra and targeted spectral regions, particularly the **asymmetry and variability of chromospheric lines** such as H \alpha and other key lines defined in the mask.

---

## Project Goals

- Preprocess high-resolution CARMENES spectra (VIS\_A arm), ensuring consistent normalization
- Train 1D convolutional autoencoders (standard, denoising, and variational) on single orders and multi-order stacks
- Track evolution of latent representations over time
- Correlate latent dimensions with activity indicators (e.g., Hα, RV, BIS, FWHM) 
- Analyze periodicity in latent space using Lomb-Scargle periodograms

---

## Directory Structure

```plaintext
aumicAE/
├── notebooks/
│   ├── check_lines/
│   ├── gif_frames/
│   ├── outputs/
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
│   │   └── preprocess.py
│   ├── models/
│   │   └── autoencoder.py
│   ├── train.py
│   └── utils/
│       └── visualization.py
├── configs/
│   └── base.yaml
├── data/
│   ├── AUMic/
│   │   ├── CARNIR/
│   │   ├── CARVIS/
│   ├── carvis_visA/
│   │   ├── vis_a_files_all.txt
│   │   └── vis_a_files.txt
│   ├── masks/
│   │   └── CARM_VIS_M15.mas
│   └── meta/
│       └── J20451-313.ccfpar.dat
├── outputs/
│   ├── latent/
│   ├── models/
│   ├── plots/
│   ├── logs/
│   └── preprocessed/
├── requirements.txt
├── .python-version
├── poetry.lock
├── pyproject.toml
└── README.md
```

> **Note**: Large `.fits` files and intermediate data products are excluded via `.gitignore`.
> If needed, contact me at [me1020@gmx.de](mailto:me1020@gmx.de) for access.

---

## Environment Setup

* Python 3.11
* Poetry-managed environment (see `pyproject.toml` or `requirements.txt`)
* Key packages: `tensorflow`, `astropy`, `scikit-learn`, `umap-learn`, `matplotlib`, `seaborn`

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

* **Multi-order Modeling**: Extend models to include stacked spectra, i.e, I will start with the entire spectra and then cut it down somewhat.
* **Architecture Improvements**: Add skip connections; explore deeper/lighter encoder-decoder variants
* **Variational & Disentangled Models**: Explore β-VAE, total correlation penalties
* **Activity Indicator Prediction**: Train regressors on latent vectors to predict RV, FWHM, BIS, contrast, and Hα
* **Refined Periodogram Study**: Focus on FAP < 0.01 signals and architecture-agnostic features
* **Final Report & Visualization**: Polish plots, prepare publication-quality visuals, consolidate results

---

## Contact

**Marvin Ernst**
Barcelona School of Economics – MSc in Data Science Methodology
Project supervised by **Dr. Manuel Perger**

---
