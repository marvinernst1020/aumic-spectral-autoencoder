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
│   └── 05_full_pipeline.ipynb
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
│   └── logs/
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

- Data loading & normalization: Multi-order stacking, standard or denoising input.

- Prototype Autoencoder: Simple Conv1D autoencoder tested on single & multi-order input.

- Latent Dimension Analysis: Explored 2–64D latent spaces; identified ~32D as a sweet spot.

- Denoising Autoencoder: Injects Gaussian noise and reconstructs clean spectra; shows robust latent representations.

- Reconstruction Evaluation: Visual comparisons of reconstructions for Hα (order 25) and Ca II lines (orders 46, 47).

---

## Next Steps

- Temporal Analysis: Examine latent evolution over time; look for periodicities or flare signatures.

- Refine Architecture: Add skip connections, BatchNorm, or deeper layers to improve reconstructions.

- Correlate with Activity Indicators: Compare latent parameters with known lines (Hα, Ca II) or photometric flare logs.

---

## Contact

Developed by Marvin Ernst  
Barcelona School of Economics – MSc in Data Science Methodology  
Project supervised by Dr. Manuel Perger

---
