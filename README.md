# aumicAE: Spectral Variability Encoding of AU Microscopii

This project applies deep learning techniques to **high-resolution optical spectra** of the active M-dwarf star **AU Microscopii**, using data from the **CARMENES VIS_A channel**. The aim is to learn a compressed latent representation of the star's spectral variability — caused by magnetic activity — using convolutional autoencoders.

The project is inspired by the methodology from:
> Mas-Buitrago et al. (2024), *A&A 687, A205*  
> [https://doi.org/10.1051/0004-6361/202449865](https://doi.org/10.1051/0004-6361/202449865)

However, while that paper estimates stellar parameters across multiple stars, this project focuses on the **spectral time series of a single star**, using the latent space to analyze temporal variability due to surface activity.

---

## Project Goals

- Load and preprocess high-resolution spectra of AU Mic (VIS_A arm)
- Normalize and stack consistent spectral orders
- Train 1D convolutional autoencoders to learn compressed representations
- Track changes in latent space over time
- Relate latent structure to stellar activity and magnetic phenomena

---

## Directory Structure

aumicAE/ ├── data/ │ ├── AUMic/ # Raw CARMENES FITS files │ └── carvis_visA/ # vis_a_files.txt (paths to used files) │ ├── notebooks/ │ ├── 01_explore_carvis_visA.ipynb # Plotting and inspecting spectra │ └── 02_prototype_autoencoder.ipynb # First autoencoder prototype │ ├── src/ │ ├── data/ │ │ └── preprocess.py # Data loading, normalization │ ├── models/ │ │ └── autoencoder.py # Autoencoder architecture │ └── train.py # Training logic │ ├── configs/ │ └── base.yaml # Model/training configuration │ ├── outputs/ │ ├── models/ # Trained models │ ├── logs/ # Training logs │ └── plots/ # Visualizations and reconstructions │ ├── requirements.txt # (Optional) Python dependencies └── README.md # This file

The actual files are saved in aumicAE1 as the size exceeds the data limit of GitHub.
---

## Environment Setup

- Python 3.11
- TensorFlow ≥ 2.13
- astropy, numpy, matplotlib, etc. (see *requirements.txt*)

---

## Status

- ✅ Data loading and normalization (single order)
- ✅ Prototype 1D convolutional autoencoder
- ✅ Plotting and reconstruction evaluation
- 🔜 Latent space visualization and temporal analysis
- 🔜 Extension to stacked multi-order inputs

---

## Contact

Developed by Marvin Ernst  
Barcelona School of Economics – MSc in Data Science Methodology  
Project supervised by Dr. Manuel Perger

---
