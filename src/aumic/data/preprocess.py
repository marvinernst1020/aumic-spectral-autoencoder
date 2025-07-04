import numpy as np
import pickle
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.constants import c

c_kms = c.to("km/s").value

def doppler_shift(wave, rv_kms):
    beta = rv_kms / c_kms
    gamma = np.sqrt((1 + beta) / (1 - beta))
    return wave * gamma

def normalize(X, unit_variance=True, subtract_mean=True):
    if subtract_mean:
        mean = np.mean(X, axis=1, keepdims=True)
        X = X - mean
    if unit_variance:
        std = np.std(X, axis=1, keepdims=True)
        X = X / np.where(std == 0, 1, std)
    return X

def preprocess_orders(vis_a_files, meta, valid_orders, use_rv=True,
                      unit_variance=True, subtract_mean=True, order_ranges=None,
                      upsample_factor=1):
    processed = {}
    common_grids = {}

    for order in valid_orders:
        shifted_grids = []
        for idx, f in enumerate(vis_a_files):
            with fits.open(f) as hdul:
                wave = hdul["WAVE"].data[order]
                rv = meta.loc[idx, "RV"]
                shifted_wave = doppler_shift(wave, -rv) if use_rv else wave
                shifted_grids.append(shifted_wave)

        full_min = max(w[0] for w in shifted_grids)
        full_max = min(w[-1] for w in shifted_grids)
        full_range = full_max - full_min
        orig_len = len(shifted_grids[0])

        if order_ranges and order in order_ranges:
            crop_min = max(full_min, order_ranges[order][0])
            crop_max = min(full_max, order_ranges[order][1])
        else:
            crop_min, crop_max = full_min, full_max

        crop_range = crop_max - crop_min
        base_pixels = int(orig_len * (crop_range / full_range))
        n_pixels = max(int(base_pixels * upsample_factor), 32)

        n_pixels = max(int(base_pixels * upsample_factor), 32)
        if n_pixels % 2 != 0:
            n_pixels += 1
        adjusted_crop_max = crop_min + (crop_range * (n_pixels / (n_pixels - 1)))
        print(f"Adjusted pixel count to even: {n_pixels}, New crop range: ({crop_min:.2f}, {adjusted_crop_max:.2f})")
        ref_grid = np.linspace(crop_min, adjusted_crop_max, n_pixels)
        common_grids[order] = ref_grid

        spectra = []
        for idx, f in enumerate(vis_a_files):
            with fits.open(f) as hdul:
                wave = hdul["WAVE"].data[order]
                flux = hdul["SPEC"].data[order]
                cont = hdul["CONT"].data[order]
                norm_flux = flux / np.clip(cont, 1e-3, np.inf)
                norm_flux = np.nan_to_num(norm_flux)

            rv = meta.loc[idx, "RV"]
            wave_shifted = doppler_shift(wave, -rv) if use_rv else wave
            interp = interp1d(wave_shifted, norm_flux, bounds_error=False, fill_value=0.0)
            spectra.append(interp(ref_grid))

        processed[order] = normalize(np.array(spectra), unit_variance=unit_variance, subtract_mean=subtract_mean)

    return processed, common_grids

def save_processed(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def preprocess_rv_shifted(vis_a_files, meta, valid_orders, unit_variance=True, subtract_mean=True):
    return preprocess_orders(vis_a_files, meta, valid_orders, use_rv=True,
                              unit_variance=unit_variance, subtract_mean=subtract_mean)

def preprocess_non_rv_shifted(vis_a_files, meta, valid_orders, unit_variance=True, subtract_mean=True):
    return preprocess_orders(vis_a_files, meta, valid_orders, use_rv=False,
                              unit_variance=unit_variance, subtract_mean=subtract_mean)