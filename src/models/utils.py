import os
import json

def save_all_models(vae, encoder, decoder, path_prefix, settings_dict=None):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    vae.save(f"{path_prefix}fullmodel.keras")
    encoder.save(f"{path_prefix}encoder.keras")
    decoder.save(f"{path_prefix}decoder.keras")

    if settings_dict is not None:
        with open(f"{path_prefix}config.json", "w") as f:
            json.dump(settings_dict, f, indent=2)

    print(f"Models and config saved to: {path_prefix}*.keras")