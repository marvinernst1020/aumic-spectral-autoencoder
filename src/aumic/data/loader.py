import pickle
import numpy as np

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_processed_orders(path="../outputs/preprocessed/with_rv.pkl"):
    return load_pickle(path)

def load_metadata(path="../outputs/exact_metadata/metadata.csv"):
    import pandas as pd
    return pd.read_csv(path).sort_values("BJD").reset_index(drop=True)

def load_file_list(path="../../aumicAE/data/carvis_visA/vis_a_files.txt"):
    with open(path, "r") as f:
        return ["../../aumicAE/" + line.strip().lstrip("../") for line in f if line.strip()]
    
def select_orders(orders_dict, order_ids):
    return np.hstack([orders_dict[o] for o in order_ids])

