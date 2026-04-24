import numpy as np
import os


def save_solution(filepath, x, t, c, params):
    if params is None:
        params = {}
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    np.savez(filepath, x=x, t=t, c=c, **params)
    print(f"Solusi disimpan ke {filepath}")

def load_solution(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} tidak ditemukan.")
    data = np.load(filepath, allow_pickle=True)

    #konversi npz file ke dict biasa
    result_dict = {key: data[key] for key in data.files}
    return result_dict


