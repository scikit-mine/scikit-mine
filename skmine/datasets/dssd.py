from scipy.io import arff
import pandas as pd
import numpy as np
from scipy import signal

def load_emotions(arff_file_path: str, sample: int = 0, return_D_y: bool = True):
    target_columns  = ["amazed-suprised","happy-pleased", "relaxing-calm", "quiet-still", "sad-lonely","angry-aggresive"]
    arf = arff.loadarff(arff_file_path)
    df = pd.DataFrame(arf[0])
    df[target_columns] = df[target_columns].astype("byte").astype("bool")
    if sample > 0:
        df = df.sample(sample)
    if return_D_y:
        return df[list(col for col in df.columns if col not in target_columns)], df[target_columns]
    return df


def load_time_series_data1(return_D_y: bool = True):
    t = np.linspace(0, 1, 50, endpoint=False)
    base_square_ts = signal.square(2 * np.pi * 5 * t)
    base_sin_ts = np.sin(2 * np.pi * t)
    base_triangular_ts = signal.sawtooth(2 * np.pi * 5 * t)
    tris = [
        {"bin": False, "num": 1.9,    "cat": "tri", "ts": base_triangular_ts},
        {"bin": True,  "num": .9,    "cat": "tri", "ts": base_triangular_ts},
        {"bin": False, "num": 1.5,  "cat": "tri", "ts": base_triangular_ts},
        {"bin": False, "num": 1.5,  "cat": "tri", "ts": base_triangular_ts},
    ]

    squares = [
        {"bin": True,  "num": 1,    "cat": "squ", "ts": base_square_ts},
        {"bin": False, "num": 6,    "cat": "squ", "ts": base_square_ts},
        {"bin": True,  "num": 8,    "cat": "sq_", "ts": base_square_ts},
        {"bin": False, "num": 5,    "cat": "squ", "ts": base_square_ts},
    ]

    sins = [
        {"bin": True,  "num": 10,    "cat": "sin", "ts": base_sin_ts},
        {"bin": False, "num": 40,    "cat": "sin", "ts": base_sin_ts},
        {"bin": True,  "num": 9,    "cat": "sin", "ts": base_sin_ts},
        {"bin": False, "num": 12,    "cat": "sin", "ts": base_sin_ts},
    ]

    df = pd.DataFrame([*tris, *squares, *sins])
    target_attributes = ["ts"]
    if return_D_y:
        return df[list(col for col in df.columns if col not in target_attributes)], df[target_attributes]
    return df
