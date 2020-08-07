import soundfile as sf
from pathlib import Path
import numpy as np
import pandas as pd
import random
from Datasets.utils import scale_minmax
from utils.paths import DATA_ROOT_DIR

noise_dir=DATA_ROOT_DIR / "noise" / "10_sec_clip"


def add_noise(audio,coeff):
	## input -> audio file wihtout noise , coeff is the proportion of noise to be added
	## output ->audio file with noise
    noise_df=pd.read_csv(Path(noise_dir / "df.csv"))
    samp=np.random.randint(len(noise_df))
    noise_file_path=noise_dir / noise_df.iloc[samp].file_name

    n_y, n_sr = sf.read(noise_file_path)
    length = n_y.shape[0]
    effective_length = sr * 5
    
    if length < effective_length:
        new_y = np.zeros(effective_length, dtype=n_y.dtype)
        start = np.random.randint(effective_length - length)
        new_y[start:start + length] = n_y
        n_y = new_y.astype(np.float32)
    elif length > effective_length:
        start = np.random.randint(length - effective_length)
        n_y = n_y[start:start + effective_length].astype(np.float32)
    else:
        n_y = n_y.astype(np.float32)

    noise_energy = np.sqrt(n_y.dot(n_y))
    audio_energy = np.sqrt(audio.dot(audio))

    audio += coeff * n_y *( audio_energy / noise_energy)
    return audio,n_sr
