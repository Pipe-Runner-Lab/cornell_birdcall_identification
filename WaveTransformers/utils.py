import soundfile as sf
from os import listdir
import numpy as np
import random
from pathlib import Path

from utils.paths import NOISE_ROOT_DIR

noise_dir = Path(NOISE_ROOT_DIR) / "random"


def get_audio_data(audio_filepath, max_length=5):
    y, sr = sf.read(audio_filepath)
    y, sr = trim_audio_data(y, sr, max_length=5)
    return y, sr


def trim_audio_data(y, sr, max_length=5):
    length = y.shape[0]
    effective_length = sr * max_length

    if length < effective_length:
        new_y = np.zeros(effective_length, dtype=y.dtype)
        start = np.random.randint(effective_length - length)
        new_y[start:start + length] = y
        y = new_y.astype(np.float32)
    elif length > effective_length:
        start = np.random.randint(length - effective_length)
        y = y[start:start + effective_length].astype(np.float32)
    else:
        y = y.astype(np.float32)

    return y, sr


def mix_background_noise(y, sr, coeff):
    """Adds noise to provided audio clip

    Args:
        y: audio file wihtout noise
        coeff: coeff is the proportion of noise to be added
    """
    noise_file_path = random.choice(listdir(noise_dir))
    noise_file_path = noise_dir / noise_file_path

    n_y, n_sr = get_audio_data(noise_file_path)

    noise_energy = np.sqrt(n_y.dot(n_y))
    audio_energy = np.sqrt(y.dot(y))

    y += coeff * n_y * (audio_energy / noise_energy)
    return y, sr


def mix_awg_noise(y, sr, SNR_db=10):
    # SNR in db  SNR = 10*log10 Pdata/PNoise
    # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
    # We can experiment with SNR_db
    audio_watts = y**2
    
    sig_avg_watts = np.mean(audio_watts)
    epsilon = 10**-6
    sig_avg_db = 10 * np.log10(sig_avg_watts+epsilon)
    noise_avg_db = sig_avg_db - SNR_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise_volts = np.random.normal(
        0, np.sqrt(noise_avg_watts), len(audio_watts))
    y_volts = y + noise_volts
    return y_volts, sr
