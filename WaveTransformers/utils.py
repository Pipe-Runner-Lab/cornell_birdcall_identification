import soundfile as sf
from os import listdir
import numpy as np
import random
import cv2
import librosa
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
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - SNR_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise_volts = np.random.normal(
        0, np.sqrt(noise_avg_watts), len(audio_watts))
    y_volts = y + noise_volts
    return y_volts, sr


def time_shift(y, sr):
    start_ = int(np.random.uniform(-80000, 80000))
    if start_ >= 0:
        y_new = np.r_[y[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        y_new = np.r_[np.random.uniform(-0.001, 0.001, -start_), y[:start_]]

    return y_new, sr


def speed_tune(y, sr, speed_rate=None):
    if not speed_rate:
        speed_rate = np.random.uniform(0.6, 1.3)

    y_new = cv2.resize(y, (1, int(len(y) * speed_rate))).squeeze()
    if len(y_new) < len(y):
        pad_len = len(y) - len(y_new)
        y_new = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len/2)),
                      y_new,
                      np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len/2)))]
    else:
        cut_len = len(y_new) - len(y)
        y_new = y_new[int(cut_len/2):int(cut_len/2)+len(y)]

    return y_new, sr


def stretch_audio(y, sr, rate=None):
    if not rate:
        rate = np.random.uniform(0.5, 1.5)

    input_length = len(y)

    y = librosa.effects.time_stretch(y, rate)

    if len(y) > input_length:
        y = y[:input_length]
    else:
        y = np.pad(y, (0, max(0, input_length - len(y))), "constant")

    return y, sr


def pitch_shift(y, sr, n_steps=None):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps), sr


def add_gaussian_noise(y, sr):
    noise = np.random.randn(len(y))
    y_new = y + 0.005*noise
    return y_new, sr

def polarity_inversion(y, sr):
    return -y, sr


def amp_gain(y, sr, min_gain_in_db=-12, max_gain_in_db=12):
    assert min_gain_in_db <= max_gain_in_db
    min_gain_in_db = min_gain_in_db
    max_gain_in_db = max_gain_in_db

    amplitude_ratio = 10**(random.uniform(min_gain_in_db, max_gain_in_db)/20)
    return y * amplitude_ratio, sr