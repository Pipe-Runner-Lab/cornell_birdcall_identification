from WaveTransformers.utils import (mix_background_noise,
                                    mix_awg_noise,
                                    time_shift,
                                    speed_tune,
                                    stretch_audio,
                                    pitch_shift,
                                    add_gaussian_noise,
                                    polarity_inversion,
                                    amp_gain
                                    )


class DefaultTransformer:
    def __init__(self):
        pass

    def __call__(self, y, sr):
        y, sr = mix_background_noise(y, sr, coeff=0.6)
        y, sr = mix_awg_noise(y, sr, SNR_db=15)
        y, sr = time_shift(y, sr)
        y, sr = speed_tune(y, sr)
        y, sr = stretch_audio(y, sr)
        y, sr = pitch_shift(y, sr, n_steps=2)
        y, sr = add_gaussian_noise(y, sr)
        y, sr = polarity_inversion(y, sr)
        y, sr = amp_gain(y, sr)
        return y, sr

    def __str__(self):
        string = "DFLT"
        return string
