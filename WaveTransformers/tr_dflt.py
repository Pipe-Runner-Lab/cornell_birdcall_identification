from WaveTransformers.utils import mix_background_noise, mix_awg_noise


class DefaultTransformer:
    def __init__(self):
        pass

    def __call__(self, y, sr):
        # y, sr = mix_background_noise(y, sr, 0.5)
        y, sr = mix_awg_noise(y, sr, SNR_db=10)
        return y, sr

    def __str__(self):
        string = "DFLT"
        return string
