from WaveTransformers.tr_dflt import DefaultTransformer

def get(tr_config):
    # ===========================================================================
    #                          Wave Transformer list
    # ===========================================================================
    tr_name = tr_config.wave_transformer


    if tr_name == "DFLT":
        return DefaultTransformer()
    else:
        raise Exception("Transformer not in list!")
