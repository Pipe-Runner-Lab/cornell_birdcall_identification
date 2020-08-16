from WaveTransformers.tr_dflt import DefaultTransformer


def get(tr_config="DFLT"):
    # ===========================================================================
    #                          Wave Transformer list
    # ===========================================================================

    tr_name = tr_config.transformer

    if tr_name == "DFLT":
        return DefaultTransformer()
    else:
        raise Exception("Transformer not in list!")
