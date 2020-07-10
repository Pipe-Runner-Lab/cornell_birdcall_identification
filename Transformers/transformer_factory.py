# from transformer.utils import (
#     DefaultTransformer, ImageTransformer, ImageTTATransformer, PolicyTransformer)

from Transformers.tr_dflt import DefaultTransformer
from Transformers.tr_img1 import IMG1
from Transformers.tr_img2 import IMG2


def _validate_config(tr_config):
    if not isinstance(tr_config.transformer, str):
        raise Exception("transformer name invalid!")

    dims = [int(dim) for dim in tr_config.resize.split("x")]

    if len(dims) < 2:
        raise Exception("resize dims missing!")
    if not isinstance(dims[0], int) or not isinstance(dims[0], int):
        raise Exception("resize dims invalid!")

    return dims[0], dims[1]


def get(tr_config=None):
    height, width = _validate_config(tr_config)

    # ===========================================================================
    #                            Transformer list
    # ===========================================================================

    tr_name = tr_config.transformer

    if tr_name == "DFLT":
        return DefaultTransformer(height, width)
    elif tr_name == "IMG1":
        return IMG1(height, width)
    elif tr_name == "IMG2":
        return IMG2(height, width)
    else:
        raise Exception("Transformer not in list!")
