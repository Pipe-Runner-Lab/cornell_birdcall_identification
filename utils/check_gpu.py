import torch


def get_training_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("[ GPU: {} (1/{}) ]".format(
            torch.cuda.get_device_name(0),
            torch.cuda.device_count())
        )
    else:
        device = torch.device("cpu")
        print("[ Running on the CPU ]")

    return device
