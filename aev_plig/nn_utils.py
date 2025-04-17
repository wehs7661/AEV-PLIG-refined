import torch    

def get_device():
    """
    Get the device to be used for training and inference.

    Returns
    -------
    device : torch.device
        The device to be used (CPU or GPU).
    """
    if(torch.cuda.is_available()):
        print("GPU is available")
        device = torch.device("cuda")
    else:
        print("GPU is NOT available")
        device = torch.device("cpu")
    
    return device