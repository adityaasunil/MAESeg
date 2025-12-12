import torch 
import os, sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if 'mps' in torch.backends and torch.backends.mps.is_available() else 'cpu')

def denorm(tensor, std, mean):
    """
    Denormalizes a tensor using the provided standard deviation and mean
    
    Args:
        tensor (torch.Tensor): The normalized tensor to be denormalized
        std (float): The standard deviation used for normalization
        mean (float): The mean used for normalization

    Returns:
        np.ndarray: The denormalized tensor as a NumPy array
    """

    tensor = tensor.detach().cpu()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)

    if tensor.ndim == 2:
        return tensor.numpy()
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            return tensor.squeeze(0).numpy()
        else:
            return tensor.permute(1, 2, 0).numpy()
    elif tensor.ndim == 4:
        if tensor.shape[1] == 1:
            return tensor.squeeze(1).numpy()
        else:
            return tensor.permute(0, 2, 3, 1).numpy()
        
def get_transforms(split):
    """
    Returns the appropriate transformations for the given dataset split.
    
    Args:
        split (str): The dataset split, either 'train' or 'val'

    Returns:
        torchvision.transforms.Compose: The composed transformations
    """
    if split == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])