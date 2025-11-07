import os, random, argparse, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader
from torchvision import transforms
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from models.csrnet import CSRNet

def set_seed(s = 42):
    random.seed(s)
    