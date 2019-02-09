import torch     #ok
from sklearn.metrics import average_precision_score
import numpy as np
import os
import torch.nn as nn
import torch.utils.data as Data

from modules.GPVModel import Network_3D, DecoderNetwork
from torchvision.models.alexnet import alexnet
from dataprocess import TrainDataset
from tensorboardX import SummaryWriter

CWD_PATH = os.getcwd()
BATCH_SIZE = 20

def evaluation(y_true, y_score):
    Y_true = y_true.detach().numpy()
    Y_score = y_score.detach().numpy()
    return average_precision_score(Y_true, Y_score)

class evalModel():
    def __init__(self, eval_dir):
        self.data_dir = eval_dir

    def