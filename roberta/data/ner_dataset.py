import math
import pkuseg
import numpy as np

from tqdm import tqdm
from roberta.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class NerDataSet(Dataset):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
