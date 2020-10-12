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
        self.src_lines = []
        self.tar_lines = []
        self.class_to_num = {}
        # 收集数据
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    self.src_lines.append(line)
        for line in self.src_lines:
            input_token_ids, input_token_classes = self.parse_ori_line(line)
            tmp = {
                'sentence': input_token_ids,
                'classes': input_token_classes
            }
            self.tar_lines.append(tmp)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]

    def parse_ori_line(self, ori_line):
        """
        :param ori_line: 六味地黄{ypcf,3}丸{yplb,1}
        :return:
        [123, 233, 334, 221, 299, ...]
        [b-ypcf, i-ypcf, i-ypcf, e-ypcf, e-yplb]
        """
        pass
