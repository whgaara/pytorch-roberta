import math
import pkuseg
import numpy as np

from tqdm import tqdm
from roberta.data.mlm_dataset import DataFactory
from roberta.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class NerDataSet(Dataset):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.data_factory = DataFactory()
        self.tokenizer = Tokenizer(VocabPath)
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
            if self.__verify_line(line):
                input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id = self.__parse_ori_line(line)
                tmp = {
                    'input_tokens': input_tokens,
                    'input_tokens_id': input_tokens_id,
                    'input_tokens_class': input_tokens_class,
                    'input_tokens_class_id': input_tokens_class_id
                }
                self.tar_lines.append(tmp)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]

    @property
    def get_classes_to_num(self):
        return self.class_to_num

    def __verify_line(self, line):
        """
        校验是否有成对得{}出现
        """
        if not line:
            return False
        else:
            total = 0
            for char in line:
                if char == '{':
                    total += 1
                if char == '}':
                    total -= 1
            if total == 0:
                return True
            else:
                return False

    def __parse_ori_line(self, ori_line):
        """
        :param ori_line: 六味地黄{3,ypcf}丸{1,yplb}
        :return:
        [123, 233, 334, 221, 299, ...]
        [b-ypcf, i-ypcf, i-ypcf, e-ypcf, e-yplb]
        """
        ori_line = ori_line.strip().replace(' ', '')
        input_tokens = ''
        input_tokens_id = []
        input_tokens_class = []
        input_tokens_class_id = []
        i = 0
        l = 0
        ori_line_list = list(ori_line)
        while i < len(ori_line_list):
            if ori_line_list[i] != '{' and ori_line_list[i] != '}':
                input_tokens += ori_line_list[i]
                input_tokens_class.append(0)
                i += 1
                l += 1
            if ori_line_list[i] == '{':
                current_type = ''
                current_len = ''
                j = i
                while True:
                    j += 1
                    if ori_line_list[j].isdigit():
                        current_len += ori_line_list[j]
                    if ori_line_list[j] == ',':
                        break
                while True:
                    j += 1
                    if ori_line_list[j] == '}':
                        break
                    current_type += ori_line_list[j]

                current_len = int(current_len)
                if current_len == 1:
                    input_tokens_class[l - 1] = 'e' + current_type
                elif current_len == 2:
                    input_tokens_class[l - 2] = 'b' + current_type
                    input_tokens_class[l - 1] = 'e' + current_type
                else:
                    input_tokens_class[l - current_len] = 'b' + current_type
                    input_tokens_class[l - 1] = 'e' + current_type
                    for k in range(current_len - 2):
                        input_tokens_class[l - 2 - k] = 'i' + current_type
                i = j
                i += 1

        for token in input_tokens:
            id = self.tokenizer.token_to_id(token)
            input_tokens_id.append(id)

        # 数值化文字分类
        for token_class in input_tokens_class:
            if token_class in self.class_to_num:
                input_tokens_class_id.append(self.class_to_num[token_class])
            else:
                self.class_to_num[token_class] = len(self.class_to_num) + 1
                input_tokens_class_id.append(self.class_to_num[token_class])

        return input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id


if __name__ == '__main__':
    dataset = NerDataSet(NerSourcePath)
    for i, data in enumerate(dataset):
        print(data)
