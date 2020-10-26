import pickle

from tqdm import tqdm
from roberta.data.mlm_dataset import DataFactory
from roberta.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class NerDataSet(Dataset):
    def __init__(self):
        self.data_factory = DataFactory()
        self.tokenizer = Tokenizer(VocabPath)
        self.src_lines = []
        self.tar_lines = []

        # 载入类别和编号的映射表
        with open(Class2NumFile, 'rb') as f:
            self.class_to_num = pickle.load(f)

        # 读取训练数据
        with open(NerCorpusPath, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    self.src_lines.append(line)

        for line in tqdm(self.src_lines):
            items = line.split(',')
            input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id = items
            if not input_tokens:
                continue
            input_tokens_id = [int(x) for x in input_tokens_id.split(' ')]
            input_tokens_class = input_tokens_class.split(' ')
            input_tokens_class_id = [int(x) for x in input_tokens_class_id.split(' ')]

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


if __name__ == '__main__':
    dataset = NerDataSet()
    for i, data in enumerate(dataset):
        print(data)
