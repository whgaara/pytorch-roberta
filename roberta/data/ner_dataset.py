import pickle

from roberta.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class NerDataSet(Dataset):
    def __init__(self):
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

        for line in self.src_lines:
            items = line.split(',')
            input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id = items
            if not input_tokens:
                continue
            input_tokens_id = [int(x) for x in input_tokens_id.split(' ')]
            input_tokens_class_id = [int(x) for x in input_tokens_class_id.split(' ')]
            segment_ids = []
            for x in input_tokens_class_id:
                if x:
                    segment_ids.append(1)
                else:
                    segment_ids.append(0)
            tmp = {
                'input_tokens_id': input_tokens_id,
                'input_tokens_class_id': input_tokens_class_id,
                'segment_ids': segment_ids
            }
            tmp = {k: torch.tensor(v, dtype=torch.long) for k, v in tmp.items()}
            self.tar_lines.append(tmp)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]


class NerTestSet(Dataset):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.src_lines = []
        self.tar_lines = []

        # 载入类别和编号的映射表
        with open(Class2NumFile, 'rb') as f:
            self.class_to_num = pickle.load(f)

        # 读取训练数据
        with open(NerTestPath, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip()
                    self.src_lines.append(line)

        for line in self.src_lines:
            items = line.split(',')
            input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id = items
            if not input_tokens:
                continue
            input_tokens_id = [int(x) for x in input_tokens_id.split(' ')]
            input_tokens_class_id = [int(x) for x in input_tokens_class_id.split(' ')]
            segment_ids = []
            for x in input_tokens_class_id:
                if x:
                    segment_ids.append(1)
                else:
                    segment_ids.append(0)
            tmp = {
                'input_tokens_id': input_tokens_id,
                'input_tokens_class_id': input_tokens_class_id,
                'segment_ids': segment_ids
            }
            tmp = {k: torch.tensor(v, dtype=torch.long) for k, v in tmp.items()}
            self.tar_lines.append(tmp)

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        return self.tar_lines[item]


if __name__ == '__main__':
    dataset = NerDataSet()
    for i, data in enumerate(dataset):
        print(data)
