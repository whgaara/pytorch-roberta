import glob
import numpy as np
import random

from pretrain_config import *
from roberta.common.tokenizers import Tokenizer


def random_wrong(text):
    tokenizer = Tokenizer(VocabPath, do_lower_case=True)
    length = len(text)
    position = random.randint(0, length-1)
    number = random.randint(672, 7992)
    text = list(text)
    text[position] = tokenizer.id_to_token(number)
    text = ''.join(text)
    return text


def gen_train_test():
    f_train = open('data/train_data/train.txt', 'w', encoding='utf-8')
    f_test = open('data/test_data/test.txt', 'w', encoding='utf-8')

    filenames = glob.glob('%s/*.txt' % SourcePath)
    np.random.shuffle(filenames)
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                rad = random.randint(0, 10)
                if rad < 1:
                    f_test.write(random_wrong(line) + '\n')
                else:
                    f_train.write(line + '\n')

    f_train.close()
    f_test.close()


if __name__ == '__main__':
    gen_train_test()
