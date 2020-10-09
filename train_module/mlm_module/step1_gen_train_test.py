import glob
import numpy as np
import pkuseg
import random

from pretrain_config import *
from roberta.common.tokenizers import Tokenizer


def check_srcdata_and_vocab():
    segment = pkuseg.pkuseg()
    f1 = open('../../data/src_data/src_data.txt', 'r', encoding='utf-8')
    f2 = open(VocabPath, 'r', encoding='utf-8')
    local_tokens = []
    vocabs = []
    missing = []
    if ModelClass == 'RobertaMlm':
        for l in f1:
            if l:
                l = l.strip()
                l_seg = segment.cut(l)
                for x in l_seg:
                    local_tokens.append(x)
    else:
        for l in f1:
            if l:
                l = l.strip()
                for x in l:
                    local_tokens.append(x)
    local_tokens = list(set(local_tokens))

    for l in f2:
        if l:
            l = l.strip()
            vocabs.append(l)
    for x in local_tokens:
        if x not in vocabs:
            missing.append(x)
    if missing:
        print('警告！本地vocab缺少以下字符：')
        for x in missing:
            print(x)


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
    f_train = open('../../data/train_data/train.txt', 'w', encoding='utf-8')
    f_test = open('../../data/test_data/test.txt', 'w', encoding='utf-8')

    # filenames = glob.glob('%s/*.txt' % SourcePath)
    # np.random.shuffle(filenames)
    # for filename in filenames:
    filename = '../../data/src_data/src_data.txt'
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            rad = random.randint(0, 10)
            if rad < 5:
                f_test.write(line + '-***-' + random_wrong(line) + '\n')
                f_train.write(line + '\n')
            else:
                f_train.write(line + '\n')

    f_train.close()
    f_test.close()


if __name__ == '__main__':
    print(len(open(VocabPath, 'r', encoding='utf-8').readlines()))
    check_srcdata_and_vocab()
    gen_train_test()