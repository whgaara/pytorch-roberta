import pickle
import random

from tqdm import tqdm
from pretrain_config import *
from roberta.common.tokenizers import Tokenizer


def verify_line(line):
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


def parse_ori_line(ori_line, class_to_num):
    """
    :param ori_line: 六味地黄{3,ypcf}丸{1,yplb}
    :return:
    [101, 123, 233, 334, 221, 299, ..., 102, ...]
    [ptzf, b-ypcf, i-ypcf, i-ypcf, e-ypcf, e-yplb, ..., pytzf, ...]
    """
    ori_line = ori_line.strip().replace(' ', '')
    input_tokens = ''
    input_tokens_id = []
    input_tokens_class = []
    input_tokens_class_id = []
    tokenizer = Tokenizer(VocabPath)
    i = 0
    l = 0
    ori_line_list = list(ori_line)
    while i < len(ori_line_list):
        if ori_line_list[i] != '{' and ori_line_list[i] != '}':
            input_tokens += ori_line_list[i]
            input_tokens_class.append(NormalChar)
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
        id = tokenizer.token_to_id(token)
        if not id:
            print('警告！本地vocab缺少以下字符：%s！' % token)
            continue
        input_tokens_id.append(id)

    # 补全类别
    if len(input_tokens_id) > MedicineLength - 2:
        return None, None, None, None
    else:
        input_tokens_id.append(102)
        input_tokens_class.append(NormalChar)
        for i in range(MedicineLength - len(input_tokens_id) - 1):
            input_tokens_id.append(0)
            input_tokens_class.append('pad')

    # 数值化文字分类
    input_tokens_id = [101] + input_tokens_id
    input_tokens_class = [NormalChar] + input_tokens_class
    for token_class in input_tokens_class:
        if token_class in class_to_num:
            input_tokens_class_id.append(class_to_num[token_class])
        else:
            class_to_num[token_class] = len(class_to_num)
            input_tokens_class_id.append(class_to_num[token_class])

    return input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id, class_to_num


def gen_train_test():
    class_to_num = {
        'pad': 0,
        NormalChar: 1
    }
    f_train = open(NerCorpusPath, 'w', encoding='utf-8')
    f_test = open(NerTestPath, 'w', encoding='utf-8')
    with open(NerSourcePath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if verify_line(line):
                line = line.strip()
                input_tokens, input_tokens_id, input_tokens_class, input_tokens_class_id, class_to_num \
                    = parse_ori_line(line, class_to_num)
                rad = random.randint(0, 10)
                if rad < 1:
                    f_test.write(input_tokens + ',' + ' '.join([str(x) for x in input_tokens_id]) + ',' +
                                 ' '.join(input_tokens_class) + ',' + ' '.join([str(x) for x in input_tokens_class_id]) + '\n')
                else:
                    f_train.write(input_tokens + ',' + ' '.join([str(x) for x in input_tokens_id]) + ',' +
                                  ' '.join(input_tokens_class) + ',' + ' '.join([str(x) for x in input_tokens_class_id]) + '\n')
    f_train.close()
    f_test.close()

    # 将类型及编号进行存储
    with open(Class2NumFile, 'wb') as f:
        pickle.dump(class_to_num, f)


if __name__ == '__main__':
    print(len(open(VocabPath, 'r', encoding='utf-8').readlines()))
    gen_train_test()
