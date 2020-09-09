import glob
import math
import pkuseg
import numpy as np

from tqdm import tqdm
from roberta.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class RobertaTrainingData(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath, do_lower_case=True)
        self.seg = pkuseg.pkuseg()
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id

    def __token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def texts_to_ids(self, texts):
        texts_ids = []
        for text in texts:
            # 处理每个句子
            # 注意roberta里并不是针对每个字进行mask，而是对字或者词进行mask
            words = self.seg.cut(text)
            for word in words:
                # text_ids首位分别是cls和sep，这里暂时去除
                word_tokes = self.tokenizer.tokenize(text=word)[1:-1]
                words_ids = self.tokenizer.tokens_to_ids(word_tokes)
                texts_ids.append(words_ids)
        return texts_ids

    def ids_to_mask(self, texts_ids):
        """
        这里只对每个字做了mask，其实还可以考虑先对句子进行分词，如果是一个词的，可以对词中所有字同时进行mask
        """
        instances = []
        total_ids = []
        total_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(texts_ids))

        for i, word_id in enumerate(texts_ids):
            # 为每个字生成对应概率
            total_ids.extend(word_id)
            if mask_rates[i] < MaskRate:
                # 因为word_id可能是一个字，也可能是一个词
                for sub_id in word_id:
                    total_masks.append(self.__token_process(sub_id))
            else:
                total_masks.extend([0]*len(word_id))

        # 每个实例的最大长度为512，因此对一个段落进行裁剪
        # 510 = 512 - 2，给cls和sep留的位置
        for i in range(math.ceil(len(total_ids)/510)):
            tmp_ids = [self.token_cls_id]
            tmp_masks = [self.token_pad_id]
            tmp_ids.extend(total_ids[i*510: min((i+1)*510, len(total_ids))])
            tmp_masks.extend(total_masks[i*510: min((i+1)*510, len(total_masks))])
            # 不足512的使用padding补全
            diff = SentenceLength - len(tmp_ids)
            if diff == 1:
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
            else:
                # 添加结束符
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
                # 将剩余部分padding补全
                tmp_ids.extend([self.token_pad_id] * (diff - 1))
                tmp_masks.extend([self.token_pad_id] * (diff - 1))
            instances.append([tmp_ids, tmp_masks])
        return instances


class RobertaDataSet(Dataset):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.roberta_data = RobertaTrainingData()
        self.src_lines = []
        self.tar_lines = []
        for i in range(RepeatNum):
            for texts in tqdm(self.__get_texts()):
                texts_ids = self.roberta_data.texts_to_ids(texts)
                self.src_lines.append(texts_ids)
        for line in self.src_lines:
            instances = self.roberta_data.ids_to_mask(line)
            for instance in instances:
                self.tar_lines.append(instance)

    def __get_texts(self):
        filenames = glob.glob('%s/*.txt' % self.corpus_path)
        np.random.shuffle(filenames)
        count, texts = 0, []
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    texts.append(line)
                    count += 1
                    # 10个句子组成一个段落
                    if count == 10:
                        yield texts
                        count, texts = 0, []
        if texts:
            yield texts

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        output = {}
        instance = self.tar_lines[item]
        token_ids = instance[0]
        mask_ids = instance[1]
        is_masked = [1 if x else 0 for x in mask_ids]
        input_token_ids = self.__gen_input_token(token_ids, mask_ids)
        segment_ids = [1 if x else 0 for x in token_ids]
        onehot_labels = self.__id_to_onehot(token_ids)

        output['input_token_ids'] = input_token_ids
        output['token_ids_labels'] = token_ids
        output['onehot_labels'] = onehot_labels
        output['is_masked'] = is_masked
        output['segment_ids'] = segment_ids

        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_input_token(self, token_ids, mask_ids):
        assert len(token_ids) == len(mask_ids)
        input_token_ids = []
        for token, mask in zip(token_ids, mask_ids):
            if mask == 0:
                input_token_ids.append(token)
            else:
                input_token_ids.append(mask)
        return input_token_ids

    def __id_to_onehot(self, ids):
        onehot = []
        for id in ids:
            tmp = [0.0 for i in range(VocabSize)]
            tmp[id] = 1.0
            onehot.append(tmp)
        return onehot
