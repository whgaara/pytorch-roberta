# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import pickle

from pretrain_config import *
from roberta.common.tokenizers import Tokenizer
from checkpoint.pretrain.ner_dict import NerClassDict
from train_module.ner_module.step2_pretrain_ner import extract_output_entities


class NerInference(object):
    def __init__(self):
        self.NerClassDict = NerClassDict
        self.tokenizer = Tokenizer(VocabPath)
        with open(Class2NumFile, 'rb') as f:
            self.class_to_num = pickle.load(f)
        self.num_to_class = {}
        for k, v in self.class_to_num.items():
            self.num_to_class[v] = k
        self.model = torch.load(NerFinetunePath).to(device).eval()
        print('加载模型完成！')

    def parse_inference_text(self, ori_line):
        ori_line = ori_line.strip().replace(' ', '')
        if len(list(ori_line)) > MedicineLength - 2:
            print('文本过长！')
            return None, None

        input_tokens_id = [101]
        segment_ids = []
        for token in list(ori_line):
            id = self.tokenizer.token_to_id(token)
            input_tokens_id.append(id)
        input_tokens_id.append(102)

        for i in range(MedicineLength - len(input_tokens_id)):
            input_tokens_id.append(0)

        for x in input_tokens_id:
            if x:
                segment_ids.append(1)
            else:
                segment_ids.append(0)

        return input_tokens_id, segment_ids

    def inference_single(self, text):
        input_tokens_id, segment_ids = self.parse_inference_text(text)
        input_tokens_id = torch.tensor(input_tokens_id)
        segment_ids = torch.tensor(segment_ids)

        input_token = input_tokens_id.unsqueeze(0).to(device)
        segment_ids = torch.tensor(segment_ids).unsqueeze(0).to(device)
        input_token_list = input_token.tolist()
        input_len = len([x for x in input_token_list[0] if x]) - 2
        mlm_output = self.model(input_token, segment_ids)[:, 1:input_len + 1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
        output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

        output2class = []
        result = []
        for i, output in enumerate(output_topk):
            output = output[0]
            output2class.append(self.num_to_class[output])
        entities = extract_output_entities(output2class)
        for key, val in entities.items():
            entity_len = len(val)
            current_text = ''
            current_entity = self.NerClassDict[val[0][1:]]
            for i in range(entity_len):
                current_text += text[key + i]
            result.append((current_text, current_entity))
        print('输入数据为：', text)
        print('实体识别结果为:', result)
        return result


if __name__ == '__main__':
    ner_infer = NerInference()
    ner_infer.inference_single('复方氨酚烷氨胶囊')
    ner_infer.inference_single('六味地黄丸')
    ner_infer.inference_single('六味地黄胶囊')
