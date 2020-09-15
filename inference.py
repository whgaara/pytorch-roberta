import torch

from char_sim import CharFuncs
from pretrain_config import FinetunePath, device, PronunciationPath, SentenceLength
from roberta.data.dataset import RobertaTrainingData


def get_id_from_text(text):
    assert isinstance(text, str)
    inputs = []
    segments = []
    text = [text]
    roberta_data = RobertaTrainingData()
    ids = roberta_data.texts_to_ids(text)
    inputs.append(roberta_data.token_cls_id)
    segments.append(1)
    for id in ids:
        if len(inputs) < SentenceLength - 1:
            if isinstance(id, list):
                for x in id:
                    inputs.append(x)
                    segments.append(1)
            else:
                inputs.append(id)
                segments.append(1)
        else:
            inputs.append(roberta_data.token_sep_id)
            segments.append(1)
            break

    if len(inputs) != len(segments):
        print('len error!')
        return None

    if len(inputs) < SentenceLength - 1:
        inputs.append(roberta_data.token_sep_id)
        segments.append(1)
        for i in range(SentenceLength - len(inputs)):
            inputs.append(roberta_data.token_pad_id)
            segments.append(roberta_data.token_pad_id)

    inputs = torch.tensor(inputs).unsqueeze(0).to(device)
    segments = torch.tensor(segments).unsqueeze(0).to(device)
    return inputs, segments, roberta_data


def get_finetune_model_parameters():
    model = torch.load('checkpoint/finetune/roberta_trained.model')
    layers = model.state_dict().keys()
    for layer in layers:
        print(layer)
    return model.state_dict()


def get_pretrain_model_parameters():
    model = torch.load('checkpoint/pretrain/pytorch_model.bin')
    layers = dict(model).keys()
    for layer in layers:
        print(layer)
    return dict(model)


def inference_test(text):
    input_len = len(text)
    text2id, segments, roberta_data = get_id_from_text(text)
    print('文字转换成功！')
    model = torch.load(FinetunePath).to(device)
    model.eval()
    print('加载模型成功！')
    with torch.no_grad():
        output_tensor = model(text2id, segments)[:, 1:input_len+1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
        output = torch.argmax(output_tensor, dim=-1).squeeze(0).tolist()
        for index, num in enumerate(output):
            word = roberta_data.tokenizer.id_to_token(num)
            print(word, num, output_tensor[0][index][num].item())


def get_topk(text):
    input_len = len(text)
    text2id, segments, roberta_data = get_id_from_text(text)
    model = torch.load(FinetunePath).to(device)
    model.eval()
    with torch.no_grad():
        result = []
        output_tensor = model(text2id, segments)[:, 1:input_len + 1, :]
        output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
        output_topk_prob = torch.topk(output_tensor, 5).values.squeeze(0).tolist()
        output_topk_indice = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
        for i, words in enumerate(output_topk_indice):
            tmp = []
            for j, candidate in enumerate(words):
                word = roberta_data.tokenizer.id_to_token(candidate)
                tmp.append(word)
            result.append(tmp)
    return result, output_topk_prob


def curve(confidence, similarity):
    flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
    flag2 = 0.1 * confidence + similarity - 0.6 > 0
    if flag1 or flag2:
        return True
    return False


def inference(text):
    char_func = CharFuncs(PronunciationPath)
    candidates, probs = get_topk(text)
    text_list = list(text)
    correct_sentence = []
    result = {
        '原句': text,
        '纠正': '',
        '纠正数据': [
        ]
    }

    for i, ori in enumerate(text_list):
        correct = {}
        correct['原字'] = ori
        candidate = candidates[i]
        confidence = probs[i]
        print('原字：%s；候选字：%s' % (ori, candidate))
        if ori in candidate:
            correct_sentence.append(ori)
            continue
        else:
            max_can = ''
            max_sim = 0
            max_conf = 0
            for j, can in enumerate(candidate):
                similarity = char_func.similarity(ori, can)
                if similarity > max_sim:
                    max_can = can
                    max_sim = similarity
                    max_conf = confidence[i]
            # if max_sim > 0.5:
            if curve(max_conf, max_sim):
                correct['新字'] = max_can
                correct['相似度'] = max_sim
                result['纠正数据'].append(correct)
                correct_sentence.append(max_can)
            else:
                correct_sentence.append(ori)
    result['纠正'] = ''.join(correct_sentence)
    return result


if __name__ == '__main__':
    # get_pretrain_model_parameters()
    # get_finetune_model_parameters()
    # inference_test('平安医保科技')
    result = inference('糖化講境惹尘烟')
    print(result)
