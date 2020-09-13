import torch

from pretrain_config import FinetunePath, device
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
        if len(inputs) < 511:
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

    if len(inputs) < 511:
        inputs.append(roberta_data.token_sep_id)
        segments.append(1)
        for i in range(512 - len(inputs)):
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


def inference_finetune(text):
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


if __name__ == '__main__':
    # get_pretrain_model_parameters()
    get_finetune_model_parameters()
    # inference_finetune('平头医保科技')
