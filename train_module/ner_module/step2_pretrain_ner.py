import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from roberta.data.ner_dataset import *
from roberta.layers.Roberta_ner import RobertaNer


def extract_entities(class_list):
    pass


if __name__ == '__main__':
    if Debug:
        print('开始训练 %s' % get_time())
    dataset = NerDataSet()
    dataloader = DataLoader(dataset=dataset, batch_size=NerBatchSize, shuffle=True, drop_last=True)
    testset = NerTestSet()

    # 加载类别映射表
    with open(Class2NumFile, 'rb') as f:
        class_to_num = pickle.load(f)
    num_to_class = {}
    for k, v in class_to_num.items():
        num_to_class[v] = k

    number_of_categories = len(dataset.class_to_num)
    roberta_ner = RobertaNer(number_of_categories).to(device)
    if Debug:
        print('Total Parameters:', sum([p.nelement() for p in roberta_ner.parameters()]))

    if UsePretrain and os.path.exists(NerFinetunePath):
        print('开始加载本地模型！')
        roberta_ner.load_pretrain(MedicineLength)
        print('完成加载本地模型！')

    optim = Adam(roberta_ner.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        if Debug:
            print('第%s个Epoch %s' % (epoch, get_time()))
        roberta_ner.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            if Debug:
                print('生成数据 %s' % get_time())
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_tokens_id']
            segment_ids = data['segment_ids']
            label = data['input_tokens_class_id']
            if Debug:
                print('获取数据 %s' % get_time())
            mlm_output = roberta_ner(input_token, segment_ids).permute(0, 2, 1)
            if Debug:
                print('完成前向 %s' % get_time())
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()
            if Debug:
                print('完成反向 %s\n' % get_time())

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = NerFinetunePath + '.ep%d' % epoch
        torch.save(roberta_ner.cpu(), output_path)
        roberta_ner.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            roberta_ner.eval()
            test_count = 0
            top1_count = 0
            top5_count = 0
            for test_data in testset:
                label2class = []
                output2class = []

                input_token = test_data['input_tokens_id'].unsqueeze(0).to(device)
                segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                input_token_list = input_token.tolist()
                label_list = test_data['input_tokens_class_id'].tolist()
                input_len = len([x for x in input_token_list[0] if x]) - 2
                mlm_output = roberta_ner(input_token, segment_ids)[:, 1:input_len + 1, :]
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

                # 累计数值
                for i, output in enumerate(output_topk):
                    output = output[0]
                    output2class.append(num_to_class[output])
                    label2class.append(num_to_class[label_list[i]])
                output_entities = extract_entities(output2class)
                label_entities = extract_entities(label2class)



            # if test_count:
            #     top1_acc = float(top1_count) / float(test_count)
            #     acc = round(top1_acc, 2)
            #     print('top1纠正正确率：%s' % acc)
            #     top5_acc = float(top5_count) / float(test_count)
            #     acc = round(top5_acc, 2)
            #     print('top5纠正正确率：%s\n' % acc)
