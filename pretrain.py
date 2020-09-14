import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from roberta.data.dataset import *
from roberta.layers.Roberta import Roberta


if __name__ == '__main__':
    roberta = Roberta().to(device)
    print('Total Parameters:', sum([p.nelement() for p in roberta.parameters()]))
    roberta.load_pretrain()

    dataset = RobertaDataSet(CorpusPath)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True)
    testset = RobertaTestSet(TestPath)

    optim = Adam(roberta.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        roberta.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            label = data['token_ids_labels']
            onehot_label = data['onehot_labels'].float()
            mlm_output = roberta(input_token, segment_ids).permute(0, 2, 1)
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()
        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(roberta.cpu(), output_path)
        roberta.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            roberta.eval()

            test_count = 0
            test_acc = 0
            for test_data in testset:
                input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                input_token_list = input_token.tolist()
                label_list = test_data['token_ids_labels'].tolist()
                input_len = len([x for x in input_token_list[0] if x]) - 2
                mlm_output = roberta(input_token, segment_ids)[:, 1:input_len + 1, :]
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()

                # 累计数值
                test_count += input_len
                for i, label in enumerate(label_list):
                    if i == input_len:
                        break
                    if label in output_topk[i]:
                        test_acc += 1
            if test_count:
                acc = float(test_acc) / float(test_count)
                acc = round(acc, 2)
                print('纠正正确率：%s\n' % acc)
