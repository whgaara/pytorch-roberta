import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from roberta.data.dataset import *
from roberta.layers.Roberta import Roberta


if __name__ == '__main__':
    roberta = Roberta().to(device)
    roberta.load_pretrain()

    dataset = RobertaDataSet(CorpusPath)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True)
    optim = Adam(roberta.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)
    print('Total Parameters:', sum([p.nelement() for p in roberta.parameters()]))

    # train
    for epoch in range(Epochs):
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            label = data['token_ids_labels']
            onehot_label = data['onehot_labels'].float()

            mlm_output = roberta(input_token, segment_ids).permute(0, 2, 1)
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            print('\tmask loss:%s' % print_loss)
            optim.zero_grad()
            mask_loss.backward()
            optim.step()

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(roberta.cpu(), output_path)
        roberta.to(device)
        print('EP:%d Model Saved on:' % epoch, output_path)
