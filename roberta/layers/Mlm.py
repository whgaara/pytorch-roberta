import torch
import torch.nn as nn


class Mlm(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Mlm, self).__init__()
        self.mlm_dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, feedforward_x, embedding_table=None):
        if embedding_table is not None:
            embedding_table = embedding_table.transpose(0, 1)
            feedforward_x = torch.matmul(feedforward_x, embedding_table)
        else:
            feedforward_x = self.mlm_dense(feedforward_x)
        return feedforward_x
