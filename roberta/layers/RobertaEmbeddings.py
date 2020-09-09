import torch.nn as nn


class RobertaEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, dropout_prob=0.1):
        super(RobertaEmbeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.type_embeddings = nn.Embedding(2, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)
        self.emb_normalization = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input_token, segment_ids):
        token_embeddings = self.token_embeddings(input_token)
        type_embeddings = self.type_embeddings(segment_ids)
        postion_embeddings = self.position_embeddings.weight
        embedding_x = token_embeddings + type_embeddings + postion_embeddings
        embedding_x = self.emb_normalization(embedding_x)
        embedding_x = self.emb_dropout(embedding_x)
        return embedding_x
