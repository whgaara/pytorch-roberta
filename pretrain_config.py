import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 文件路径 ## #
ModelSavePath = 'checkpoint/finetune'
SourcePath = 'data/src_data'
CorpusPath = 'data/train_data'
TestPath = 'data/test_data/test.txt'
VocabPath = 'checkpoint/pretrain/vocab.txt'
FinetunePath = 'checkpoint/finetune/roberta_trained.model'
PretrainPath = 'checkpoint/pretrain/pytorch_model.bin'
PronunciationPath = 'data/char_meta.txt'

# Debug开关
Debug = False

# 任务模式
ModelClass = 'RobertaMlm'

# ## 训练参数 ## #
if ModelClass == 'Bert':
    RepeatNum = 1
    BatchSize = 8
    SentenceLength = 128
if ModelClass == 'RobertaMlm':
    RepeatNum = 10
    BatchSize = 1
    SentenceLength = 512

Epochs = 16
DropOut = 0.1
MaskRate = 0.15
VocabSize = 21128
HiddenSize = 768
LearningRate = 5e-4
# transformer块个数
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12

# 参数名配对
local2target_emb = {
    'roberta_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'roberta_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
    'roberta_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'roberta_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.weight',
    'roberta_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.bias'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.weight',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.bias',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.weight',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.bias',
}


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
