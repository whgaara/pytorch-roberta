import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## 文件路径 ## #
ModelSavePath = 'checkpoint/finetune'
CorpusPath = 'data'
VocabPath = 'checkpoint/pretrain/vocab.txt'
FinetunePath = 'checkpoint/finetune/roberta_trained.model'

# ## 训练参数 ## #
Epochs = 16
MaskRate = 0.15
BatchSize = 16
RepeatNum = 10
HiddenSize = 768
IntermediateSize = 3072
VocabSize = 21128
AttentionHeadNum = 12
HiddenLayerNum = 12
SentenceLength = 512
DropOut = 0.1

LearningRate = 1e-3
NumWarmupSteps = 0
