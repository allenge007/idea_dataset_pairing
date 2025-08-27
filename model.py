# model.py (已修复)

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import config

class Encoder(nn.Module):
    """
    编码塔，包含一个SBERT基础模型和几个全连接层。
    """
    def __init__(self, base_model_name, sbert_output_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.sbert = SentenceTransformer(base_model_name)
        # SBERT模型本身就包含了一个tokenizer
        # self.tokenizer = self.sbert.tokenizer 
        
        self.dense_layers = nn.Sequential(
            nn.Linear(sbert_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, texts):
        # 错误的方式： .encode() 是为推理设计的，可能在内部使用 torch.no_grad()
        # embeddings = self.sbert.encode(texts, convert_to_tensor=True, device=self.sbert.device)
        
        # 将SBERT模型作为标准的PyTorch模块使用
        # 1. 手动进行分词
        #    padding=True: 将批次内的句子填充到同样长度
        #    truncation=True: 将过长的句子截断
        #    return_tensors='pt': 返回PyTorch张量
        features = self.sbert.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # 2. 将分词结果移动到正确的设备 (GPU or CPU)
        features = {key: value.to(self.sbert.device) for key, value in features.items()}
        
        # 3. 执行SBERT模型的前向传播
        #    这会确保计算图被正确构建
        sbert_output = self.sbert(features)
        
        # 4. 从输出字典中提取句子嵌入向量
        embeddings = sbert_output['sentence_embedding']
        
        # 5. 将包含梯度信息的向量传入全连接层
        return self.dense_layers(embeddings)

class TwoTowerModel(nn.Module):
    """
    双塔模型，包含查询塔和候选塔。
    """
    def __init__(self):
        super(TwoTowerModel, self).__init__()
        self.query_encoder = Encoder(
            base_model_name=config.BASE_MODEL,
            sbert_output_dim=config.SBERT_OUTPUT_DIM,
            embedding_dim=config.EMBEDDING_DIM
        )
        self.candidate_encoder = Encoder(
            base_model_name=config.BASE_MODEL,
            sbert_output_dim=config.SBERT_OUTPUT_DIM,
            embedding_dim=config.EMBEDDING_DIM
        )

    def forward(self, query_texts, candidate_texts):
        query_embedding = self.query_encoder(query_texts)
        candidate_embedding = self.candidate_encoder(candidate_texts)
        return query_embedding, candidate_embedding