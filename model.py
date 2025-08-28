import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import config

class Encoder(nn.Module):
    """
    编码塔，包含一个SBERT基础模型和几个全连接层。
    已加入BatchNorm来稳定训练。
    """
    def __init__(self, base_model_name, sbert_output_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.sbert = SentenceTransformer(base_model_name)
        
        self.dense_layers = nn.Sequential(
            nn.Linear(sbert_output_dim, 512),
            nn.BatchNorm1d(512), # 加入批量归一化
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), # 加入批量归一化
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, texts):
        features = self.sbert.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        features = {key: value.to(self.sbert.device) for key, value in features.items()}
        sbert_output = self.sbert(features)
        embeddings = sbert_output['sentence_embedding']
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

    def forward(self, query_texts, candidate_texts, negative_texts=None):
        """
        前向传播。如果提供了negative_texts，则同时编码它们以用于三元组损失。
        """
        query_embedding = self.query_encoder(query_texts)
        candidate_embedding = self.candidate_encoder(candidate_texts)
        
        if negative_texts:
            negative_embedding = self.candidate_encoder(negative_texts)
            return query_embedding, candidate_embedding, negative_embedding
        
        return query_embedding, candidate_embedding