# loss.py

import torch
import torch.nn.functional as F

def contrastive_loss(query_embeddings, candidate_embeddings, temperature=0.05):
    """
    计算对比损失，使用批内负采样。
    query_embeddings: (batch_size, embedding_dim)
    candidate_embeddings: (batch_size, embedding_dim)
    """
    # 归一化嵌入向量
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    
    # 计算点积相似度 (batch_size, batch_size)
    similarity_matrix = torch.matmul(query_embeddings, candidate_embeddings.T)
    
    # 构造标签，对角线上的元素是正样本
    labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
    
    # 使用交叉熵损失函数
    # 它等价于计算InfoNCE损失
    loss = F.cross_entropy(similarity_matrix / temperature, labels)
    
    return loss