import torch
import torch.nn.functional as F

def contrastive_loss(query_embeddings, candidate_embeddings, temperature=0.07):
    """计算对比损失，使用批内负采样。"""
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
    similarity_matrix = torch.matmul(query_embeddings, candidate_embeddings.T)
    labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
    return F.cross_entropy(similarity_matrix / temperature, labels)

def triplet_loss(anchor, positive, negative, margin=0.5):
    """计算三元组损失。"""
    loss_func = torch.nn.TripletMarginLoss(margin=margin, p=2)
    return loss_func(anchor, positive, negative)