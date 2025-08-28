import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import faiss

import config
from data_loader import create_data_loaders
from model import TwoTowerModel
from loss import contrastive_loss, triplet_loss

def evaluate_retrieval(model, eval_loader, all_candidates, device, top_k_list=[1, 5, 10]):
    """
    优化版的评估函数，使用FAISS进行高效检索并采用向量化计算排名。
    
    Args:
        model: 待评估的模型
        eval_loader: 验证集或测试集的DataLoader
        all_candidates: 包含所有候选idea文本的列表
        device: 'cuda' or 'cpu'
        top_k_list: 一个列表，例如 [1, 5, 10]，用于计算Recall@K
    """
    model.eval()
    
    # ==================== 1. 一次性编码所有候选并构建FAISS索引 ====================
    # 这部分是此函数的核心优化，避免在外部循环中重复编码。
    all_candidate_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_candidates), 256), desc="Encoding all candidates for eval"):
            batch_ideas = all_candidates[i:i + 256]
            embeddings = model.candidate_encoder(batch_ideas)
            all_candidate_embeddings.append(embeddings.cpu().numpy().astype('float32'))
            
    all_candidate_embeddings = np.vstack(all_candidate_embeddings)
    
    # 构建FAISS索引
    index = faiss.IndexFlatIP(model.query_encoder.dense_layers[-1].out_features)
    faiss.normalize_L2(all_candidate_embeddings)
    index.add(all_candidate_embeddings)

    # ==================== 2. 准备地面实况 (Ground Truth) ====================
    candidate_map = {idea: i for i, idea in enumerate(all_candidates)}
    
    all_ground_truth_indices = []
    all_query_embeddings_list = []

    print("Preparing ground truth and encoding queries...")
    with torch.no_grad():
        for batch in eval_loader:
            ground_truth_ideas = list(batch['candidate'])
            # 将文本形式的ground truth转换为在候选库中的索引
            indices = [candidate_map.get(idea, -1) for idea in ground_truth_ideas]
            all_ground_truth_indices.extend(indices)
            
            # 同时编码查询
            query_embeddings = model.query_encoder(list(batch['query']))
            all_query_embeddings_list.append(query_embeddings.cpu().numpy().astype('float32'))
    
    all_query_embeddings = np.vstack(all_query_embeddings_list)
    faiss.normalize_L2(all_query_embeddings)
    
    # ==================== 3. 使用FAISS进行批量搜索 ====================
    max_k = max(top_k_list)
    print(f"Performing FAISS search for top {max_k} candidates...")
    # D是距离（相似度），I是索引
    _distances, top_k_indices = index.search(all_query_embeddings, max_k)
    
    # ==================== 4. 向量化计算排名和指标 ====================
    # 将ground truth索引向量的形状从 (n,) 变为 (n, 1) 以便进行广播比较
    ground_truth_vec = np.array(all_ground_truth_indices).reshape(-1, 1)
    
    # 创建一个布尔矩阵，表示每个召回的item是否是ground truth
    # 结果形状为 [num_queries, max_k]
    matches = (top_k_indices == ground_truth_vec)
    
    # 找到每个查询的第一个匹配项的排名（列索引）
    # np.argmax在找到第一个True后就会停止，其索引即为排名-1
    # 如果某一行全是False（即未命中），argmax会返回0，我们需要后续处理
    ranks = np.argmax(matches, axis=1) + 1
    
    # 处理未命中的情况：如果一行中没有True，那么matches.any(axis=1)会是False
    # 我们将这些未命中样本的排名设为0或一个大数，以便在计算MRR时不影响结果
    not_found_mask = ~matches.any(axis=1)
    ranks[not_found_mask] = 0
    
    # 计算MRR
    reciprocal_ranks = np.where(ranks > 0, 1.0 / ranks, 0)
    mrr = reciprocal_ranks.mean()
    
    # 计算Recall@K
    total_samples = len(all_ground_truth_indices)
    recall_at_k = {}
    for k in top_k_list:
        # 如果排名在1到k之间（包括k），则认为是在top-k内命中
        hits = np.sum((ranks > 0) & (ranks <= k))
        recall_at_k[f"Recall@{k}"] = hits / total_samples
        
    return recall_at_k, mrr

def train_one_epoch(model, data_loader, optimizer, device, mode):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc=f"Training ({mode})"):
        optimizer.zero_grad()
        
        if mode == 'round1':
            query_texts = list(batch['query'])
            candidate_texts = list(batch['candidate'])
            query_embeddings, candidate_embeddings = model(query_texts, candidate_texts)
            loss = contrastive_loss(query_embeddings, candidate_embeddings, config.TEMPERATURE)
        else: # round2
            query_texts = list(batch['query'])
            positive_texts = list(batch['positive'])
            negative_texts = list(batch['negative'])
            q_emb, p_emb, n_emb = model(query_texts, positive_texts, negative_texts)
            loss = triplet_loss(q_emb, p_emb, n_emb, config.TRIPLET_MARGIN)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def main():
    parser = argparse.ArgumentParser(description="Train a two-tower model in two stages.")
    parser.add_argument('--mode', type=str, required=True, choices=['round1', 'round2'],
                        help="Training stage: 'round1' for contrastive pre-training, 'round2' for triplet fine-tuning.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TwoTowerModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-7)

    if args.mode == 'round1':
        print("--- Starting Training: Round 1 (Contrastive) ---")
        train_loader, val_loader, _, full_df = create_data_loaders(config.DATA_FILE)
        model_save_path = config.MODEL_SAVE_PATH_ROUND1
    else: # round2
        print("--- Starting Training: Round 2 (Triplet Fine-tuning) ---")
        # 加载第一阶段的模型权重作为起点
        print(f"Loading weights from {config.MODEL_SAVE_PATH_ROUND1}...")
        try:
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_ROUND1))
        except FileNotFoundError:
            print("Error: Round 1 model not found. Please run '--mode round1' first.")
            return

        triplet_df = pd.read_csv(config.TRIPLET_DATA_PATH)
        train_loader, val_loader, _, full_df = create_data_loaders(config.DATA_FILE, train_df_override=triplet_df)
        model_save_path = config.MODEL_SAVE_PATH_FINAL

    all_unique_ideas = full_df['idea'].dropna().unique().tolist()
    best_mrr = 0.0

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.mode)
        scheduler.step() # 更新学习率

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.8f}")
        
        # 评估
        recall_at_k, mrr = evaluate_retrieval(model, val_loader, all_unique_ideas, device)
        print(f"Validation MRR = {mrr:.4f}")
        for metric, value in recall_at_k.items():
            print(f"Validation {metric} = {value:.4f}")

        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with MRR: {best_mrr:.4f} to {model_save_path}")

    print("\n--- Training Complete ---")
    # 此处可以添加最终在test set上的测试逻辑

if __name__ == '__main__':
    main()