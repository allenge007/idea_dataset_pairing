# train.py

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

import config
from data_loader import create_data_loaders
from model import TwoTowerModel
from loss import contrastive_loss

def train_one_epoch(model, data_loader, optimizer, device):
    """训练逻辑保持不变"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        query_texts = list(batch['query'])
        candidate_texts = list(batch['candidate'])
        
        query_embeddings, candidate_embeddings = model(query_texts, candidate_texts)
        
        loss = contrastive_loss(query_embeddings, candidate_embeddings, config.TEMPERATURE)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate_retrieval(model, eval_loader, all_candidate_embeddings, all_candidates, device, top_k=[1, 5, 10]):
    """
    评估函数，执行全库检索。
    
    Args:
        model: 待评估的模型
        eval_loader: 验证集或测试集的DataLoader
        all_candidate_embeddings: 包含所有候选idea向量的Tensor
        all_candidates: 包含所有候选idea文本的列表
        device: 'cuda' or 'cpu'
        top_k: 一个列表，例如 [1, 5, 10]，用于计算Recall@1, Recall@5, Recall@10
    """
    model.eval()
    
    # 将所有候选idea的文本映射到一个快速查找的字典中，key是idea文本，value是其在向量库中的索引
    candidate_map = {idea: i for i, idea in enumerate(all_candidates)}
    
    total_hits = {k: 0 for k in top_k}
    total_mrr = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            query_texts = list(batch['query'])
            ground_truth_ideas = list(batch['candidate'])
            
            # 1. 编码查询
            query_embeddings = model.query_encoder(query_texts) # Shape: [batch_size, dim]
            
            # 2. 计算与所有候选向量的相似度
            #    (query_embeddings @ all_candidate_embeddings.T) -> [batch_size, num_candidates]
            similarity_scores = torch.matmul(query_embeddings, all_candidate_embeddings.T)
            
            # 3. 对每个查询的相似度进行排序，得到排名最高的候选索引
            #    torch.topk返回 (values, indices)
            _, top_k_indices = torch.topk(similarity_scores, max(top_k), dim=1) # Shape: [batch_size, max_top_k]
            
            top_k_indices = top_k_indices.cpu().numpy()

            for i in range(len(query_texts)):
                ground_truth_idea = ground_truth_ideas[i]
                
                # 获取真实idea在候选库中的索引
                if ground_truth_idea not in candidate_map:
                    continue # 如果验证集中的某个idea不在总候选库中，跳过
                
                ground_truth_idx = candidate_map[ground_truth_idea]
                
                # 查找真实idea的排名
                # np.where返回一个元组，我们需要第一个数组
                rank_list = np.where(top_k_indices[i] == ground_truth_idx)[0]
                
                if len(rank_list) > 0:
                    rank = rank_list[0] + 1  # 排名从1开始
                    total_mrr += 1.0 / rank
                    for k in top_k:
                        if rank <= k:
                            total_hits[k] += 1
            
            total_samples += len(query_texts)

    recall_at_k = {f"Recall@{k}": hits / total_samples for k, hits in total_hits.items()}
    mrr = total_mrr / total_samples
    
    return recall_at_k, mrr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 准备数据加载器和完整的候选库
    train_loader, val_loader, test_loader, full_df = create_data_loaders(config.DATA_FILE)
    
    # 获取所有唯一的idea作为候选库
    all_unique_ideas = full_df['idea'].dropna().unique().tolist()
    print(f"Total unique candidate ideas: {len(all_unique_ideas)}")
    
    # 2. 初始化模型和优化器
    model = TwoTowerModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    best_mrr = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS} ---")
        
        # 训练阶段保持不变
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        
        # 验证阶段使用新的评估函数
        print("Starting validation...")
        # 首先，编码所有候选idea
        model.eval()
        with torch.no_grad():
            candidate_embeddings_list = []
            for i in tqdm(range(0, len(all_unique_ideas), config.BATCH_SIZE), desc="Encoding all candidates"):
                batch_ideas = all_unique_ideas[i:i+config.BATCH_SIZE]
                embeddings = model.candidate_encoder(batch_ideas)
                candidate_embeddings_list.append(embeddings)
            all_candidate_embeddings = torch.cat(candidate_embeddings_list, dim=0)

        # 执行检索评估
        recall_at_k, mrr = evaluate_retrieval(model, val_loader, all_candidate_embeddings, all_unique_ideas, device)
        
        print(f"Validation MRR = {mrr:.4f}")
        for metric, value in recall_at_k.items():
            print(f"Validation {metric} = {value:.4f}")
        
        # 使用MRR作为模型选择的标准
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with MRR: {best_mrr:.4f} to {config.MODEL_SAVE_PATH}")

    # --- 最终测试 ---
    print("\n--- Testing the best model on the test set ---")
    # 加载最佳模型
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # 再次为测试集编码所有候选idea（因为模型权重已更新）
    model.eval()
    with torch.no_grad():
        candidate_embeddings_list = []
        for i in tqdm(range(0, len(all_unique_ideas), config.BATCH_SIZE), desc="Encoding all candidates for test"):
            batch_ideas = all_unique_ideas[i:i+config.BATCH_SIZE]
            embeddings = model.candidate_encoder(batch_ideas)
            candidate_embeddings_list.append(embeddings)
        all_candidate_embeddings = torch.cat(candidate_embeddings_list, dim=0)
    
    # 在测试集上进行最终评估
    recall_at_k, mrr = evaluate_retrieval(model, test_loader, all_candidate_embeddings, all_unique_ideas, device)
    print("\nFinal Test Results:")
    print(f"Test MRR = {mrr:.4f}")
    for metric, value in recall_at_k.items():
        print(f"Test {metric} = {value:.4f}")


if __name__ == '__main__':
    main()