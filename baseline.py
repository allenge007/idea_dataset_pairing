import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import platform
import faiss
# 导入必要的库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

import config
from data_loader import create_data_loaders

def calculate_metrics(ranks, total_samples, top_k_list=[1, 5, 10]):
    """根据排名列表计算MRR和Recall@K"""
    ranks = np.array(ranks)
    
    # 处理未命中的情况 (rank=0)
    not_found_mask = (ranks == 0)
    ranks[not_found_mask] = 0

    reciprocal_ranks = np.where(ranks > 0, 1.0 / ranks, 0)
    mrr = reciprocal_ranks.mean()

    recall_at_k = {}
    for k in top_k_list:
        hits = np.sum((ranks > 0) & (ranks <= k))
        recall_at_k[f"Recall@{k}"] = hits / total_samples
        
    return recall_at_k, mrr

def run_tfidf_baseline(eval_loader, all_candidates):
    """
    Baseline 1: TF-IDF + Cosine Similarity
    """
    print("\n--- Running TF-IDF Baseline ---")
    
    # 1. 创建并训练TfidfVectorizer
    print("Fitting TfidfVectorizer on all candidate ideas...")
    vectorizer = TfidfVectorizer()
    candidate_vectors = vectorizer.fit_transform(all_candidates)

    candidate_map = {idea: i for i, idea in enumerate(all_candidates)}
    
    all_ranks = []
    total_samples = 0

    for batch in tqdm(eval_loader, desc="Evaluating TF-IDF"):
        query_texts = list(batch['query'])
        ground_truth_ideas = list(batch['candidate'])
        
        # 2. 转换查询文本
        query_vectors = vectorizer.transform(query_texts)
        
        # 3. 计算余弦相似度
        similarity_scores = cosine_similarity(query_vectors, candidate_vectors)
        
        # 4. 获取排名
        # argsort默认升序，我们需要降序，所以对负数排序
        sorted_indices = np.argsort(-similarity_scores, axis=1)
        
        gt_indices_batch = [candidate_map.get(idea, -1) for idea in ground_truth_ideas]

        for i, gt_idx in enumerate(gt_indices_batch):
            if gt_idx == -1:
                all_ranks.append(0) # 无法找到的样本，排名为0
                continue
            
            rank_list = np.where(sorted_indices[i] == gt_idx)[0]
            rank = rank_list[0] + 1 if len(rank_list) > 0 else 0
            all_ranks.append(rank)
            
        total_samples += len(query_texts)

    return calculate_metrics(all_ranks, total_samples)

def evaluate_zero_shot_sbert(model, eval_loader, all_candidates, device, top_k_list=[1, 5, 10]):
    """
    专门为 Zero-Shot SBERT Baseline 设计的评估函数。
    它直接使用 model.encode() API，不依赖于双塔结构。
    """
    print("\nRunning evaluation tailored for a raw SentenceTransformer model.")
    model.eval()

    # 1. 一次性编码所有候选
    print("Encoding all candidate ideas for eval (Zero-Shot)...")
    # 使用 .encode() API，它非常适合推理
    all_candidate_embeddings = model.encode(
        all_candidates, 
        batch_size=256, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        device=device
    )
    # 归一化，并转为numpy准备FAISS
    all_candidate_embeddings = torch.nn.functional.normalize(all_candidate_embeddings, p=2, dim=1)
    all_candidate_embeddings_np = all_candidate_embeddings.cpu().numpy().astype('float32')

    # 2. 构建FAISS索引
    index = faiss.IndexFlatIP(all_candidate_embeddings_np.shape[1])
    index.add(all_candidate_embeddings_np)
    
    candidate_map = {idea: i for i, idea in enumerate(all_candidates)}
    
    all_query_texts = []
    all_ground_truth_indices = []

    print("Preparing ground truth and collecting queries...")
    for batch in eval_loader:
        all_query_texts.extend(list(batch['query']))
        gt_ideas = list(batch['candidate'])
        all_ground_truth_indices.extend([candidate_map.get(idea, -1) for idea in gt_ideas])

    # 3. 批量编码所有查询并搜索
    print("Encoding all queries and performing FAISS search (Zero-Shot)...")
    query_embeddings = model.encode(
        all_query_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    query_embeddings_np = query_embeddings.cpu().numpy().astype('float32')
    
    _, top_k_indices = index.search(query_embeddings_np, max(top_k_list))

    # 4. 向量化计算指标 (这部分逻辑可以复用)
    ground_truth_vec = np.array(all_ground_truth_indices).reshape(-1, 1)
    matches = (top_k_indices == ground_truth_vec)
    ranks = np.argmax(matches, axis=1) + 1
    not_found_mask = ~matches.any(axis=1)
    ranks[not_found_mask] = 0
    
    reciprocal_ranks = np.where(ranks > 0, 1.0 / ranks, 0)
    mrr = reciprocal_ranks.mean()

    total_samples = len(all_ground_truth_indices)
    recall_at_k = {}
    for k in top_k_list:
        hits = np.sum((ranks > 0) & (ranks <= k))
        recall_at_k[f"Recall@{k}"] = hits / total_samples
        
    return recall_at_k, mrr

def run_zero_shot_sbert_baseline(eval_loader, all_candidates, device):
    """
    Baseline 2: Zero-Shot Pre-trained SBERT
    """
    print("\n--- Running Zero-Shot SBERT Baseline ---")

    print(f"Loading pre-trained model: {config.BASE_MODEL}...")
    # 1. 加载一个未经微调的原始SentenceTransformer模型
    model = SentenceTransformer(config.BASE_MODEL, device=device)

    # 2. 调用为它量身定做的评估函数
    return evaluate_zero_shot_sbert(model, eval_loader, all_candidates, device)


def main():
    parser = argparse.ArgumentParser(description="Run baseline models for evaluation.")
    parser.add_argument('--mode', type=str, required=True, choices=['tfidf', 'zero_shot_sbert'],
                        help="Which baseline to run: 'tfidf' or 'zero_shot_sbert'.")
    args = parser.parse_args()

    # 准备数据
    # 我们只关心验证集和测试集，这里用val_loader进行演示
    _, val_loader, _, full_df = create_data_loaders(config.DATA_FILE)
    all_unique_ideas = full_df['idea'].dropna().unique().tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'tfidf':
        recall_at_k, mrr = run_tfidf_baseline(val_loader, all_unique_ideas)
    elif args.mode == 'zero_shot_sbert':
        # 确保 inference_utils.py 在您的项目中
        recall_at_k, mrr = run_zero_shot_sbert_baseline(val_loader, all_unique_ideas, device)
    
    print("\n--- Baseline Results ---")
    print(f"Mode: {args.mode}")
    print(f"Validation MRR = {mrr:.4f}")
    for metric, value in recall_at_k.items():
        print(f"Validation {metric} = {value:.4f}")

if __name__ == '__main__':
    main()