# inference.py

import faiss
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

import config
from model import TwoTowerModel

def build_faiss_index(model, all_ideas):
    """使用候选塔对所有idea进行编码并构建FAISS索引"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # print("Encoding candidate ideas...")
    candidate_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_ideas), config.BATCH_SIZE), desc="Encoding ideas"):
            batch_ideas = all_ideas[i:i+config.BATCH_SIZE]
            embeddings = model.candidate_encoder(batch_ideas)
            candidate_embeddings.append(embeddings.cpu().numpy())

    # 明确将数据类型转换为 float32，这是FAISS所期望的
    candidate_embeddings = np.vstack(candidate_embeddings).astype('float32')
    
    # 检查一下是否有数据
    if candidate_embeddings.shape[0] == 0:
        print("Error: No candidate embeddings were generated.")
        return

    # 构建FAISS索引
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)  # IP = Inner Product (点积)
    # 归一化向量，使得点积等价于余弦相似度
    faiss.normalize_L2(candidate_embeddings)
    index.add(candidate_embeddings)

def search(query_text, top_k=5):
    """根据查询文本在FAISS索引中搜索最相似的idea"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和索引
    model = TwoTowerModel()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.to(device)
    model.eval()
    
    index = faiss.read_index(config.FAISS_INDEX_PATH)
    with open(config.CANDIDATE_IDEAS_PATH, 'r', encoding='utf-8') as f:
        candidate_ideas = json.load(f)
        
    # 编码查询
    with torch.no_grad():
        query_embedding = model.query_encoder([query_text]).cpu().numpy()
    
    # 归一化查询向量
    faiss.normalize_L2(query_embedding)
    
    # 执行搜索
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(top_k):
        result_idx = indices[0][i]
        results.append({
            "idea": candidate_ideas[result_idx],
            "similarity": float(distances[0][i])
        })
        
    return results

if __name__ == '__main__':
    # --- 步骤1: 构建并保存FAISS索引 (通常离线执行一次) ---
    print("Building FAISS index...")
    full_df = pd.read_csv(config.DATA_FILE)
    # 获取所有唯一的idea作为候选库
    unique_ideas = full_df['idea'].dropna().unique().tolist()

    # 加载训练好的模型
    trained_model = TwoTowerModel()
    trained_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    print("Model loaded successfully.")

    build_faiss_index(trained_model, unique_ideas)
    
    # --- 步骤2: 执行实时推荐 ---
    print("\n--- Performing a search ---")
    # 示例查询
    sample_query = "一个包含大量用户评论和商家信息的数据集，常用于情感分析和推荐系统研究。"
    
    recommendations = search(sample_query, top_k=3)
    
    print(f"Query: \"{sample_query}\"")
    print("\nTop 3 Recommended Ideas:")
    for rec in recommendations:
        print(f"  - Idea: {rec['idea'][:100]}...")
        print(f"    Similarity: {rec['similarity']:.4f}")