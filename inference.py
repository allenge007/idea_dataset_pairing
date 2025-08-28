# inference.py (已修改，增加了命令行参数控制)

import faiss
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import argparse  # 导入argparse

import config
from model import TwoTowerModel

def build_faiss_index(model, all_ideas):
    """
    使用候选塔对所有idea进行编码并构建FAISS索引 (内存优化版)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if not all_ideas:
        print("Error: The list of candidate ideas is empty.")
        return

    print(f"Building FAISS index for {len(all_ideas)} unique ideas...")
    
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

    with torch.no_grad():
        for i in tqdm(range(0, len(all_ideas), config.BATCH_SIZE), desc="Encoding ideas and adding to index"):
            batch_ideas = all_ideas[i:i+config.BATCH_SIZE]
            embeddings = model.candidate_encoder(batch_ideas)
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            faiss.normalize_L2(embeddings_np)
            index.add(embeddings_np)

    print(f"Index built successfully with {index.ntotal} vectors.")
    
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CANDIDATE_IDEAS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_ideas, f, ensure_ascii=False, indent=4)
        
    print(f"FAISS index saved to {config.FAISS_INDEX_PATH}")
    print(f"Candidate ideas saved to {config.CANDIDATE_IDEAS_PATH}")

def search(query_text, top_k=5):
    """根据查询文本在FAISS索引中搜索最相似的idea"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和索引
    model = TwoTowerModel()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_FINAL))
    model.to(device)
    model.eval()
    
    # 确认文件存在
    try:
        index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.CANDIDATE_IDEAS_PATH, 'r', encoding='utf-8') as f:
            candidate_ideas = json.load(f)
    except Exception as e:
        print(f"Error loading index or candidate ideas: {e}")
        print("Please run the script with '--mode build' first to create the index.")
        return None
        
    # 编码查询
    with torch.no_grad():
        query_embedding = model.query_encoder([query_text]).cpu().numpy().astype('float32')
    
    faiss.normalize_L2(query_embedding)
    
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
    # 创建一个解析器
    parser = argparse.ArgumentParser(description="Build FAISS index or perform search.")
    # 添加一个参数来决定运行模式，默认为 'search'
    parser.add_argument('--mode', type=str, default='search', choices=['build', 'search'],
                        help="Run mode: 'build' to create the index, 'search' to perform a query.")
    
    args = parser.parse_args()

    # 根据模式执行不同的代码块
    if args.mode == 'build':
        print("--- Mode: Build Index ---")
        # 加载训练好的模型
        print("Loading trained model...")
        model = TwoTowerModel()
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_FINAL))
        
        # 加载全部数据以获取候选idea
        print("Loading data to get unique ideas...")
        full_df = pd.read_csv(config.DATA_FILE)
        unique_ideas = full_df['idea'].dropna().unique().tolist()
        
        # 构建索引
        build_faiss_index(model, unique_ideas)
        
    elif args.mode == 'search':
        print("--- Mode: Search ---")
        # 示例查询
        sample_query = "一个包含大量人脸图像信息的数据集，用于人脸识别。"
        
        recommendations = search(sample_query, top_k=3)
        
        if recommendations:
            print(f"Query: \"{sample_query}\"")
            print("\nTop 3 Recommended Ideas:")
            for rec in recommendations:
                print(f"  - Idea: {rec['idea'][:]}...")
                print(f"    Similarity: {rec['similarity']:.4f}")