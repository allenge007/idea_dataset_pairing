import pandas as pd
import torch
import faiss
from tqdm import tqdm
import numpy as np

import config
from model import TwoTowerModel
from data_loader import create_data_loaders

def main():
    print("Starting hard negative mining process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载第一阶段训练好的模型
    print("Loading the best model from Round 1...")
    model = TwoTowerModel()
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH_ROUND1))
    model.to(device)
    model.eval()

    # 2. 准备数据和候选库
    # 我们只需要原始训练集部分来进行挖掘
    df = pd.read_csv(config.DATA_FILE).dropna(subset=['dataset_description', 'idea'])
    train_df, _ = torch.utils.data.random_split(df, [config.TRAIN_RATIO, 1 - config.TRAIN_RATIO], generator=torch.Generator().manual_seed(42))
    train_df = pd.DataFrame(train_df.dataset.iloc[train_df.indices])


    all_unique_ideas = df['idea'].unique().tolist()
    idea_to_idx = {idea: i for i, idea in enumerate(all_unique_ideas)}
    
    # 3. 编码所有候选idea并构建FAISS索引
    print("Encoding all candidate ideas and building FAISS index...")
    candidate_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_unique_ideas), config.BATCH_SIZE), desc="Encoding ideas"):
            batch_ideas = all_unique_ideas[i:i + config.BATCH_SIZE]
            embeddings = model.candidate_encoder(batch_ideas)
            candidate_embeddings.append(embeddings.cpu().numpy())
    
    candidate_embeddings = np.vstack(candidate_embeddings).astype('float32')
    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    faiss.normalize_L2(candidate_embeddings)
    index.add(candidate_embeddings)

    # 4. 为训练集中的每个查询挖掘难负例
    print("Mining hard negatives for each query in the training set...")
    triplet_data = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining"):
        query_text = str(row['dataset_description'])
        positive_idea = str(row['idea'])

        with torch.no_grad():
            query_embedding = model.query_encoder([query_text]).cpu().numpy().astype('float32')
        faiss.normalize_L2(query_embedding)

        # 搜索K+1个最近邻，因为第一个很可能是正样本自己
        _, indices = index.search(query_embedding, config.TOP_K_HARD_NEGATIVES + 1)
        
        positive_idea_idx = idea_to_idx.get(positive_idea)

        for idx in indices[0]:
            if idx != positive_idea_idx: # 确保不是正样本
                hard_negative_idea = all_unique_ideas[idx]
                triplet_data.append({
                    'query': query_text,
                    'positive': positive_idea,
                    'negative': hard_negative_idea
                })
                if len(triplet_data) % 1000 == 0:
                    print(f"Generated {len(triplet_data)} triplets...")


    # 5. 保存三元组数据
    triplet_df = pd.DataFrame(triplet_data)
    triplet_df.to_csv(config.TRIPLET_DATA_PATH, index=False)
    print(f"Hard negative mining complete. Saved {len(triplet_df)} triplets to {config.TRIPLET_DATA_PATH}")

if __name__ == '__main__':
    main()