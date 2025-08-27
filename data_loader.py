# data_loader.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import config

class RecommendationDataset(Dataset):
    """自定义PyTorch数据集"""
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        query = str(row['dataset_description'])
        candidate = str(row['idea'])
        
        return {
            'query': query,
            'candidate': candidate
        }

def create_data_loaders(data_file):
    """加载数据、划分并创建DataLoaders"""
    df = pd.read_csv(data_file)
    df.dropna(subset=['dataset_description', 'idea'], inplace=True)
    
    # 划分训练、验证和测试集
    train_df, temp_df = train_test_split(
        df, train_size=config.TRAIN_RATIO, random_state=42
    )
    val_size = config.VALIDATION_RATIO / (config.VALIDATION_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size, random_state=42
    )
    
    # 使用SentenceTransformer的分词器
    # 在这里初始化是为了能够传递给Dataset，但实际上编码将在模型内部完成
    tokenizer = SentenceTransformer(config.BASE_MODEL).tokenizer

    train_dataset = RecommendationDataset(train_df, tokenizer)
    val_dataset = RecommendationDataset(val_df, tokenizer)
    test_dataset = RecommendationDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, df # 返回完整的df以便后续建立索引

if __name__ == '__main__':
    # 测试数据加载
    train_loader, val_loader, _, _ = create_data_loaders(config.DATA_FILE)
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    sample_batch = next(iter(train_loader))
    print("\n一批数据的样本:")
    print("Queries:", sample_batch['query'][:2])
    print("Candidates:", sample_batch['candidate'][:2])