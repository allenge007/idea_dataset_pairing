import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config

class RecommendationDataset(Dataset):
    """用于对比学习的原始数据集 (query, candidate)"""
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'query': str(row['dataset_description']),
            'candidate': str(row['idea'])
        }

class TripletDataset(Dataset):
    """用于三元组损失的数据集 (query, positive_candidate, negative_candidate)"""
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'query': str(row['query']),
            'positive': str(row['positive']),
            'negative': str(row['negative'])
        }
        
def create_data_loaders(data_file, train_df_override=None):
    """
    加载数据、划分并创建DataLoaders。
    train_df_override: 允许传入一个已经处理过的训练集DataFrame（例如，包含难负例的）。
    """
    df = pd.read_csv(data_file)
    df.dropna(subset=['dataset_description', 'idea'], inplace=True)
    
    if train_df_override is None:
        train_df, temp_df = train_test_split(
            df, train_size=config.TRAIN_RATIO, random_state=42
        )
    else:
        # 如果提供了外部训练集，则不进行划分，只创建验证和测试集
        train_df = train_df_override
        _, temp_df = train_test_split(
            df, train_size=config.TRAIN_RATIO, random_state=42
        )

    val_size = config.VALIDATION_RATIO / (config.VALIDATION_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_size, random_state=42
    )
    
    if 'negative' in train_df.columns:
        train_dataset = TripletDataset(train_df)
    else:
        train_dataset = RecommendationDataset(train_df)

    val_dataset = RecommendationDataset(val_df)
    test_dataset = RecommendationDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, df