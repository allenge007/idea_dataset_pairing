# 数据相关路径
DATA_FILE = "data/dataset_idea_mini.csv"
TRIPLET_DATA_PATH = "data/triplet_train_data.csv" # 存放包含难负例的训练数据

# 模型相关配置
# 更换为更强的中文模型, shibing624/text2vec-base-chinese 在语义相似度任务上表现优异
BASE_MODEL = "shibing624/text2vec-base-chinese"
EMBEDDING_DIM = 128
# 'shibing624/text2vec-base-chinese' 输出维度为 768
SBERT_OUTPUT_DIM = 768

# 训练相关参数
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
BATCH_SIZE = 64  # 尽可能增大Batch Size
EPOCHS = 10
LEARNING_RATE = 2e-5 # 对于微调预训练模型，较小的学习率通常更好

# 损失函数相关参数
TEMPERATURE = 0.07      # 对比损失中的温度系数
TRIPLET_MARGIN = 0.5    # 三元组损失中的边界 (margin)

# 难负例挖掘配置
TOP_K_HARD_NEGATIVES = 5 # 为每个正样本挖掘5个难负例

# 保存路径
MODEL_SAVE_PATH_ROUND1 = "saved_models/best_model_round1.pt"
MODEL_SAVE_PATH_FINAL = "saved_models/best_model_final.pt"
FAISS_INDEX_PATH = "saved_models/faiss_index.bin"
CANDIDATE_IDEAS_PATH = "saved_models/candidate_ideas.json"