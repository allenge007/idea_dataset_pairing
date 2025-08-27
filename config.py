# 数据相关路径
DATA_FILE = "data/dataset_idea_mini.csv"

# 模型相关配置
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # SBERT 预训练模型
EMBEDDING_DIM = 128  # 最终输出的向量维度
# 注意：BASE_MODEL的输出维度需要与第一个Dense层的输入匹配。
# 'all-MiniLM-L6-v2' 的输出是 384 维。
SBERT_OUTPUT_DIM = 384

# 训练相关参数
TRAIN_RATIO = 0.6
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
TEMPERATURE = 0.05 # 对比损失中的温度系数

# 保存路径
MODEL_SAVE_PATH = "saved_models/best_model.pt"
FAISS_INDEX_PATH = "saved_models/faiss_index.bin"
CANDIDATE_IDEAS_PATH = "saved_models/candidate_ideas.json"