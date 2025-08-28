# 工作流程

1.  **第一阶段训练**：
    运行命令，进行初步的对比学习训练。
    ```bash
    python train.py --mode round1
    ```
    这会生成 `saved_models/best_model_round1.pt`。

2.  **难负例挖掘**：
    使用第一阶段的模型来挖掘难负例。
    ```bash
    python mine_hard_negatives.py
    ```
    这会读取 `best_model_round1.pt` 并生成 `data/triplet_train_data.csv`。

3.  **第二阶段训练**：
    在包含难负例的数据上进行微调。
    ```bash
    python train.py --mode round2
    ```
    这会加载 `best_model_round1.pt` 的权重作为起点，并最终生成 `saved_models/best_model_final.pt`。

4.  **构建最终索引与服务**：
    使用**最终的模型** `best_model_final.pt` 来构建 FAISS 索引并提供服务。
    ```bash
    # (确保inference.py加载的是 MODEL_SAVE_PATH_FINAL)
    python inference.py --mode build
    python inference.py --mode search
    ```