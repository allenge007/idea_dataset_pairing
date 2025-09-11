# download_model.py

import os
from huggingface_hub import snapshot_download

def main():
    """
    从Hugging Face Hub下载模型文件到本地。
    """
    model_id = "shibing624/text2vec-base-chinese"
    
    local_dir = os.path.join("models", model_id)
    
    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"将要保存到: {local_dir}")
    
    # 使用 snapshot_download 下载模型仓库的所有文件
    # 它会自动处理缓存和多线程下载，非常高效
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    print(f"\n模型下载完成！文件已保存于 '{local_dir}' 目录下。")
    print("现在您可以修改 config.py 文件，将 BASE_MODEL 指向这个本地路径。")

if __name__ == "__main__":
    main()