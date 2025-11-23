import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B' 

INPUT_FILE = 'data/cleaned_sales_data_test.csv'
OUTPUT_FILE = 'data/cleaned_sales_data_with_embedding.parquet' 
COLUMN_TO_EMBED_0 = 'predicted_category'
COLUMN_TO_EMBED_1 = 'clean_description'

print(f"正在載入模型: {MODEL_NAME}...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"偵測到的裝置: {device}")

try:
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    print(f"模型載入成功 (使用 {device.upper()})。")
except Exception as e:
    print(f"模型載入發生嚴重錯誤: {e}")
    exit()

print(f"正在讀取輸入檔案: {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {INPUT_FILE}。")
    exit()
except Exception as e:
    print(f"讀取 CSV 時發生錯誤: {e}")
    exit()

required_cols = [COLUMN_TO_EMBED_0, COLUMN_TO_EMBED_1]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"錯誤：CSV 中缺少以下欄位: {missing_cols}")
    print(f"可用的欄位有: {df.columns.tolist()}")
    exit()

print("開始預處理文字資料...")
texts = (
    df[COLUMN_TO_EMBED_0].fillna('').astype(str) + 
    " - " + 
    df[COLUMN_TO_EMBED_1].fillna('').astype(str)
).tolist()

print(f"正在為 {len(texts)} 筆資料產生 embeddings...")

try:
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    print("Embeddings 產生完畢。")
    
    dim = len(embeddings[0])
    print(f"向量維度: {dim}")
    
    df['item_embedding'] = list(embeddings)

except Exception as e:
    print(f"產生 Embedding 時發生錯誤: {e}")
    exit()

print(f"正在將結果儲存至: {OUTPUT_FILE}")
try:
    df.to_parquet(OUTPUT_FILE, index=False)
    print("檔案儲存成功 (Parquet 格式)！")
    
except Exception as e:
    print(f"儲存檔案時發生錯誤: {e}")

print("處理完成。")