import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'

INPUT_FILE = 'data/cleaned_sales_data.csv'
OUTPUT_FILE = 'data/cleaned_sales_data_with_embedding.parquet'
COLUMN_TO_EMBED = 'clean_description'


print(f"正在載入模型: {MODEL_NAME}...")
try:
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    print("模型載入成功 (使用 GPU)。")
except Exception as e:
    print("無法載入 GPU，嘗試使用 CPU... (錯誤: {e})")
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    print("模型載入成功 (使用 CPU)。")


print(f"正在讀取輸入檔案: {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {INPUT_FILE}。請確保檔案在同一個資料夾中。")
    exit()
except Exception as e:
    print(f"讀取 CSV 時發生錯誤: {e}")
    exit()

if COLUMN_TO_EMBED not in df.columns:
    print(f"錯誤：在CSV中找不到指定的欄位 '{COLUMN_TO_EMBED}'。")
    print(f"可用的欄位有: {df.columns.tolist()}")
    exit()

print("開始處理文字資料...")
texts = df[COLUMN_TO_EMBED].fillna('').astype(str).tolist()

print(f"正在為 {len(texts)} 筆資料產生 embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
print("Embeddings 產生完畢。")
df['embedding'] = [emb.tolist() for emb in embeddings]
print(len(df['embedding'][0]))
print(f"正在將結果儲存至: {OUTPUT_FILE} (Parquet 格式)...")
try:
    df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    print("檔案儲存成功！")
except Exception as e:
    print(f"儲存 Parquet 檔案時發生錯誤: {e}")

print("處理完成。")