import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import glob
import os

def transfer_excel_to_feather():
    output_dir_feather = 'data/測試資料_feather'
    os.makedirs(output_dir_feather, exist_ok=True)
    file_list = glob.glob('data/測試資料/銷售資料_*.xlsx')

    print(f"找到 {len(file_list)} 個 Excel 檔案，開始轉換為 Feather 格式...")

    for file in file_list:
        try:
            df = pd.read_excel(file)
            
            base_name = os.path.basename(file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_path = os.path.join(output_dir_feather, f"{file_name_without_ext}.feather")
            
            df.reset_index(drop=True).to_feather(output_path)
            print(f"已成功轉換: {output_path}")
            
        except Exception as e:
            print(f"處理檔案 {file} 時發生錯誤: {e}")

    print("\n所有檔案轉換完成！")

def concat_feather_files():

    data_dir = 'data/測試資料_feather/'
    pattern = os.path.join(data_dir, '銷售資料_*.feather')
    file_list = glob.glob(pattern)

    print(f"找到 {len(file_list)} 個 Feather 檔案，開始讀取...")

    dfs = [pd.read_feather(fp) for fp in file_list[0:10]]
    data = pd.concat(dfs, ignore_index=True)
    print(data.head().keys())
    print("資料合併完成！")
    return data

def data_preprocess():
    # transfer_excel_to_feather()
    data = concat_feather_files()
    data = data.rename(columns={
        'ASID': 'user_id',
        'USER_ID': 'streamer_id',
        '商品ID': 'item_id',
        '商品名稱': 'item_name',
        '單價': 'unit_price',
        '數量': 'quantity',
        '總金額': 'total_amount',
        '付款方式': 'payment_method',
        '寄送地址': 'shipping_address',
        '下單日期': 'order_date',
        'POST_ID': 'post_id',
        '留言': 'comment',
        '時間戳記': 'order_time',
    })
    data = data.dropna(subset=['order_date']).copy()
    data['datetime_str'] = data['order_date'].astype(int).astype(str) + ' ' + data['order_time'].astype(str)

    data['timestamp'] = pd.to_datetime(data['datetime_str'], format='%Y%m%d %H:%M:%S')

    print("轉換失敗的筆數：", data['timestamp'].isna().sum())
    print(data[['order_date', 'order_time', 'timestamp']].head())

    data = data.drop_duplicates()
    print("各欄位缺失值比例：")
    print(data.isna().mean())

    total_transactions = len(data)
    unique_users       = data['user_id'].nunique()
    unique_streamers   = data['streamer_id'].nunique()

    print(f"總交易筆數：{total_transactions}")
    print(f"獨立用戶數：{unique_users}")
    print(f"獨立直播主數：{unique_streamers}")

    monthly = data.set_index('timestamp').resample('ME').size()
    print("\n每月交易量：")
    print(monthly)

    daily = data.set_index('timestamp').resample('D').size()
    print("\n每日交易量前五：")
    print(daily.head())


    print(data.keys())
    print("移除前筆數：", len(data))
    data = data[data['unit_price'] != 0].reset_index(drop=True)
    print("移除後筆數：", len(data))

    data.to_csv('data/processed_sales_data.csv', index=False, encoding='utf-8-sig')
    return data

def extract_subset(
    df: pd.DataFrame,
    start_date: str = "2021-01-01",
    end_date: str = "2022-12-31",
    n_streamers: int = 10,
    random_state: int = 42,
    essential_cols: List[str] = None,
    null_tokens: List[str] = None,   # 其他想當成缺失的字串
    min_purchases_per_user: int = 10 # 可參數化
) -> Tuple[pd.DataFrame, List, Dict]:

    if essential_cols is None:
        essential_cols = ['timestamp', 'user_id', 'streamer_id', 'item_id']
    if null_tokens is None:
        null_tokens = ['blank', '', 'NA', 'N/A', 'null', 'None']

    work = df.copy()

    work['timestamp'] = pd.to_datetime(work['timestamp'], errors='coerce')
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    work = work[(work['timestamp'] >= start) & (work['timestamp'] <= end)]

    for col in set(essential_cols) & set(work.columns):
        if col != 'timestamp':
            work[col] = work[col].astype('string', copy=False).str.strip()
            work.loc[work[col].isin(null_tokens), col] = pd.NA

    before_dropna = len(work)
    work = work.dropna(subset=[c for c in essential_cols if c in work.columns])
    after_dropna = len(work)

    user_counts = work['user_id'].value_counts(dropna=False)
    keep_users = user_counts[user_counts >= min_purchases_per_user].index
    work = work[work['user_id'].isin(keep_users)]

    if work.empty:
        stats = {
            'rows_after_time_filter': int(before_dropna),
            'rows_after_dropna': int(after_dropna),
            'rows_after_user_count_filter': 0,
            'unique_streamers_after_filters': 0,
            'picked_streamers': 0
        }
        return work, [], stats

    unique_streamers = pd.Series(work['streamer_id'].dropna().unique())
    n_pick = min(n_streamers, len(unique_streamers))
    picked = unique_streamers.sample(n=n_pick, random_state=random_state).tolist()

    new_df = work[work['streamer_id'].isin(picked)].copy()
    new_df.reset_index(drop=True, inplace=True)

    stats = {
        'rows_after_time_filter': int(before_dropna),
        'rows_after_dropna': int(after_dropna),
        'rows_after_user_count_filter': int(len(work)),
        'unique_streamers_after_filters': int(len(unique_streamers)),
        'picked_streamers': int(len(picked))
    }

    return new_df, picked, stats

if __name__ == "__main__":
    data = data_preprocess()
    data_cleaned, selected_streamers, statistics = extract_subset(data)
    print(selected_streamers)
    print(statistics)
    data_cleaned.to_csv('data/cleaned_sales_data.csv', index=False, encoding='utf-8-sig')
