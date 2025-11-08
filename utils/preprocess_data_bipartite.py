# utils/preprocess_data.py  (Bipartite version: user-item only)

import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = "data/cleaned_sales_data.csv"

df = pd.read_csv(IN_PATH)
df_cols = {c.lower(): c for c in df.columns}

# 必要欄位：user_id, item_id, ts
user_col = df_cols.get("user_id")
item_col = df_cols.get("item_id")
ts_col   = df_cols.get("timestamp") or df_cols.get("ts")

missing = [n for n, c in [("user_id", user_col), ("item_id", item_col)] if c is None]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

# 時間處理
if ts_col is None:
    df = df.reset_index(drop=False).rename(columns={"index": "__rowidx"})
    df["ts"] = df["__rowidx"].astype(float)
    ts_col = "ts"
else:
    if not np.issubdtype(df[ts_col].dtype, np.number):
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if parsed.isna().mean() < 0.5:
            df["ts"] = parsed.view("int64") // 10**9
            ts_col = "ts"
        else:
            df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        df[ts_col] = df[ts_col].fillna(np.arange(len(df), dtype=float))

# label 處理（無則全 1）
label_col = df_cols.get("label") or df_cols.get("y") or df_cols.get("interaction") or df_cols.get("purchase")
if label_col is None:
    df["label"] = 1
    label_col = "label"
else:
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(1).astype(int)
    label_col = "label"

# 只 mapping user / item，streamer 不再需要
users = sorted(df[user_col].unique().tolist())
items = sorted(df[item_col].unique().tolist())
user2cid = {u: i for i, u in enumerate(users)}
item2cid = {it: i for i, it in enumerate(items)}

# 連續節點空間：user [1..U]，item [U+1 .. U+I]（保留 0 pad）
USER_OFFSET = 1
N_USERS = len(users)
N_ITEMS = len(items)

def uid_global(x): return USER_OFFSET + int(user2cid[x])
def iid_global(x): return USER_OFFSET + N_USERS + int(item2cid[x])

records = []
for _, r in df.iterrows():
    ug = uid_global(r[user_col])
    ig = iid_global(r[item_col])
    ts = float(r[ts_col])
    y  = int(r[label_col])
    records.append((ug, ig, ts, y))

edges = pd.DataFrame(records, columns=["src", "dst", "ts", "label"]).sort_values("ts", kind="mergesort").reset_index(drop=True)
edges["idx"] = edges.index + 1  # 1-based if you like

# Edge feats：先給 1 維常數/全 0 皆可
edge_feat = np.zeros((len(edges) + 1, 1), dtype=float)  # [0] 為 padding

# Node feats：先用 0 佔位（之後你會換成 user/item 真實 embedding）
N_NODES = USER_OFFSET + N_USERS + N_ITEMS
node_feat = np.zeros((N_NODES + 1, 172), dtype=float)

# 輸出路徑改 bipartite 名稱
OUT_EDGES = "training_data/ml_bipartite_edges.csv"
OUT_EFEAT = "training_data/ml_bipartite_features.npy"
OUT_NFEAT = "training_data/ml_bipartite_node.npy"
OUT_USERS = "training_data/id_maps_users.csv"
OUT_ITEMS = "training_data/id_maps_items.csv"

Path("training_data").mkdir(parents=True, exist_ok=True)
edges.to_csv(OUT_EDGES, index=False)
np.save(OUT_EFEAT, edge_feat)
np.save(OUT_NFEAT, node_feat)
pd.DataFrame({"user_id": users, "user_id_cid": [user2cid[u] for u in users]}).to_csv(OUT_USERS, index=False)
pd.DataFrame({"item_id": items, "item_id_cid": [item2cid[i] for i in items]}).to_csv(OUT_ITEMS, index=False)

print(edges.head())
print("\nSummary:",
      {"n_users": N_USERS, "n_items": N_ITEMS, "n_nodes_total": N_NODES, "n_edges": len(edges)})
