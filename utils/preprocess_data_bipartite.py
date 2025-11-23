import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = "data/cleaned_sales_data.csv"

df = pd.read_csv(IN_PATH)
df_cols = {c.lower(): c for c in df.columns}

user_col = df_cols.get("user_id")
item_col = df_cols.get("item_id")
ts_col = df_cols.get("timestamp") or df_cols.get("ts")

missing = [n for n, c in [("user_id", user_col), ("item_id", item_col)] if c is None]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}. Found: {list(df.columns)}")

if ts_col is None:
    df = df.reset_index(drop=False).rename(columns={"index": "__rowidx"})
    df["ts"] = df["__rowidx"].astype(float)
    ts_col = "ts"
else:
    if not np.issubdtype(df[ts_col].dtype, np.number):
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        if parsed.isna().mean() < 0.5:
            df["ts"] = parsed.astype('int64') // 10**9
            ts_col = "ts"
        else:
            df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    if df[ts_col].isna().any():
        df[ts_col] = df[ts_col].fillna(np.arange(len(df), dtype=float))

label_col = df_cols.get("label") or df_cols.get("y") or df_cols.get("interaction") or df_cols.get("purchase")
if label_col is None:
    df["label"] = 1
    label_col = "label"
else:
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(1).astype(int)
    label_col = "label"


def build_id_map(series: pd.Series, name: str):
    """Helper to create a stable mapping from original IDs to new contiguous IDs."""
    uniq = pd.Index(series.dropna().unique())
    mapping = pd.DataFrame({name: uniq, f"{name}_cid": np.arange(len(uniq), dtype=int)})
    return mapping

users_map = build_id_map(df[user_col], "user_id")
items_map = build_id_map(df[item_col], "item_id")

tmp = df[[user_col, item_col, ts_col, label_col]].copy()
tmp = tmp.merge(users_map, left_on=user_col, right_on="user_id", how="left")
tmp = tmp.merge(items_map, left_on=item_col, right_on="item_id", how="left")

tmp = tmp.dropna(subset=["user_id_cid", "item_id_cid"])

USER_OFFSET = 1
N_USERS = len(users_map)
N_ITEMS = len(items_map)

def uid_global(x): return USER_OFFSET + int(x)
def iid_global(x): return USER_OFFSET + N_USERS + int(x)

records = []
for _, r in tmp.iterrows():
    ug = uid_global(r["user_id_cid"])
    ig = iid_global(r["item_id_cid"])
    ts = float(r[ts_col])
    y  = int(r[label_col])
    records.append((ug, ig, ts, y))

edges = pd.DataFrame(records, columns=["src", "dst", "ts", "label"]).sort_values("ts", kind="mergesort").reset_index(drop=True)
edges["idx"] = edges.index + 1

edge_feat = np.zeros((len(edges) + 1, 1), dtype=float) 

N_NODES = USER_OFFSET + N_USERS + N_ITEMS
node_feat = np.zeros((N_NODES + 1, 172), dtype=float)

OUT_EDGES = "training_data/ml_bipartite_edges.csv"
OUT_EFEAT = "training_data/ml_bipartite_features.npy"
OUT_NFEAT = "training_data/ml_bipartite_node.npy"
OUT_USERS = "training_data/id_maps_users.csv"
OUT_ITEMS = "training_data/id_maps_items.csv"

Path("training_data").mkdir(parents=True, exist_ok=True)
edges.to_csv(OUT_EDGES, index=False)
np.save(OUT_EFEAT, edge_feat)
np.save(OUT_NFEAT, node_feat)

users_map.to_csv(OUT_USERS, index=False)
items_map.to_csv(OUT_ITEMS, index=False)

print(edges.head())
print("\nSummary:",
      {"n_users": N_USERS, "n_items": N_ITEMS, "n_nodes_total": N_NODES, "n_edges": len(edges)})