import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import ast

IN_PATH = "data/cleaned_sales_data_with_embedding.parquet"

df = pd.read_parquet(IN_PATH)

df_cols = {c.lower(): c for c in df.columns}

user_col = df_cols.get("user_id")
streamer_col = df_cols.get("streamer_id")
item_col = df_cols.get("item_id")

missing = [n for n, c in [("user_id", user_col), ("streamer_id", streamer_col), ("item_id", item_col)] if c is None]
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}. Found columns: {list(df.columns)}")

# Timestamp handling
ts_col = df_cols.get("timestamp")
if ts_col is None:
    # Create strictly increasing timestamps if none provided
    df = df.reset_index(drop=False).rename(columns={"index": "__rowidx"})
    df["ts"] = df["__rowidx"].astype(float)
    ts_col = "ts"
else:
    # Try to parse as numeric; if datetime strings, convert to POSIX seconds
    if not np.issubdtype(df[ts_col].dtype, np.number):
        try:
            parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            # If many NaT, assume it was numeric-like text
            if parsed.isna().mean() < 0.5:
                df["ts"] = parsed.view("int64") // 10**9  # to seconds
                ts_col = "ts"
            else:
                df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
        except Exception:
            df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    # Fill missing with increasing integers
    if df[ts_col].isna().any():
        df[ts_col] = df[ts_col].fillna(np.arange(len(df), dtype=float))

# Label handling
label_col = df_cols.get("label") or df_cols.get("y") or df_cols.get("interaction") or df_cols.get("purchase")
if label_col is None:
    df["label"] = 1
    label_col = "label"
else:
    # Coerce to int if possible
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(1).astype(int)
    label_col = "label"

emb_col = df_cols.get("item_embedding")
if emb_col is None:
    raise ValueError("Missing 'item_embedding' column in parquet.")
if isinstance(df[emb_col].iloc[0], str):
    df[emb_col] = df[emb_col].apply(ast.literal_eval)

first_emb = df[emb_col].iloc[0]
if not isinstance(first_emb, (list, np.ndarray)):
    raise ValueError(f"'item_embedding' must be list/ndarray per row, got {type(first_emb)}")
embed_dim = len(first_emb)

# ---------- Build ID maps per type ----------
def build_id_map(series: pd.Series, name: str):
    uniq = pd.Index(series.unique())
    # Stable order
    mapping = pd.DataFrame({name: uniq, f"{name}_cid": np.arange(len(uniq), dtype=int)})
    return mapping

users_map = build_id_map(df[user_col], "user_id")
streamers_map = build_id_map(df[streamer_col], "streamer_id")
items_map = build_id_map(df[item_col], "item_id")

# Merge to get contiguous ids
tmp = df[[user_col, streamer_col, item_col, ts_col, label_col, emb_col]].copy()
tmp = tmp.merge(users_map, left_on=user_col, right_on="user_id", how="left")
tmp = tmp.merge(streamers_map, left_on=streamer_col, right_on="streamer_id", how="left", suffixes=("",""))
tmp = tmp.merge(items_map, left_on=item_col, right_on="item_id", how="left", suffixes=("",""))

# Offsets to place all nodes in a single contiguous space
n_users = len(users_map)
n_streamers = len(streamers_map)
n_items = len(items_map)

USER_OFFSET = 1  # reserve 0 for padding if needed
STREAMER_OFFSET = USER_OFFSET + n_users
ITEM_OFFSET = STREAMER_OFFSET + n_streamers

def uid_global(x): return USER_OFFSET + int(x)
def sid_global(x): return STREAMER_OFFSET + int(x)
def iid_global(x): return ITEM_OFFSET + int(x)

# ---------- Factorize into 3 dyadic edges per triplet ----------
# Edge types: 0=U-I, 1=U-S, 2=S-I
records = []

for _, r in tmp.iterrows():
    ug = uid_global(r["user_id_cid"])
    sg = sid_global(r["streamer_id_cid"])
    ig = iid_global(r["item_id_cid"])
    ts = float(r[ts_col])
    y  = int(r[label_col])

    # U-I
    records.append((ug, ig, ts, y, 0))
    # U-S
    records.append((ug, sg, ts, y, 1))
    # S-I
    records.append((sg, ig, ts, y, 2))

edges = pd.DataFrame(records, columns=["src", "dst", "ts", "label", "etype"])
edges = edges.sort_values(["ts", "etype"]).reset_index(drop=True)
edges["idx"] = edges.index + 1  # 1-based like original script

# ---------- Edge features ----------
# Simple 1-hot edge-type features of length 3
etype_eye = np.eye(3, dtype=float)
edge_feat_rows = [etype_eye[e] for e in edges["etype"].to_numpy()]
edge_feat = np.vstack([np.zeros((1, 3), dtype=float), np.array(edge_feat_rows, dtype=float)])

# ---------- Node features ----------
item_emb_df = (
    tmp.groupby("item_id_cid")[emb_col]
    .apply(lambda x: np.mean(np.stack(x.values), axis=0))
    .reset_index()
    .rename(columns={emb_col: "item_embedding_vec"})
)
n_nodes = ITEM_OFFSET + n_items   # last index + 1 (since offsets started at 1)
node_feat = np.zeros((n_nodes + 1, embed_dim), dtype=float)

for _, row in item_emb_df.iterrows():
    cid = int(row["item_id_cid"])
    idx = ITEM_OFFSET + cid
    vec = row["item_embedding_vec"]
    node_feat[idx, :] = np.asarray(vec, dtype=float)
#! change node_feat into corresponding embeddings:
#* item embedding
    #* channel_name + item_name -> gemma3 -> qwen_embedding -> item_embedding
#* user embedding
    #* user bought history -> average or sum -> user_embedding
#* streamer embedding
    #* streamer sold history -> average or sum -> streamer_embedding

# ---------- Save ----------
Path("training_data").mkdir(parents=True, exist_ok=True)
OUT_EDGES = "training_data/ml_tripartite_edges.csv"
OUT_EFEAT = "training_data/ml_tripartite_features.npy"
OUT_NFEAT = "training_data/ml_tripartite_node.npy"
OUT_USERS = "training_data/id_maps_users.csv"
OUT_STREAMERS = "training_data/id_maps_streamers.csv"
OUT_ITEMS = "training_data/id_maps_items.csv"

edges.to_csv(OUT_EDGES, index=False)
np.save(OUT_EFEAT, edge_feat)
np.save(OUT_NFEAT, node_feat)
users_map.to_csv(OUT_USERS, index=False)
streamers_map.to_csv(OUT_STREAMERS, index=False)
items_map.to_csv(OUT_ITEMS, index=False)

# Show a small preview to you
preview = edges.head(12)
print(preview)

# Also show unique counts for sanity
summary = pd.DataFrame({
    "n_users":[n_users],
    "n_streamers":[n_streamers],
    "n_items":[n_items],
    "n_nodes_total":[n_nodes],
    "n_edges":[len(edges)]
})

print("\nSummary:", summary)
