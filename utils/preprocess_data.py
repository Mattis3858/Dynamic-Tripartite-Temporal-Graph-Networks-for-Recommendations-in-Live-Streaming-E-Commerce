# Tripartite TGN-style preprocessing for your CSV at /mnt/data/cleaned_sales_data.csv
# - Builds a single contiguous node ID space with disjoint offsets for users, streamers, and items
# - Factorizes each (user, streamer, item, ts, label) row into 3 dyadic edges: U-I, U-S, S-I
# - Produces:
#   1) /mnt/data/ml_tripartite_edges.csv        (src, dst, ts, label, etype, idx)
#   2) /mnt/data/ml_tripartite_features.npy     (edge features; first row is zeros as padding)
#   3) /mnt/data/ml_tripartite_node.npy         (node features; zeros [n_nodes x 172])
#   4) /mnt/data/id_maps_users.csv              (original->contiguous map)
#   5) /mnt/data/id_maps_streamers.csv
#   6) /mnt/data/id_maps_items.csv
# 
# Notes:
# - If timestamp column is missing, we synthesize a strictly increasing integer ts.
# - If label column is missing, defaults to 1.
# - Accepts both 'streame_id' (typo) and 'streamer_id'.

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path

IN_PATH = "data/cleaned_sales_data.csv"

# ---------- Load ----------
df = pd.read_csv(IN_PATH)

# Normalize column names (lowercase for matching)
df_cols = {c.lower(): c for c in df.columns}

# Required entity columns
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
tmp = df[[user_col, streamer_col, item_col, ts_col, label_col]].copy()
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

# Prepend zero row as padding (idx=0)
edge_feat = np.vstack([np.zeros((1, 3), dtype=float), np.array(edge_feat_rows, dtype=float)])

# ---------- Node features ----------
n_nodes = ITEM_OFFSET + n_items   # last index + 1 (since offsets started at 1)

#! change node_feat into corresponding embeddings:
#* item embedding
    #* channel_name + item_name -> gemma3 -> qwen_embedding -> item_embedding
#* user embedding
    #* user bought history -> average or sum -> user_embedding
#* streamer embedding
    #* streamer sold history -> average or sum -> streamer_embedding


node_feat = np.zeros((n_nodes + 1, 172), dtype=float)

# ---------- Save ----------
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
