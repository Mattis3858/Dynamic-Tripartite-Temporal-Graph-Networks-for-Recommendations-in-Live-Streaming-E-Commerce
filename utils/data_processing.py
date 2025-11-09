import numpy as np
import random
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def _load_tripartite(dataset_name, base_dir="training_data"):
    graph_df = pd.read_csv(f"{base_dir}/ml_{dataset_name}_edges.csv")
    edge_features = np.load(f"{base_dir}/ml_{dataset_name}_features.npy")
    node_features = np.load(f"{base_dir}/ml_{dataset_name}_node.npy")

    # 欄位期望：src, dst, ts, label, idx
    required_cols = {"src", "dst", "ts", "label", "idx"}
    missing = required_cols - set(graph_df.columns)
    if missing:
        raise ValueError(f"Missing columns in edges csv: {missing}")

    return graph_df, node_features, edge_features

def _load_bipartite(base_dir="training_data"):
    graph_df = pd.read_csv(f"{base_dir}/ml_bipartite_edges.csv")
    edge_features = np.load(f"{base_dir}/ml_bipartite_features.npy")
    node_features = np.load(f"{base_dir}/ml_bipartite_node.npy")

    required_cols = {"src", "dst", "ts", "label", "idx"}
    missing = required_cols - set(graph_df.columns)
    if missing:
        raise ValueError(f"Missing columns in bipartite edges csv: {missing}")
    return graph_df, node_features, edge_features


def get_data_node_classification(
    dataset_name, use_validation=False, base_dir="training_data"
):
    random.seed(2020)
    graph_df, node_features, edge_features = _load_tripartite(
        dataset_name, base_dir=base_dir
    )

    # val_time, test_time = list(np.quantile(graph_df.ts.values, [0.70, 0.85]))

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    
    order = np.argsort(graph_df.ts.values, kind="mergesort")  # 稳定排序
    n = len(order)
    train_end = max(1, int(0.70 * n))
    val_end   = max(train_end + 1, int(0.85 * n))  # 確保 val 不為空；test 至少留 1 筆

    idx_train = order[:train_end]
    idx_val   = order[train_end:val_end]
    idx_test  = order[val_end:]

    mask = np.zeros(n, dtype=bool)
    train_mask = mask.copy(); train_mask[idx_train] = True
    val_mask   = mask.copy(); val_mask[idx_val]   = True
    test_mask  = mask.copy(); test_mask[idx_test] = True


    # train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    # test_mask = timestamps > test_time
    # val_mask = (
    #     np.logical_and(timestamps <= test_time, timestamps > val_time)
    #     if use_validation
    #     else test_mask
    # )

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)
    train_data = Data(
        sources[train_mask],
        destinations[train_mask],
        timestamps[train_mask],
        edge_idxs[train_mask],
        labels[train_mask],
    )
    val_data = Data(
        sources[val_mask],
        destinations[val_mask],
        timestamps[val_mask],
        edge_idxs[val_mask],
        labels[val_mask],
    )
    test_data = Data(
        sources[test_mask],
        destinations[test_mask],
        timestamps[test_mask],
        edge_idxs[test_mask],
        labels[test_mask],
    )

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(
    dataset_name,
    different_new_nodes_between_val_and_test=False,
    randomize_features=False,
    base_dir="training_data",
):
    if dataset_name == "bipartite":
        graph_df, node_features, edge_features = _load_bipartite(base_dir=base_dir)
    else:
        graph_df, node_features, edge_features = _load_tripartite(
            dataset_name, base_dir=base_dir
        )

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    sources = graph_df.src.values
    destinations = graph_df.dst.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    # -------- 1) 只用「時間排序 + 依索引比例切」 --------
    order = np.argsort(timestamps, kind="mergesort")
    n = len(order)
    train_end = max(1, int(0.70 * n))
    val_end   = max(train_end + 1, int(0.85 * n))
    if val_end >= n:
        val_end = n - 1  # 至少留 1 筆給 test

    idx_train = order[:train_end]
    idx_val   = order[train_end:val_end]
    idx_test  = order[val_end:]

    # 保底：確保 val / test 非空
    if len(idx_val) == 0 and n >= 2:
        idx_val = order[train_end:train_end+1]
        idx_test = order[train_end+1:]
    if len(idx_test) == 0 and len(idx_val) > 1:
        idx_test = np.array([idx_val[-1]])
        idx_val  = idx_val[:-1]

    mask = np.zeros(n, dtype=bool)
    train_mask = mask.copy(); train_mask[idx_train] = True
    val_mask   = mask.copy(); val_mask[idx_val]     = True
    test_mask  = mask.copy(); test_mask[idx_test]   = True

    # -------- 2) inductive new nodes：從 val+test 期的節點抽樣 --------
    random.seed(2020)
    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    post_train_idx = np.concatenate([idx_val, idx_test])  # val+test 段
    test_node_set = set(sources[post_train_idx]).union(set(destinations[post_train_idx])) if len(post_train_idx) > 0 else set()

    test_node_list = sorted(test_node_set)
    k = int(0.1 * n_total_unique_nodes)
    k = max(0, min(k, len(test_node_list)))  # 允許 0
    new_test_node_set = set(random.sample(test_node_list, k)) if k > 0 else set()

    # 只把「含新節點」的邊從訓練集移除（val/test 不動）
    new_test_source_mask = graph_df.src.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.dst.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(train_mask, observed_edges_mask)

    # 重新定義 train_data；val/test 保持索引切分
    train_data = Data(sources[train_mask], destinations[train_mask],
                      timestamps[train_mask], edge_idxs[train_mask], labels[train_mask])

    # 定義 new_node_set：未出現在訓練集的節點
    train_node_set = set(train_data.sources).union(train_data.destinations)
    new_node_set = node_set - train_node_set

    # new node 的 val/test 遮罩（基於「索引切分」的 val_mask/test_mask）
    if different_new_nodes_between_val_and_test and len(new_test_node_set) > 0:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set  = set(sorted(list(new_test_node_set))[:n_new_nodes])
        test_new_node_set = set(sorted(list(new_test_node_set))[n_new_nodes:])
        edge_contains_new_val_node_mask  = np.array([(a in val_new_node_set  or b in val_new_node_set) for a,b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array([(a in test_new_node_set or b in test_new_node_set) for a,b in zip(sources, destinations)])
        new_node_val_mask  = np.logical_and(val_mask,  edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)
    else:
        edge_contains_new_node_mask = np.array([(a in new_node_set or b in new_node_set) for a,b in zip(sources, destinations)])
        new_node_val_mask  = np.logical_and(val_mask,  edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # -------- 打包資料（val/test 完全沿用「索引切分」的遮罩） --------
    val_data = Data(sources[val_mask], destinations[val_mask],
                    timestamps[val_mask], edge_idxs[val_mask], labels[val_mask])
    test_data = Data(sources[test_mask], destinations[test_mask],
                     timestamps[test_mask], edge_idxs[test_mask], labels[test_mask])

    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask], edge_idxs[new_node_val_mask],
                             labels[new_node_val_mask])
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    print(f"The dataset has {full_data.n_interactions} interactions, involving {full_data.n_unique_nodes} different nodes")
    print(f"The training dataset has {train_data.n_interactions} interactions, involving {train_data.n_unique_nodes} different nodes")
    print(f"The validation dataset has {val_data.n_interactions} interactions, involving {val_data.n_unique_nodes} different nodes")
    print(f"The test dataset has {test_data.n_interactions} interactions, involving {test_data.n_unique_nodes} different nodes")
    print(f"The new node validation dataset has {new_node_val_data.n_interactions} interactions, involving {new_node_val_data.n_unique_nodes} different nodes")
    print(f"The new node test dataset has {new_node_test_data.n_interactions} interactions, involving {new_node_test_data.n_unique_nodes} different nodes")
    print(f"{len(new_test_node_set)} nodes were used for the inductive testing, i.e. are never seen during training")

    return (node_features, edge_features, full_data, train_data, val_data, test_data,
            new_node_val_data, new_node_test_data)



def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return (
        mean_time_shift_src,
        std_time_shift_src,
        mean_time_shift_dst,
        std_time_shift_dst,
    )
