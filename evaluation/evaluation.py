import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def _precision_at_k(scores, labels, k):
    """單一查詢的 Precision@K：前 K 個預測中相關項的比例。"""
    order = np.argsort(-scores)
    top_k_labels = labels[order[:k]]
    return np.sum(top_k_labels) / min(k, len(labels))


def _recall_at_k(scores, labels, k):
    """單一查詢的 Recall@K：前 K 個預測中相關項數 / 總相關項數。"""
    n_relevant = np.sum(labels)
    if n_relevant == 0:
        return 0.0
    order = np.argsort(-scores)
    top_k_labels = labels[order[:k]]
    return np.sum(top_k_labels) / n_relevant


def _ndcg_at_k(scores, labels, k):
    """單一查詢的 NDCG@K。"""
    order = np.argsort(-scores)
    top_k_labels = labels[order[:k]]
    # DCG@K = sum(rel_i / log2(rank_i + 1))
    dcg = np.sum(
        top_k_labels / np.log2(np.arange(1, len(top_k_labels) + 1) + 1)
    )
    # IDCG@K：理想排序下的 DCG
    n_relevant = min(k, int(np.sum(labels)))
    if n_relevant == 0:
        return 0.0
    ideal_labels = np.zeros(k)
    ideal_labels[:n_relevant] = 1
    idcg = np.sum(
        ideal_labels / np.log2(np.arange(1, k + 1) + 1)
    )
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _ranking_metrics_at_k(pos_scores, neg_scores, k_list=(1, 5, 10)):
    """
    依 pos_scores、neg_scores 計算每個查詢的 Precision@K、Recall@K、NDCG@K，
    再對查詢取平均。每個查詢為 1 個正樣本 + 1 個負樣本。
    """
    n = len(pos_scores)
    precision_k = {k: [] for k in k_list}
    recall_k = {k: [] for k in k_list}
    ndcg_k = {k: [] for k in k_list}

    for i in range(n):
        scores = np.array([pos_scores[i], neg_scores[i]], dtype=np.float64)
        labels = np.array([1, 0])
        for k in k_list:
            k_actual = min(k, 2)  # 每查詢只有 2 個候選
            precision_k[k].append(_precision_at_k(scores, labels, k_actual))
            recall_k[k].append(_recall_at_k(scores, labels, k_actual))
            ndcg_k[k].append(_ndcg_at_k(scores, labels, k_actual))

    return (
        {k: np.mean(precision_k[k]) for k in k_list},
        {k: np.mean(recall_k[k]) for k in k_list},
        {k: np.mean(ndcg_k[k]) for k in k_list},
    )


def eval_edge_prediction(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200, k_list=(1, 5, 10)
):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    all_precision_k = {k: [] for k in k_list}
    all_recall_k = {k: [] for k in k_list}
    all_ndcg_k = {k: [] for k in k_list}
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            size = len(sources_batch)
            _, negative_samples = negative_edge_sampler.sample(size)

            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch,
                destinations_batch,
                negative_samples,
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors,
            )

            pred_score = np.concatenate(
                [(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()]
            )
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

            pos_scores = pos_prob.cpu().numpy()
            neg_scores = neg_prob.cpu().numpy()
            p_k, r_k, n_k = _ranking_metrics_at_k(pos_scores, neg_scores, k_list)
            for k in k_list:
                all_precision_k[k].append(p_k[k])
                all_recall_k[k].append(r_k[k])
                all_ndcg_k[k].append(n_k[k])

    precision_at_k = {k: np.mean(all_precision_k[k]) for k in k_list}
    recall_at_k = {k: np.mean(all_recall_k[k]) for k in k_list}
    ndcg_at_k = {k: np.mean(all_ndcg_k[k]) for k in k_list}

    return np.mean(val_ap), np.mean(val_auc), precision_at_k, recall_at_k, ndcg_at_k


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]

            source_embedding, destination_embedding, _ = (
                tgn.compute_temporal_embeddings(
                    sources_batch,
                    destinations_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    n_neighbors,
                )
            )
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx:e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
