#!/usr/bin/env python3
"""
BPR vs Pointwise Comparison for Hardcover Recommendations

Compares:
1. Current approach: Pointwise sigmoid cross-entropy (classification)
2. BPR: Bayesian Personalized Ranking (pairwise ranking)

BPR is designed for implicit feedback where we only observe positive interactions.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_USERS_PER_BOOK = 5  # Match precompute_all.py
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
RANDOM_SEED = 42
SAMPLE_USERS = 500  # For faster evaluation
NEG_SAMPLES_PER_POS = 4  # Negative samples per positive for BPR

print("=" * 80)
print("BPR vs POINTWISE COMPARISON")
print("=" * 80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[1/6] Loading data...")
start = time.time()

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

filtered_books = [
    book_entry for book_entry in books_data['books']
    if book_entry['user_count'] >= MIN_USERS_PER_BOOK
]

book_id_to_idx = {book_entry['book']['id']: idx for idx, book_entry in enumerate(filtered_books)}
user_id_to_idx = {user_entry['user']['id']: idx for idx, user_entry in enumerate(users_data['user_books'])}

num_books = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_books} books")
print(f"  Loaded in {time.time() - start:.1f}s")

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[2/6] Building interaction data...")
start = time.time()

# Build list of (user_idx, book_idx, label) tuples
# and user -> positive books mapping
interactions = []
user_positives = {u: set() for u in range(num_users)}
user_negatives = {u: set() for u in range(num_users)}

for book_entry in filtered_books:
    book_idx = book_id_to_idx[book_entry['book']['id']]
    for user_entry in book_entry['users']:
        user_id = user_entry['user_id']
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]
        status_id = user_entry['status_id']
        rating = user_entry.get('rating')

        if status_id == 3:  # Read
            if rating is not None:
                label = 1 if rating >= 3 else 0
            else:
                label = 1  # Assume positive if read without rating
            interactions.append((user_idx, book_idx, label))
            if label == 1:
                user_positives[user_idx].add(book_idx)
            else:
                user_negatives[user_idx].add(book_idx)
        elif status_id == 5:  # DNF
            interactions.append((user_idx, book_idx, 0))
            user_negatives[user_idx].add(book_idx)

interactions = np.array(interactions)
num_positive = np.sum(interactions[:, 2] == 1)
num_negative = np.sum(interactions[:, 2] == 0)

print(f"  Total interactions: {len(interactions):,}")
print(f"  Positive: {num_positive:,} ({100*num_positive/len(interactions):.1f}%)")
print(f"  Negative: {num_negative:,} ({100*num_negative/len(interactions):.1f}%)")
print(f"  Built in {time.time() - start:.1f}s")

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[3/6] Creating train/test split...")

np.random.seed(RANDOM_SEED)
indices = np.random.permutation(len(interactions))
split = int(0.8 * len(interactions))
train_idx, test_idx = indices[:split], indices[split:]

train_data = interactions[train_idx]
test_data = interactions[test_idx]

# Build train positives for each user (for negative sampling)
train_user_positives = {u: set() for u in range(num_users)}
for u, b, label in train_data:
    if label == 1:
        train_user_positives[int(u)].add(int(b))

print(f"  Training: {len(train_data):,}")
print(f"  Testing: {len(test_data):,}")

# ============================================================================
# MODEL TRAINING
# ============================================================================

def create_embeddings():
    """Create fresh embeddings for a model."""
    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1), name='W')
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1), name='X')
    return W, X


def train_pointwise(train_data, iterations=ITERATIONS):
    """Train with pointwise sigmoid cross-entropy loss."""
    W, X = create_embeddings()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Convert to tensors
    users = tf.constant(train_data[:, 0], dtype=tf.int32)
    items = tf.constant(train_data[:, 1], dtype=tf.int32)
    labels = tf.constant(train_data[:, 2], dtype=tf.float32)

    for iter in range(iterations):
        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            item_emb = tf.nn.embedding_lookup(X, items)
            logits = tf.reduce_sum(user_emb * item_emb, axis=1)

            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            )
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if iter % 10 == 0:
            print(f"    Iter {iter}: loss={total_loss:.4f}")

    return W, X


def train_bpr(train_data, iterations=ITERATIONS, neg_samples=NEG_SAMPLES_PER_POS):
    """Train with BPR pairwise ranking loss."""
    W, X = create_embeddings()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Get positive interactions only
    pos_mask = train_data[:, 2] == 1
    pos_data = train_data[pos_mask]

    print(f"    Training on {len(pos_data):,} positive interactions")
    print(f"    Sampling {neg_samples} negatives per positive")

    for iter in range(iterations):
        # Sample negatives for each positive
        users_list = []
        pos_items_list = []
        neg_items_list = []

        np.random.seed(RANDOM_SEED + iter)
        for u, pos_b, _ in pos_data:
            u, pos_b = int(u), int(pos_b)
            # Sample random negative items (not in user's positives)
            user_pos = train_user_positives[u]
            for _ in range(neg_samples):
                neg_b = np.random.randint(0, num_books)
                while neg_b in user_pos:
                    neg_b = np.random.randint(0, num_books)
                users_list.append(u)
                pos_items_list.append(pos_b)
                neg_items_list.append(neg_b)

        users = tf.constant(users_list, dtype=tf.int32)
        pos_items = tf.constant(pos_items_list, dtype=tf.int32)
        neg_items = tf.constant(neg_items_list, dtype=tf.int32)

        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            pos_emb = tf.nn.embedding_lookup(X, pos_items)
            neg_emb = tf.nn.embedding_lookup(X, neg_items)

            pos_scores = tf.reduce_sum(user_emb * pos_emb, axis=1)
            neg_scores = tf.reduce_sum(user_emb * neg_emb, axis=1)

            # BPR loss: -log(sigmoid(pos - neg))
            loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if iter % 10 == 0:
            print(f"    Iter {iter}: loss={total_loss:.4f}")

    return W, X


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_ranking(W, X, test_data, k_values=[5, 10, 20], sample_size=SAMPLE_USERS):
    """Evaluate ranking quality using Precision@K, Recall@K, NDCG@K."""
    # Group test data by user
    test_by_user = {}
    for u, b, label in test_data:
        u, b = int(u), int(b)
        if u not in test_by_user:
            test_by_user[u] = {'pos': [], 'neg': []}
        if label == 1:
            test_by_user[u]['pos'].append(b)
        else:
            test_by_user[u]['neg'].append(b)

    # Filter users with at least 1 positive test item
    valid_users = [u for u, data in test_by_user.items() if len(data['pos']) > 0]

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}

    W_np = W.numpy()
    X_np = X.numpy()

    for user_idx in sample_users:
        user_emb = W_np[user_idx]
        scores = X_np @ user_emb  # Score all items

        # Exclude training positives
        for b in train_user_positives[user_idx]:
            scores[b] = -np.inf

        test_pos = set(test_by_user[user_idx]['pos'])

        for k in k_values:
            # Get top K
            top_k = np.argpartition(-scores, k)[:k]
            top_k = top_k[np.argsort(-scores[top_k])]

            hits = len(set(top_k) & test_pos)

            # Precision@K
            precision = hits / k
            results[k]['precision'].append(precision)

            # Recall@K
            recall = hits / len(test_pos) if len(test_pos) > 0 else 0
            results[k]['recall'].append(recall)

            # NDCG@K
            dcg = sum(1.0 / np.log2(rank + 2) for rank, b in enumerate(top_k) if b in test_pos)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(test_pos))))
            ndcg = dcg / idcg if idcg > 0 else 0
            results[k]['ndcg'].append(ndcg)

    metrics = {}
    for k in k_values:
        metrics[k] = {
            'precision': np.mean(results[k]['precision']),
            'recall': np.mean(results[k]['recall']),
            'ndcg': np.mean(results[k]['ndcg'])
        }

    return metrics


def evaluate_auc(W, X, test_data, sample_size=SAMPLE_USERS):
    """Evaluate AUC - probability that a random positive is ranked above a random negative."""
    # Group by user
    test_by_user = {}
    for u, b, label in test_data:
        u, b = int(u), int(b)
        if u not in test_by_user:
            test_by_user[u] = {'pos': [], 'neg': []}
        if label == 1:
            test_by_user[u]['pos'].append(b)
        else:
            test_by_user[u]['neg'].append(b)

    # Users with both positive and negative test items
    valid_users = [u for u, data in test_by_user.items()
                   if len(data['pos']) > 0 and len(data['neg']) > 0]

    if len(valid_users) == 0:
        return 0.5  # Random baseline

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    W_np = W.numpy()
    X_np = X.numpy()

    auc_scores = []
    for user_idx in sample_users:
        user_emb = W_np[user_idx]
        scores = X_np @ user_emb

        pos_items = test_by_user[user_idx]['pos']
        neg_items = test_by_user[user_idx]['neg']

        pos_scores = scores[pos_items]
        neg_scores = scores[neg_items]

        # AUC = P(pos_score > neg_score)
        comparisons = 0
        wins = 0
        for ps in pos_scores:
            for ns in neg_scores:
                comparisons += 1
                if ps > ns:
                    wins += 1
                elif ps == ns:
                    wins += 0.5

        if comparisons > 0:
            auc_scores.append(wins / comparisons)

    return np.mean(auc_scores) if auc_scores else 0.5


# ============================================================================
# RUN COMPARISON
# ============================================================================

print("\n[4/6] Training POINTWISE model...")
start = time.time()
W_point, X_point = train_pointwise(train_data, iterations=ITERATIONS)
pointwise_time = time.time() - start
print(f"  Trained in {pointwise_time:.1f}s")

print("\n[5/6] Training BPR model...")
start = time.time()
W_bpr, X_bpr = train_bpr(train_data, iterations=ITERATIONS)
bpr_time = time.time() - start
print(f"  Trained in {bpr_time:.1f}s")

print("\n[6/6] Evaluating models...")
print("=" * 80)

print("\nPOINTWISE (Sigmoid Cross-Entropy):")
metrics_point = evaluate_ranking(W_point, X_point, test_data)
auc_point = evaluate_auc(W_point, X_point, test_data)
for k, m in metrics_point.items():
    print(f"  K={k}: Precision={m['precision']*100:.2f}%, Recall={m['recall']*100:.2f}%, NDCG={m['ndcg']:.4f}")
print(f"  AUC: {auc_point:.4f}")

print("\nBPR (Bayesian Personalized Ranking):")
metrics_bpr = evaluate_ranking(W_bpr, X_bpr, test_data)
auc_bpr = evaluate_auc(W_bpr, X_bpr, test_data)
for k, m in metrics_bpr.items():
    print(f"  K={k}: Precision={m['precision']*100:.2f}%, Recall={m['recall']*100:.2f}%, NDCG={m['ndcg']:.4f}")
print(f"  AUC: {auc_bpr:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\n{'Metric':<20} {'Pointwise':<15} {'BPR':<15} {'Diff':<15}")
print("-" * 65)

for k in [5, 10, 20]:
    p_point = metrics_point[k]['precision'] * 100
    p_bpr = metrics_bpr[k]['precision'] * 100
    diff = p_bpr - p_point
    sign = "+" if diff > 0 else ""
    print(f"Precision@{k:<9} {p_point:>13.2f}% {p_bpr:>13.2f}% {sign}{diff:>13.2f}%")

for k in [5, 10, 20]:
    n_point = metrics_point[k]['ndcg']
    n_bpr = metrics_bpr[k]['ndcg']
    diff = n_bpr - n_point
    sign = "+" if diff > 0 else ""
    print(f"NDCG@{k:<14} {n_point:>14.4f} {n_bpr:>14.4f} {sign}{diff:>14.4f}")

auc_diff = auc_bpr - auc_point
sign = "+" if auc_diff > 0 else ""
print(f"{'AUC':<20} {auc_point:>14.4f} {auc_bpr:>14.4f} {sign}{auc_diff:>14.4f}")

print(f"\nTraining time: Pointwise={pointwise_time:.1f}s, BPR={bpr_time:.1f}s")

winner = "BPR" if auc_bpr > auc_point else "Pointwise"
print(f"\nWinner: {winner}")
print("=" * 80)
