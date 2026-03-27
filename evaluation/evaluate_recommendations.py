#!/usr/bin/env python3
"""
Hardcover Recommendation Evaluation

Measures recommendation quality using multiple metrics:
- Binary accuracy (baseline)
- Precision@K, Recall@K (top-K recommendation quality)
- NDCG (ranking quality)
- Comparison to random and popularity baselines
- Probability distribution analysis
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
from collections import defaultdict

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_USERS_PER_BOOK = 2
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
RANDOM_SEED = 42

print("="*80)
print("HARDCOVER RECOMMENDATION EVALUATION")
print("="*80)

# ============================================================================
# DATA LOADING (same as before)
# ============================================================================
print("\n[1/7] Loading data...")

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

num_movies = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_movies} books")

# ============================================================================
# BUILD MATRICES
# ============================================================================
print("\n[2/7] Building rating matrices...")

Y = np.full((num_movies, num_users), 0.5)
R = np.zeros((num_movies, num_users))

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
                Y[book_idx, user_idx] = 1 if rating >= 3 else 0
                R[book_idx, user_idx] = 1
            else:
                Y[book_idx, user_idx] = 1
                R[book_idx, user_idx] = 1
        elif status_id == 5:  # DNF
            Y[book_idx, user_idx] = 0
            R[book_idx, user_idx] = 1

total_ratings = int(np.sum(R))
print(f"  Total ratings: {total_ratings:,}")

# ============================================================================
# TRAIN/TEST SPLIT (80/20)
# ============================================================================
print("\n[3/7] Creating 80/20 train/test split...")

def decompose_matrix_80_20(R, random_seed=RANDOM_SEED):
    """Vectorized 80/20 split - much faster than nested loops."""
    np.random.seed(random_seed)
    mask = np.random.rand(*R.shape) < 0.8
    A = R * mask
    B = R * (~mask)
    return A, B

def replace_with_half(Y, R):
    result = Y.copy()
    result[R == 0] = 0.5
    return result

R_train, R_test = decompose_matrix_80_20(R)
Y_train = replace_with_half(Y, R_train)
Y_test = replace_with_half(Y, R_test)

train_count = int(np.sum(R_train))
test_count = int(np.sum(R_test))

print(f"  Training: {train_count:,} ratings")
print(f"  Testing: {test_count:,} ratings")

# ============================================================================
# TRAIN MODEL
# ============================================================================
print(f"\n[4/7] Training model ({ITERATIONS} iterations)...")

def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))
    Y_binary = tf.cast(Y, tf.float32)
    Y_binary = tf.where(tf.equal(R_mask, 1), Y_binary, tf.zeros_like(Y_binary))
    logits = tf.matmul(X, W, transpose_b=True) + b
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_binary, logits=logits)
    masked_loss = loss * R_mask
    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )
    return total_cost

tf.random.set_seed(RANDOM_SEED)
W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), dtype=tf.float32), name='W')
X = tf.Variable(tf.random.normal((num_movies, NUM_FEATURES), dtype=tf.float32), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float32), name='b')

optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

for iter in range(ITERATIONS):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y_train, R_train, LAMBDA)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    if iter % 50 == 0:
        print(f"  Iteration {iter}: loss = {cost_value:.1f}")

print(f"  ✓ Training complete!")

# ============================================================================
# COMPUTE PREDICTIONS
# ============================================================================
print("\n[5/7] Computing predictions...")

logits = tf.matmul(X, W, transpose_b=True) + b
probabilities = tf.sigmoid(logits).numpy()
binary_predictions = (probabilities >= 0.5).astype(int)

print(f"  ✓ Generated predictions for all {num_movies * num_users:,} user-book pairs")

# ============================================================================
# EVALUATION METRICS
# ============================================================================
print("\n[6/7] Computing evaluation metrics...")
print("="*80)

# 1. BINARY ACCURACY
def compute_accuracy(P, Y_test, R_test):
    mask = (R_test == 1)
    P_masked = P[mask]
    Y_masked = Y_test[mask]
    matches = np.sum(P_masked == Y_masked)
    total = np.sum(mask)
    return matches / total if total > 0 else 0.0

test_accuracy = compute_accuracy(binary_predictions, Y_test, R_test)
print(f"\n1. BINARY ACCURACY")
print(f"   Test accuracy: {test_accuracy*100:.2f}%")
print(f"   (Baseline: 95% - always predict 'like')")

# 2. PRECISION@K and RECALL@K
SAMPLE_USERS = 1000  # Sample users for faster evaluation

def precision_recall_at_k(probabilities, Y_test, R_test, k_values=[5, 10, 20], sample_size=SAMPLE_USERS):
    """
    For each user, recommend top K books and measure:
    - Precision@K: % of top K that user actually liked
    - Recall@K: % of user's liked books that appear in top K
    Uses sampling for speed.
    """
    results = {k: {'precision': [], 'recall': []} for k in k_values}

    # Sample users for faster evaluation
    np.random.seed(RANDOM_SEED)
    user_indices = np.random.choice(num_users, min(sample_size, num_users), replace=False)

    for user_idx in user_indices:
        # Get user's test set likes
        user_test_mask = R_test[:, user_idx] == 1
        user_test_likes = np.where((Y_test[:, user_idx] == 1) & user_test_mask)[0]

        if len(user_test_likes) == 0:
            continue  # Skip users with no test likes

        # Get user's predictions (sorted by probability)
        user_probs = probabilities[:, user_idx]
        # Only consider books not in training set
        user_train_mask = R_train[:, user_idx] == 1
        user_probs_masked = user_probs.copy()
        user_probs_masked[user_train_mask] = -1  # Exclude training books

        max_k = max(k_values)
        top_k_indices = np.argpartition(-user_probs_masked, max_k)[:max_k]
        top_k_indices = top_k_indices[np.argsort(-user_probs_masked[top_k_indices])]

        for k in k_values:
            # Precision: % of top K that are in test likes
            hits = len(set(top_k_indices[:k]) & set(user_test_likes))
            precision = hits / k
            results[k]['precision'].append(precision)

            # Recall: % of test likes that are in top K
            recall = hits / len(user_test_likes)
            results[k]['recall'].append(recall)

    # Average across all users
    metrics = {}
    for k in k_values:
        metrics[k] = {
            'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
            'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0
        }

    return metrics

print(f"\n2. PRECISION@K and RECALL@K")
print(f"   (How good are the top K recommendations?)")
pr_metrics = precision_recall_at_k(probabilities, Y_test, R_test, k_values=[5, 10, 20])

for k, metrics in pr_metrics.items():
    print(f"\n   K={k}:")
    print(f"     Precision@{k}: {metrics['precision']*100:.2f}% (of top {k}, % user liked)")
    print(f"     Recall@{k}: {metrics['recall']*100:.2f}% (% of user's likes in top {k})")

# 3. NDCG (Normalized Discounted Cumulative Gain)
def ndcg_at_k(probabilities, Y_test, R_test, k=10, sample_size=SAMPLE_USERS):
    """
    Measures ranking quality - higher score = better ranking
    Uses sampling for speed.
    """
    ndcg_scores = []
    np.random.seed(RANDOM_SEED + 1)  # Different seed for variety
    user_indices = np.random.choice(num_users, min(sample_size, num_users), replace=False)

    for user_idx in user_indices:
        user_test_mask = R_test[:, user_idx] == 1
        if np.sum(user_test_mask) == 0:
            continue

        relevance = Y_test[:, user_idx].copy()
        user_probs = probabilities[:, user_idx].copy()
        user_train_mask = R_train[:, user_idx] == 1
        user_probs[user_train_mask] = -1

        top_k_indices = np.argpartition(-user_probs, k)[:k]
        top_k_indices = top_k_indices[np.argsort(-user_probs[top_k_indices])]

        dcg = 0
        for rank, book_idx in enumerate(top_k_indices, 1):
            if user_test_mask[book_idx]:
                dcg += relevance[book_idx] / np.log2(rank + 1)

        test_relevances = relevance[user_test_mask]
        sorted_relevances = np.sort(test_relevances)[::-1][:k]
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(sorted_relevances, 1))

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)

    return np.mean(ndcg_scores) if ndcg_scores else 0

ndcg_score = ndcg_at_k(probabilities, Y_test, R_test, k=10)
print(f"\n3. NDCG@10 (Ranking Quality)")
print(f"   NDCG@10: {ndcg_score:.4f}")
print(f"   (1.0 = perfect ranking, 0.0 = worst)")

# 4. BASELINE COMPARISONS
print(f"\n4. BASELINE COMPARISONS")

# Random baseline
np.random.seed(RANDOM_SEED)
random_probs = np.random.rand(num_movies, num_users)
random_predictions = (random_probs >= 0.5).astype(int)
random_accuracy = compute_accuracy(random_predictions, Y_test, R_test)
random_pr = precision_recall_at_k(random_probs, Y_test, R_test, k_values=[10])

# Popularity baseline (recommend most popular books)
book_popularity = np.sum(R, axis=1).reshape(-1, 1)  # Count of ratings per book
popularity_probs = np.tile(book_popularity, (1, num_users)) / num_users
popularity_predictions = (popularity_probs >= np.median(popularity_probs)).astype(int)
popularity_accuracy = compute_accuracy(popularity_predictions, Y_test, R_test)
popularity_pr = precision_recall_at_k(popularity_probs, Y_test, R_test, k_values=[10])

print(f"\n   Random baseline:")
print(f"     Accuracy: {random_accuracy*100:.2f}%")
print(f"     Precision@10: {random_pr[10]['precision']*100:.2f}%")

print(f"\n   Popularity baseline (recommend popular books):")
print(f"     Accuracy: {popularity_accuracy*100:.2f}%")
print(f"     Precision@10: {popularity_pr[10]['precision']*100:.2f}%")

print(f"\n   Our model:")
print(f"     Accuracy: {test_accuracy*100:.2f}%")
print(f"     Precision@10: {pr_metrics[10]['precision']*100:.2f}%")

improvement = (pr_metrics[10]['precision'] - popularity_pr[10]['precision']) / popularity_pr[10]['precision'] * 100
print(f"\n   → {improvement:+.1f}% improvement over popularity baseline")

# 5. PROBABILITY DISTRIBUTION ANALYSIS
print(f"\n5. PROBABILITY DISTRIBUTION")

# Analyze predictions on test set
test_mask = R_test == 1
test_probs_liked = probabilities[(Y_test == 1) & test_mask]
test_probs_disliked = probabilities[(Y_test == 0) & test_mask]

print(f"\n   Books user LIKED:")
print(f"     Mean probability: {np.mean(test_probs_liked):.1%}")
print(f"     Median probability: {np.median(test_probs_liked):.1%}")
print(f"     Std deviation: {np.std(test_probs_liked):.3f}")

print(f"\n   Books user DISLIKED:")
print(f"     Mean probability: {np.mean(test_probs_disliked):.1%}")
print(f"     Median probability: {np.median(test_probs_disliked):.1%}")
print(f"     Std deviation: {np.std(test_probs_disliked):.3f}")

separation = np.mean(test_probs_liked) - np.mean(test_probs_disliked)
print(f"\n   Probability separation: {separation:.1%}")
print(f"   (Higher = better discrimination)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("[7/7] EVALUATION SUMMARY")
print("="*80)

print(f"\n✓ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✓ Precision@10: {pr_metrics[10]['precision']*100:.2f}%")
print(f"✓ Recall@10: {pr_metrics[10]['recall']*100:.2f}%")
print(f"✓ NDCG@10: {ndcg_score:.4f}")
print(f"✓ Improvement over popularity: {improvement:+.1f}%")

print(f"\nKEY INSIGHTS:")
print(f"1. Model discriminates between likes/dislikes ({separation:.1%} probability gap)")
print(f"2. Top-10 recommendations are {pr_metrics[10]['precision']*100:.1f}% accurate")
print(f"3. Model captures {pr_metrics[10]['recall']*100:.1f}% of user's liked books in top-10")
print(f"4. Ranking quality (NDCG) is {ndcg_score:.2f} (1.0 = perfect)")

if pr_metrics[10]['precision'] > 0.9:
    print(f"\n✓ EXCELLENT: Recommendations are highly accurate!")
elif pr_metrics[10]['precision'] > 0.7:
    print(f"\n✓ GOOD: Recommendations are reliable!")
else:
    print(f"\n⚠ MODERATE: Recommendations may need improvement")

print("="*80)
