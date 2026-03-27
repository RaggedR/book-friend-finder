#!/usr/bin/env python3
"""
Evaluate Hyperparameters: NUM_CLUSTERS and NUM_FEATURES

Tests different values for:
1. NUM_FEATURES (latent dimensions): 10, 20, 30, 50
2. NUM_CLUSTERS (K-means clusters): 10, 15, 20, 30

Uses predictive evaluation: Do friend matches predict held-out liked books
better than random users?
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import time
import sys

print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_USERS_PER_BOOK = 5
MAX_BOOK_POPULARITY_PCT = 0.10  # Use the 10% filter
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
NEG_SAMPLES = 4
RANDOM_SEED = 42
SAMPLE_USERS = 300
HOLDOUT_FRACTION = 0.2

# Values to test
FEATURES_TO_TEST = [10, 20, 30, 50]
CLUSTERS_TO_TEST = [10, 15, 20, 30]

# Default values (for when testing the other parameter)
DEFAULT_FEATURES = 20
DEFAULT_CLUSTERS = 15

print("=" * 70)
print("HYPERPARAMETER EVALUATION: CLUSTERS AND FEATURES")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/4] Loading data...")
start = time.time()

with open(BOOKS_USERS_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

# Helper functions
def get_user_count(b):
    if 'user_count' in b:
        return b['user_count']
    return len(b.get('users', []))

def get_book_info(b):
    if 'book' in b and isinstance(b['book'], dict):
        if 'book' in b['book'] and isinstance(b['book']['book'], dict):
            return b['book']['book']
        return b['book']
    return b

STATUS_MAP = {'want_to_read': 1, 'currently_reading': 2, 'read': 3, 'did_not_finish': 5}

def get_user_entry_info(user_entry):
    if 'user' in user_entry and isinstance(user_entry['user'], dict):
        user_id = user_entry['user']['id']
        status_str = user_entry.get('status', '')
        status_id = STATUS_MAP.get(status_str, 0)
        rating = user_entry.get('rating')
        return user_id, status_id, rating
    return user_entry.get('user_id'), user_entry.get('status_id', 0), user_entry.get('rating')

total_users = len(users_data['user_books'])

# Filter books (min users + popularity filter)
filtered_books = [
    b for b in books_data['books']
    if get_user_count(b) >= MIN_USERS_PER_BOOK
]

max_readers = int(total_users * MAX_BOOK_POPULARITY_PCT)
filtered_books = [b for b in filtered_books if get_user_count(b) <= max_readers]

print(f"  Total users: {total_users}")
print(f"  Filtered books: {len(filtered_books)}")
print(f"  Loaded in {time.time() - start:.1f}s")

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[2/4] Building interaction data...")

book_id_to_idx = {get_book_info(b)['id']: idx for idx, b in enumerate(filtered_books)}
user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(users_data['user_books'])}

num_books = len(filtered_books)
num_users = len(users_data['user_books'])

user_liked_books = defaultdict(list)

for book_entry in filtered_books:
    book_info = get_book_info(book_entry)
    book_id = book_info['id']
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    for user_entry in book_entry.get('users', []):
        user_id, status_id, rating = get_user_entry_info(user_entry)
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]

        if status_id == 3:  # Read
            if rating is None or rating >= 3:
                user_liked_books[user_idx].append(book_idx)

total_liked = sum(len(l) for l in user_liked_books.values())
print(f"  Total liked interactions: {total_liked:,}")

# Train/test split
np.random.seed(RANDOM_SEED)
train_user_books = defaultdict(set)
test_user_books = defaultdict(set)

for user_idx, liked_books in user_liked_books.items():
    if len(liked_books) < 5:
        train_user_books[user_idx] = set(liked_books)
        continue

    shuffled = liked_books.copy()
    np.random.shuffle(shuffled)

    n_holdout = max(1, int(len(shuffled) * HOLDOUT_FRACTION))
    test_user_books[user_idx] = set(shuffled[:n_holdout])
    train_user_books[user_idx] = set(shuffled[n_holdout:])

# ============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_bpr(train_user_books, num_users, num_books, num_features):
    """Train BPR model with specified number of features."""
    pos_interactions = []
    for user_idx, book_set in train_user_books.items():
        for book_idx in book_set:
            pos_interactions.append((user_idx, book_idx))

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, num_features), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, num_features), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(ITERATIONS):
        np.random.seed(RANDOM_SEED + iter)
        users_list, pos_list, neg_list = [], [], []

        for u, pos_b in pos_interactions:
            user_pos = train_user_books[u]
            for _ in range(NEG_SAMPLES):
                neg_b = np.random.randint(0, num_books)
                while neg_b in user_pos:
                    neg_b = np.random.randint(0, num_books)
                users_list.append(u)
                pos_list.append(pos_b)
                neg_list.append(neg_b)

        users = tf.constant(users_list, dtype=tf.int32)
        pos_items = tf.constant(pos_list, dtype=tf.int32)
        neg_items = tf.constant(neg_list, dtype=tf.int32)

        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            pos_emb = tf.nn.embedding_lookup(X, pos_items)
            neg_emb = tf.nn.embedding_lookup(X, neg_items)

            pos_scores = tf.reduce_sum(user_emb * pos_emb, axis=1)
            neg_scores = tf.reduce_sum(user_emb * neg_emb, axis=1)

            loss = -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

    return W.numpy(), X.numpy()


def get_top_k_friends_in_cluster(W, user_idx, cluster_labels, k=10):
    """Get top-k most similar users within the same cluster."""
    W_norm = normalize(W, norm='l2', axis=1)
    W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)

    user_cluster = cluster_labels[user_idx]
    cluster_mask = cluster_labels == user_cluster

    similarities = W_norm @ W_norm[user_idx]
    similarities[user_idx] = -np.inf  # Exclude self
    similarities[~cluster_mask] = -np.inf  # Exclude other clusters

    top_k = np.argsort(-similarities)[:k]
    return top_k


def compute_predicted_score(W, X, user_idx, book_idx):
    """Compute model's predicted score for a user-book pair."""
    score = np.dot(W[user_idx], X[book_idx])
    return 1 / (1 + np.exp(-score))


def evaluate_with_clusters(W, X, cluster_labels, test_user_books, train_user_books):
    """Evaluate friend-finding quality with clustering."""
    valid_users = [u for u in test_user_books
                   if len(test_user_books[u]) > 0 and len(train_user_books.get(u, set())) >= 5]

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(SAMPLE_USERS, len(valid_users)), replace=False)

    friend_scores = []
    random_scores = []
    hit_rates_friends = []
    hit_rates_random = []

    all_users = list(train_user_books.keys())

    for user_idx in sample_users:
        holdout_books = test_user_books[user_idx]
        friends = get_top_k_friends_in_cluster(W, user_idx, cluster_labels, k=10)

        random_users = []
        while len(random_users) < 10:
            r = np.random.choice(all_users)
            if r != user_idx and r not in friends:
                random_users.append(r)

        for book_idx in holdout_books:
            friend_preds = [compute_predicted_score(W, X, f, book_idx) for f in friends]
            friend_scores.append(np.mean(friend_preds))

            random_preds = [compute_predicted_score(W, X, r, book_idx) for r in random_users]
            random_scores.append(np.mean(random_preds))

            friend_hits = sum(1 for f in friends if book_idx in train_user_books.get(f, set()))
            random_hits = sum(1 for r in random_users if book_idx in train_user_books.get(r, set()))
            hit_rates_friends.append(friend_hits / 10)
            hit_rates_random.append(random_hits / 10)

    return {
        'friend_score': np.mean(friend_scores),
        'random_score': np.mean(random_scores),
        'friend_vs_random': np.mean(friend_scores) / np.mean(random_scores) if np.mean(random_scores) > 0 else 0,
        'friend_hit_rate': np.mean(hit_rates_friends),
        'random_hit_rate': np.mean(hit_rates_random),
        'hit_rate_ratio': np.mean(hit_rates_friends) / np.mean(hit_rates_random) if np.mean(hit_rates_random) > 0 else 0,
    }


# ============================================================================
# TEST NUM_FEATURES
# ============================================================================
print("\n" + "=" * 70)
print("[3/4] Testing NUM_FEATURES (with {} clusters)".format(DEFAULT_CLUSTERS))
print("=" * 70)

features_results = {}

for num_features in FEATURES_TO_TEST:
    print(f"\n  Testing NUM_FEATURES = {num_features}...")
    start = time.time()

    # Train
    W, X = train_bpr(train_user_books, num_users, num_books, num_features)

    # Cluster
    W_norm = normalize(W, norm='l2', axis=1)
    W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)
    kmeans = KMeans(n_clusters=DEFAULT_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(W_norm)

    # Evaluate
    metrics = evaluate_with_clusters(W, X, cluster_labels, test_user_books, train_user_books)

    features_results[num_features] = metrics
    print(f"    Friend/Random: {metrics['friend_vs_random']:.2f}x, Hit Ratio: {metrics['hit_rate_ratio']:.2f}x ({time.time() - start:.1f}s)")


# ============================================================================
# TEST NUM_CLUSTERS
# ============================================================================
print("\n" + "=" * 70)
print("[4/4] Testing NUM_CLUSTERS (with {} features)".format(DEFAULT_FEATURES))
print("=" * 70)

# Train once with default features
print(f"\n  Training model with {DEFAULT_FEATURES} features...")
W, X = train_bpr(train_user_books, num_users, num_books, DEFAULT_FEATURES)
W_norm = normalize(W, norm='l2', axis=1)
W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)

clusters_results = {}

for num_clusters in CLUSTERS_TO_TEST:
    print(f"\n  Testing NUM_CLUSTERS = {num_clusters}...")
    start = time.time()

    # Cluster
    kmeans = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(W_norm)

    # Calculate avg cluster size
    cluster_sizes = [np.sum(cluster_labels == i) for i in range(num_clusters)]
    avg_size = np.mean(cluster_sizes)

    # Evaluate
    metrics = evaluate_with_clusters(W, X, cluster_labels, test_user_books, train_user_books)
    metrics['avg_cluster_size'] = avg_size

    clusters_results[num_clusters] = metrics
    print(f"    Avg cluster size: {avg_size:.0f}, Friend/Random: {metrics['friend_vs_random']:.2f}x, Hit Ratio: {metrics['hit_rate_ratio']:.2f}x ({time.time() - start:.1f}s)")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: HYPERPARAMETER EVALUATION")
print("=" * 70)

print("\n### NUM_FEATURES (latent dimensions)")
print(f"\n{'Features':<12} {'Friend/Random':<15} {'Hit Ratio':<12}")
print("-" * 40)
for nf in FEATURES_TO_TEST:
    m = features_results[nf]
    print(f"{nf:<12} {m['friend_vs_random']:<15.2f}x {m['hit_rate_ratio']:<12.2f}x")

best_features = max(features_results.keys(), key=lambda k: features_results[k]['hit_rate_ratio'])
print(f"\nBest: NUM_FEATURES = {best_features} (hit ratio: {features_results[best_features]['hit_rate_ratio']:.2f}x)")

print("\n### NUM_CLUSTERS")
print(f"\n{'Clusters':<12} {'Avg Size':<12} {'Friend/Random':<15} {'Hit Ratio':<12}")
print("-" * 55)
for nc in CLUSTERS_TO_TEST:
    m = clusters_results[nc]
    print(f"{nc:<12} {m['avg_cluster_size']:<12.0f} {m['friend_vs_random']:<15.2f}x {m['hit_rate_ratio']:<12.2f}x")

best_clusters = max(clusters_results.keys(), key=lambda k: clusters_results[k]['hit_rate_ratio'])
print(f"\nBest: NUM_CLUSTERS = {best_clusters} (hit ratio: {clusters_results[best_clusters]['hit_rate_ratio']:.2f}x)")

print("\n" + "=" * 70)
print(f"RECOMMENDATION: NUM_FEATURES={best_features}, NUM_CLUSTERS={best_clusters}")
print("=" * 70)
