#!/usr/bin/env python3
"""
Parameter tuning for friend-finding quality.
Tests different values of NUM_CLUSTERS, NUM_FEATURES, etc.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Data files
TEMP_FILE = '/tmp/books_users_filtered.json'
DATA_DIR = os.path.expanduser("~/data/hardcover/")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

# Fixed params
LAMBDA = 1.0
LEARNING_RATE = 0.1
RANDOM_SEED = 42
SAMPLE_USERS = 300
HOLDOUT_FRACTION = 0.2

print("=" * 70)
print("PARAMETER TUNING FOR FRIEND-FINDING")
print("=" * 70)

# Load data once
print("\n[1/2] Loading data...")
with open(TEMP_FILE, 'r') as f:
    books_data = json.load(f)
with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

filtered_books = books_data['books']

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

book_id_to_idx = {}
for idx, b in enumerate(filtered_books):
    book_info = get_book_info(b)
    book_id = book_info.get('id') if isinstance(book_info, dict) else b.get('id')
    if book_id:
        book_id_to_idx[book_id] = idx

user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(users_data['user_books'])}
num_books = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_books} books")

# Build interaction data
print("\n[2/2] Building interaction data...")
user_liked_books = defaultdict(set)
for book_entry in filtered_books:
    book_info = get_book_info(book_entry)
    book_id = book_info.get('id')
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    for user_entry in book_entry.get('users', []):
        user_id, status_id, rating = get_user_entry_info(user_entry)
        if user_id not in user_id_to_idx:
            continue
        user_idx = user_id_to_idx[user_id]

        # Liked = read with rating >= 3 or no rating
        if status_id == 3:
            if rating is None or rating >= 3:
                user_liked_books[user_idx].add(book_idx)

# Create holdout sets
np.random.seed(RANDOM_SEED)
user_train_books = {}
user_holdout_books = {}

for user_idx, books in user_liked_books.items():
    books_list = list(books)
    if len(books_list) < 5:
        continue
    np.random.shuffle(books_list)
    split = max(1, int(len(books_list) * HOLDOUT_FRACTION))
    user_holdout_books[user_idx] = set(books_list[:split])
    user_train_books[user_idx] = set(books_list[split:])

print(f"  Users with holdout: {len(user_holdout_books)}")


def train_and_evaluate(num_features, num_clusters, iterations, neg_samples):
    """Train BPR model and evaluate friend-finding quality."""

    # Build training matrix
    Y = np.zeros((num_books, num_users), dtype=np.float32)
    for user_idx, books in user_train_books.items():
        for book_idx in books:
            Y[book_idx, user_idx] = 1.0

    # Initialize
    tf.random.set_seed(RANDOM_SEED)
    X = tf.Variable(tf.random.normal((num_books, num_features), stddev=0.1))
    W = tf.Variable(tf.random.normal((num_users, num_features), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Get positive interactions
    pos_pairs = []
    for user_idx, books in user_train_books.items():
        for book_idx in books:
            pos_pairs.append((book_idx, user_idx))
    pos_pairs = np.array(pos_pairs)

    # BPR training
    for iter in range(iterations):
        indices = np.random.permutation(len(pos_pairs))

        for batch_start in range(0, len(indices), 1024):
            batch_idx = indices[batch_start:batch_start + 1024]
            batch = pos_pairs[batch_idx]

            pos_items = batch[:, 0]
            users = batch[:, 1]

            # Sample negatives
            neg_items = np.random.randint(0, num_books, size=(len(batch), neg_samples))

            with tf.GradientTape() as tape:
                user_emb = tf.gather(W, users)
                pos_emb = tf.gather(X, pos_items)
                pos_scores = tf.reduce_sum(user_emb * pos_emb, axis=1)

                total_loss = 0.0
                for j in range(neg_samples):
                    neg_emb = tf.gather(X, neg_items[:, j])
                    neg_scores = tf.reduce_sum(user_emb * neg_emb, axis=1)
                    total_loss += -tf.reduce_mean(tf.math.log_sigmoid(pos_scores - neg_scores))

                total_loss /= neg_samples
                total_loss += (LAMBDA / 2) * (tf.reduce_mean(X**2) + tf.reduce_mean(W**2))

            grads = tape.gradient(total_loss, [X, W])
            optimizer.apply_gradients(zip(grads, [X, W]))

    # Get user embeddings
    W_np = W.numpy()
    W_norm = normalize(W_np, norm='l2', axis=1)

    # Cluster users
    kmeans = KMeans(n_clusters=num_clusters, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(W_norm)

    # Build cluster membership
    cluster_members = defaultdict(list)
    for user_idx, cluster in enumerate(cluster_labels):
        cluster_members[cluster].append(user_idx)

    # Evaluate on sample users
    sample_users = [u for u in user_holdout_books.keys() if len(user_holdout_books[u]) >= 3]
    np.random.seed(RANDOM_SEED + 1)
    sample_users = np.random.choice(sample_users, min(SAMPLE_USERS, len(sample_users)), replace=False)

    X_np = X.numpy()

    friend_hits = 0
    friend_total = 0
    random_hits = 0
    random_total = 0

    for user_idx in sample_users:
        holdout = user_holdout_books[user_idx]
        cluster = cluster_labels[user_idx]

        # Find friends in same cluster
        candidates = [u for u in cluster_members[cluster] if u != user_idx]
        if len(candidates) < 10:
            continue

        # Compute similarities
        similarities = W_norm @ W_norm[user_idx]
        candidate_sims = [(c, similarities[c]) for c in candidates]
        candidate_sims.sort(key=lambda x: -x[1])
        friends = [c for c, _ in candidate_sims[:10]]

        # Random users for comparison
        all_others = [u for u in range(num_users) if u != user_idx]
        random_users = np.random.choice(all_others, 10, replace=False)

        # Check hit rates
        for book_idx in holdout:
            for friend_idx in friends:
                friend_total += 1
                if book_idx in user_liked_books[friend_idx]:
                    friend_hits += 1

            for rand_idx in random_users:
                random_total += 1
                if book_idx in user_liked_books[rand_idx]:
                    random_hits += 1

    friend_hit_rate = friend_hits / friend_total if friend_total > 0 else 0
    random_hit_rate = random_hits / random_total if random_total > 0 else 0
    hit_ratio = friend_hit_rate / random_hit_rate if random_hit_rate > 0 else 0

    return {
        'friend_hit_rate': friend_hit_rate * 100,
        'random_hit_rate': random_hit_rate * 100,
        'hit_ratio': hit_ratio
    }


print("\n" + "=" * 70)
print("TESTING PARAMETERS")
print("=" * 70)

results = []

# Test different cluster counts
print("\n--- Testing NUM_CLUSTERS ---")
for clusters in [10, 15, 20, 25, 30, 40]:
    print(f"  Clusters={clusters}...", end=" ")
    r = train_and_evaluate(num_features=20, num_clusters=clusters, iterations=50, neg_samples=4)
    print(f"Hit ratio: {r['hit_ratio']:.2f}x (friend={r['friend_hit_rate']:.1f}%, random={r['random_hit_rate']:.1f}%)")
    results.append(('clusters', clusters, r))

# Test different feature dimensions
print("\n--- Testing NUM_FEATURES ---")
for features in [10, 20, 30, 50, 80]:
    print(f"  Features={features}...", end=" ")
    r = train_and_evaluate(num_features=features, num_clusters=20, iterations=50, neg_samples=4)
    print(f"Hit ratio: {r['hit_ratio']:.2f}x (friend={r['friend_hit_rate']:.1f}%, random={r['random_hit_rate']:.1f}%)")
    results.append(('features', features, r))

# Test different iteration counts
print("\n--- Testing ITERATIONS ---")
for iters in [25, 50, 100, 150]:
    print(f"  Iterations={iters}...", end=" ")
    r = train_and_evaluate(num_features=20, num_clusters=20, iterations=iters, neg_samples=4)
    print(f"Hit ratio: {r['hit_ratio']:.2f}x (friend={r['friend_hit_rate']:.1f}%, random={r['random_hit_rate']:.1f}%)")
    results.append(('iterations', iters, r))

# Test different negative samples
print("\n--- Testing NEG_SAMPLES ---")
for negs in [1, 2, 4, 8, 16]:
    print(f"  NegSamples={negs}...", end=" ")
    r = train_and_evaluate(num_features=20, num_clusters=20, iterations=50, neg_samples=negs)
    print(f"Hit ratio: {r['hit_ratio']:.2f}x (friend={r['friend_hit_rate']:.1f}%, random={r['random_hit_rate']:.1f}%)")
    results.append(('neg_samples', negs, r))

print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

# Find best for each parameter
for param_name in ['clusters', 'features', 'iterations', 'neg_samples']:
    param_results = [(v, r) for p, v, r in results if p == param_name]
    best = max(param_results, key=lambda x: x[1]['hit_ratio'])
    print(f"\nBest {param_name}: {best[0]}")
    print(f"  Hit ratio: {best[1]['hit_ratio']:.2f}x")
    print(f"  Friend hit rate: {best[1]['friend_hit_rate']:.1f}%")
    print(f"  Random hit rate: {best[1]['random_hit_rate']:.1f}%")
