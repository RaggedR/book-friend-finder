#!/usr/bin/env python3
"""
Evaluate IDF-Weighted Training for Friend-Finding

Compares 4 training methods:
1. Pointwise (original)
2. Pointwise + IDF weighting
3. BPR (original)
4. BPR + IDF weighting

IDF weighting makes the model learn more from rare books than popular ones.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from collections import defaultdict
import os

# Force unbuffered output
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
TEMP_FILE = '/tmp/books_users_filtered.json'
DATA_DIR = os.path.expanduser("~/data/hardcover/")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
NEG_SAMPLES = 4
RANDOM_SEED = 42
SAMPLE_USERS = 200

print("=" * 70)
print("IDF-WEIGHTED TRAINING EVALUATION")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

with open(TEMP_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

filtered_books = books_data['books']

# Build mappings
book_id_to_idx = {}
for idx, b in enumerate(filtered_books):
    book_info = b.get('book', b)
    if isinstance(book_info, dict):
        book_id = book_info.get('id')
    else:
        book_id = b.get('id')
    if book_id:
        book_id_to_idx[book_id] = idx

user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(users_data['user_books'])}

num_books = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_books} books")

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[2/8] Building interaction data...")

user_ratings = defaultdict(dict)
user_positives = defaultdict(set)
book_reader_count = defaultdict(int)

STATUS_MAP = {'want_to_read': 1, 'currently_reading': 2, 'read': 3, 'did_not_finish': 5}

for book_entry in filtered_books:
    book_info = book_entry.get('book', book_entry)
    book_id = book_info.get('id') if isinstance(book_info, dict) else book_entry.get('id')
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    for user_entry in book_entry.get('users', []):
        if 'user' in user_entry and isinstance(user_entry['user'], dict):
            user_id = user_entry['user']['id']
            status_str = user_entry.get('status', '')
            status_id = STATUS_MAP.get(status_str, 0)
            rating = user_entry.get('rating')
        else:
            user_id = user_entry.get('user_id')
            status_id = user_entry.get('status_id', 0)
            rating = user_entry.get('rating')

        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]

        if status_id == 3:  # Read
            book_reader_count[book_idx] += 1
            liked = 1 if (rating is None or rating >= 3) else 0
            user_ratings[user_idx][book_idx] = liked
            if liked:
                user_positives[user_idx].add(book_idx)
        elif status_id == 5:  # DNF
            user_ratings[user_idx][book_idx] = 0

# Compute IDF weights
max_readers = max(book_reader_count.values()) if book_reader_count else 1
book_idf = {}
for book_idx in range(num_books):
    readers = book_reader_count.get(book_idx, 1)
    book_idf[book_idx] = np.log(max_readers / readers) + 1

# Normalize IDF to have mean=1 (so total loss magnitude stays similar)
idf_values = list(book_idf.values())
idf_mean = np.mean(idf_values)
book_idf_normalized = {k: v / idf_mean for k, v in book_idf.items()}

total_ratings = sum(len(r) for r in user_ratings.values())
print(f"  Total ratings: {total_ratings:,}")
print(f"  IDF range: {min(book_idf.values()):.2f} - {max(book_idf.values()):.2f}")
print(f"  IDF normalized range: {min(book_idf_normalized.values()):.2f} - {max(book_idf_normalized.values()):.2f}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_pointwise(user_ratings, num_users, num_books, use_idf=False, book_idf=None):
    """Train pointwise model with optional IDF weighting."""
    name = "Pointwise + IDF" if use_idf else "Pointwise"
    print(f"  Training {name}...")

    users_list, books_list, labels_list, weights_list = [], [], [], []
    for user_idx, ratings in user_ratings.items():
        for book_idx, label in ratings.items():
            users_list.append(user_idx)
            books_list.append(book_idx)
            labels_list.append(label)
            weights_list.append(book_idf.get(book_idx, 1.0) if use_idf else 1.0)

    users = tf.constant(users_list, dtype=tf.int32)
    books = tf.constant(books_list, dtype=tf.int32)
    labels = tf.constant(labels_list, dtype=tf.float32)
    weights = tf.constant(weights_list, dtype=tf.float32)

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(ITERATIONS):
        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            book_emb = tf.nn.embedding_lookup(X, books)
            logits = tf.reduce_sum(user_emb * book_emb, axis=1)

            # IDF-weighted loss
            per_sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(per_sample_loss * weights)
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if (iter + 1) % 25 == 0:
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy()


def train_bpr(user_positives, num_users, num_books, use_idf=False, book_idf=None):
    """Train BPR model with optional IDF weighting."""
    name = "BPR + IDF" if use_idf else "BPR"
    print(f"  Training {name}...")

    pos_interactions = []
    for user_idx, book_indices in user_positives.items():
        for book_idx in book_indices:
            pos_interactions.append((user_idx, book_idx))

    print(f"    {len(pos_interactions):,} positive interactions")

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(ITERATIONS):
        np.random.seed(RANDOM_SEED + iter)
        users_list, pos_list, neg_list, weights_list = [], [], [], []

        for u, pos_b in pos_interactions:
            user_pos = user_positives[u]
            pos_idf = book_idf.get(pos_b, 1.0) if use_idf else 1.0

            for _ in range(NEG_SAMPLES):
                neg_b = np.random.randint(0, num_books)
                while neg_b in user_pos:
                    neg_b = np.random.randint(0, num_books)
                users_list.append(u)
                pos_list.append(pos_b)
                neg_list.append(neg_b)
                # Weight by positive item's IDF (rare positives matter more)
                weights_list.append(pos_idf)

        users = tf.constant(users_list, dtype=tf.int32)
        pos_items = tf.constant(pos_list, dtype=tf.int32)
        neg_items = tf.constant(neg_list, dtype=tf.int32)
        weights = tf.constant(weights_list, dtype=tf.float32)

        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            pos_emb = tf.nn.embedding_lookup(X, pos_items)
            neg_emb = tf.nn.embedding_lookup(X, neg_items)

            pos_scores = tf.reduce_sum(user_emb * pos_emb, axis=1)
            neg_scores = tf.reduce_sum(user_emb * neg_emb, axis=1)

            # IDF-weighted BPR loss
            per_sample_loss = -tf.math.log_sigmoid(pos_scores - neg_scores)
            loss = tf.reduce_mean(per_sample_loss * weights)
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if (iter + 1) % 25 == 0:
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def get_top_k_friends(W, user_idx, k=10):
    """Get top-k most similar users based on cosine similarity."""
    W_norm = normalize(W, norm='l2', axis=1)
    # Handle any NaN/Inf values
    W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = W_norm @ W_norm[user_idx]
    similarities[user_idx] = -1
    top_k = np.argsort(-similarities)[:k]
    return top_k


def compute_idf_weighted_agreement(user_a, user_b, user_ratings, book_idf):
    """Compute IDF-weighted rating agreement between two users."""
    ratings_a = user_ratings.get(user_a, {})
    ratings_b = user_ratings.get(user_b, {})
    shared_books = set(ratings_a.keys()) & set(ratings_b.keys())

    if len(shared_books) == 0:
        return 0, 0, 0

    agreement_score = 0
    total_score = 0

    for book_idx in shared_books:
        idf = book_idf.get(book_idx, 1)
        total_score += idf
        if ratings_a[book_idx] == ratings_b[book_idx]:
            agreement_score += idf

    return agreement_score, total_score, len(shared_books)


def evaluate_friend_quality(W, user_ratings, book_idf, sample_size=SAMPLE_USERS):
    """Evaluate friend-finding quality using IDF-weighted agreement."""
    valid_users = [u for u in user_ratings.keys() if len(user_ratings[u]) >= 10]
    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    all_agreement_ratios = []
    all_shared_counts = []

    for user_idx in sample_users:
        top_friends = get_top_k_friends(W, user_idx, k=10)

        for friend_idx in top_friends:
            agree, total, num_shared = compute_idf_weighted_agreement(
                user_idx, friend_idx, user_ratings, book_idf
            )
            if total > 0:
                all_agreement_ratios.append(agree / total)
                all_shared_counts.append(num_shared)

    return {
        'mean_agreement': np.mean(all_agreement_ratios) if all_agreement_ratios else 0,
        'median_agreement': np.median(all_agreement_ratios) if all_agreement_ratios else 0,
        'mean_shared_books': np.mean(all_shared_counts) if all_shared_counts else 0,
        'num_pairs': len(all_agreement_ratios)
    }


def random_baseline(user_ratings, book_idf, sample_size=SAMPLE_USERS):
    """Random friend matching baseline."""
    valid_users = [u for u in user_ratings.keys() if len(user_ratings[u]) >= 10]
    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    all_agreement_ratios = []
    for user_idx in sample_users:
        other_users = [u for u in valid_users if u != user_idx]
        random_friends = np.random.choice(other_users, min(10, len(other_users)), replace=False)
        for friend_idx in random_friends:
            agree, total, _ = compute_idf_weighted_agreement(user_idx, friend_idx, user_ratings, book_idf)
            if total > 0:
                all_agreement_ratios.append(agree / total)

    return np.mean(all_agreement_ratios) if all_agreement_ratios else 0


# ============================================================================
# RUN ALL TRAINING AND EVALUATION
# ============================================================================

print("\n[3/8] Training Pointwise (no IDF)...")
W_pointwise = train_pointwise(user_ratings, num_users, num_books, use_idf=False)

print("\n[4/8] Training Pointwise + IDF...")
W_pointwise_idf = train_pointwise(user_ratings, num_users, num_books, use_idf=True, book_idf=book_idf_normalized)

print("\n[5/8] Training BPR (no IDF)...")
W_bpr = train_bpr(user_positives, num_users, num_books, use_idf=False)

print("\n[6/8] Training BPR + IDF...")
W_bpr_idf = train_bpr(user_positives, num_users, num_books, use_idf=True, book_idf=book_idf_normalized)

print("\n[7/8] Evaluating all methods...")

results = {}

print("\n  Random baseline...")
results['Random'] = {'mean_agreement': random_baseline(user_ratings, book_idf)}

print("  Pointwise...")
results['Pointwise'] = evaluate_friend_quality(W_pointwise, user_ratings, book_idf)

print("  Pointwise + IDF...")
results['Pointwise + IDF'] = evaluate_friend_quality(W_pointwise_idf, user_ratings, book_idf)

print("  BPR...")
results['BPR'] = evaluate_friend_quality(W_bpr, user_ratings, book_idf)

print("  BPR + IDF...")
results['BPR + IDF'] = evaluate_friend_quality(W_bpr_idf, user_ratings, book_idf)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[8/8] SUMMARY: IDF-WEIGHTED TRAINING RESULTS")
print("=" * 70)

random_agreement = results['Random']['mean_agreement']

print(f"\n{'Method':<20} {'Agreement':<15} {'vs Random':<12} {'vs Base':<12}")
print("-" * 60)

base_pointwise = results['Pointwise']['mean_agreement']
base_bpr = results['BPR']['mean_agreement']

for method in ['Random', 'Pointwise', 'Pointwise + IDF', 'BPR', 'BPR + IDF']:
    agreement = results[method]['mean_agreement']
    vs_random = agreement / random_agreement if random_agreement > 0 else 0

    if method == 'Random':
        vs_base = '-'
    elif 'Pointwise' in method:
        vs_base = f"{(agreement - base_pointwise) / base_pointwise * 100:+.1f}%"
    else:
        vs_base = f"{(agreement - base_bpr) / base_bpr * 100:+.1f}%"

    print(f"{method:<20} {agreement*100:>12.1f}% {vs_random:>11.2f}x {vs_base:>12}")

print("\n" + "-" * 60)

# Find best method
best_method = max(['Pointwise', 'Pointwise + IDF', 'BPR', 'BPR + IDF'],
                  key=lambda m: results[m]['mean_agreement'])
best_agreement = results[best_method]['mean_agreement']

print(f"\nBest method: {best_method} ({best_agreement*100:.1f}% agreement)")

# Does IDF help?
pointwise_improvement = (results['Pointwise + IDF']['mean_agreement'] - results['Pointwise']['mean_agreement']) / results['Pointwise']['mean_agreement'] * 100
bpr_improvement = (results['BPR + IDF']['mean_agreement'] - results['BPR']['mean_agreement']) / results['BPR']['mean_agreement'] * 100

print(f"\nIDF weighting impact:")
print(f"  Pointwise: {pointwise_improvement:+.1f}%")
print(f"  BPR: {bpr_improvement:+.1f}%")

if pointwise_improvement > 0 or bpr_improvement > 0:
    print("\n✓ IDF weighting improves friend-finding!")
else:
    print("\n✗ IDF weighting does not help")

print("=" * 70)
