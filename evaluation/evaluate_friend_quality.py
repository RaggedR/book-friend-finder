#!/usr/bin/env python3
"""
Evaluate Friend-Finding Quality: BPR vs Pointwise

Measures how well the user embeddings capture "taste similarity" by checking
if matched friends agree on book ratings, especially for rare books.

Metric: IDF-weighted rating agreement
- For each friend pair, find books both rated
- Check if they agree (both liked or both disliked)
- Weight agreements by IDF (rare books count more)
- Higher score = better friend matches
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
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
SAMPLE_USERS = 200  # Users to evaluate friend quality for

print("=" * 70)
print("FRIEND-FINDING QUALITY EVALUATION")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

# Check if filtered file exists, if not create it
if not os.path.exists(TEMP_FILE):
    print("  Creating filtered books file...")
    BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
    with open(BOOKS_USERS_FILE, 'r') as f:
        books_data = json.load(f)
    filtered_books = [
        b for b in books_data['books']
        if b.get('user_count', len(b.get('users', []))) >= 5
    ]
    with open(TEMP_FILE, 'w') as f:
        json.dump({'books': filtered_books}, f)
    del books_data

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
idx_to_user_id = {idx: uid for uid, idx in user_id_to_idx.items()}

num_books = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_books} books")

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[2/6] Building interaction data...")

# Track user ratings: user_idx -> {book_idx: rating}
# rating: 1 = liked, 0 = disliked
user_ratings = defaultdict(dict)
user_positives = defaultdict(set)

# Book popularity for IDF
book_reader_count = defaultdict(int)

STATUS_MAP = {'want_to_read': 1, 'currently_reading': 2, 'read': 3, 'did_not_finish': 5}

for book_entry in filtered_books:
    book_info = book_entry.get('book', book_entry)
    book_id = book_info.get('id') if isinstance(book_info, dict) else book_entry.get('id')
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    for user_entry in book_entry.get('users', []):
        # Handle different data formats
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

        # Only count reads for rating agreement
        if status_id == 3:  # Read
            book_reader_count[book_idx] += 1
            if rating is not None:
                liked = 1 if rating >= 3 else 0
            else:
                liked = 1  # Assume liked if no rating
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

total_ratings = sum(len(r) for r in user_ratings.values())
print(f"  Total ratings: {total_ratings:,}")
print(f"  IDF range: {min(book_idf.values()):.2f} - {max(book_idf.values()):.2f}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_pointwise(user_ratings, num_users, num_books, iterations=ITERATIONS):
    """Train pointwise model, return user embeddings."""
    print("  Training pointwise model...")

    # Build training data
    users_list, books_list, labels_list = [], [], []
    for user_idx, ratings in user_ratings.items():
        for book_idx, label in ratings.items():
            users_list.append(user_idx)
            books_list.append(book_idx)
            labels_list.append(label)

    users = tf.constant(users_list, dtype=tf.int32)
    books = tf.constant(books_list, dtype=tf.int32)
    labels = tf.constant(labels_list, dtype=tf.float32)

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(iterations):
        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            book_emb = tf.nn.embedding_lookup(X, books)
            logits = tf.reduce_sum(user_emb * book_emb, axis=1)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if (iter + 1) % 20 == 0:
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy()


def train_bpr(user_positives, num_users, num_books, iterations=ITERATIONS):
    """Train BPR model, return user embeddings."""
    print("  Training BPR model...")

    # Build positive interactions list
    pos_interactions = []
    for user_idx, book_indices in user_positives.items():
        for book_idx in book_indices:
            pos_interactions.append((user_idx, book_idx))

    print(f"    {len(pos_interactions):,} positive interactions")

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(iterations):
        np.random.seed(RANDOM_SEED + iter)
        users_list, pos_list, neg_list = [], [], []

        for u, pos_b in pos_interactions:
            user_pos = user_positives[u]
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

        if (iter + 1) % 20 == 0:
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy()


# ============================================================================
# FRIEND QUALITY EVALUATION
# ============================================================================

def get_top_k_friends(W, user_idx, k=10):
    """Get top-k most similar users based on cosine similarity."""
    W_norm = normalize(W, norm='l2', axis=1)
    similarities = W_norm @ W_norm[user_idx]
    similarities[user_idx] = -1  # Exclude self
    top_k = np.argsort(-similarities)[:k]
    return top_k, similarities[top_k]


def compute_idf_weighted_agreement(user_a, user_b, user_ratings, book_idf):
    """
    Compute IDF-weighted rating agreement between two users.

    Returns:
        agreement_score: Sum of IDF weights for agreed books
        total_score: Sum of IDF weights for all shared books
        num_shared: Number of books both rated
    """
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
    """
    Evaluate friend-finding quality using IDF-weighted agreement.

    For each sampled user:
    1. Get their top-10 friend matches
    2. Compute IDF-weighted agreement with each match
    3. Average across all matches
    """
    # Sample users who have enough ratings
    valid_users = [u for u in user_ratings.keys() if len(user_ratings[u]) >= 10]
    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    all_agreement_ratios = []
    all_shared_counts = []

    for user_idx in sample_users:
        top_friends, _ = get_top_k_friends(W, user_idx, k=10)

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
        'num_pairs_evaluated': len(all_agreement_ratios)
    }


# ============================================================================
# RUN EVALUATION
# ============================================================================

print("\n[3/6] Training Pointwise model...")
W_pointwise = train_pointwise(user_ratings, num_users, num_books)

print("\n[4/6] Training BPR model...")
W_bpr = train_bpr(user_positives, num_users, num_books)

print("\n[5/6] Evaluating friend-finding quality...")

print("\n  Pointwise embeddings:")
results_pointwise = evaluate_friend_quality(W_pointwise, user_ratings, book_idf)
print(f"    Mean IDF-weighted agreement: {results_pointwise['mean_agreement']*100:.1f}%")
print(f"    Median agreement: {results_pointwise['median_agreement']*100:.1f}%")
print(f"    Mean shared books per pair: {results_pointwise['mean_shared_books']:.1f}")
print(f"    Pairs evaluated: {results_pointwise['num_pairs_evaluated']}")

print("\n  BPR embeddings:")
results_bpr = evaluate_friend_quality(W_bpr, user_ratings, book_idf)
print(f"    Mean IDF-weighted agreement: {results_bpr['mean_agreement']*100:.1f}%")
print(f"    Median agreement: {results_bpr['median_agreement']*100:.1f}%")
print(f"    Mean shared books per pair: {results_bpr['mean_shared_books']:.1f}")
print(f"    Pairs evaluated: {results_bpr['num_pairs_evaluated']}")

# ============================================================================
# RANDOM BASELINE
# ============================================================================
print("\n[6/6] Computing random baseline...")

def random_friend_quality(user_ratings, book_idf, sample_size=SAMPLE_USERS):
    """Baseline: random friend matching."""
    valid_users = [u for u in user_ratings.keys() if len(user_ratings[u]) >= 10]
    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    all_agreement_ratios = []

    for user_idx in sample_users:
        # Pick 10 random other users
        other_users = [u for u in valid_users if u != user_idx]
        random_friends = np.random.choice(other_users, min(10, len(other_users)), replace=False)

        for friend_idx in random_friends:
            agree, total, _ = compute_idf_weighted_agreement(
                user_idx, friend_idx, user_ratings, book_idf
            )
            if total > 0:
                all_agreement_ratios.append(agree / total)

    return np.mean(all_agreement_ratios) if all_agreement_ratios else 0

random_agreement = random_friend_quality(user_ratings, book_idf)
print(f"  Random matching agreement: {random_agreement*100:.1f}%")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: FRIEND-FINDING QUALITY")
print("=" * 70)

print(f"\n{'Method':<20} {'IDF-Weighted Agreement':<25} {'vs Random':<15}")
print("-" * 60)
print(f"{'Random':<20} {random_agreement*100:>20.1f}% {'-':>15}")
print(f"{'Pointwise':<20} {results_pointwise['mean_agreement']*100:>20.1f}% {results_pointwise['mean_agreement']/random_agreement:>14.2f}x")
print(f"{'BPR':<20} {results_bpr['mean_agreement']*100:>20.1f}% {results_bpr['mean_agreement']/random_agreement:>14.2f}x")

improvement = (results_bpr['mean_agreement'] - results_pointwise['mean_agreement']) / results_pointwise['mean_agreement'] * 100
print(f"\nBPR vs Pointwise: {improvement:+.1f}%")

if results_bpr['mean_agreement'] > results_pointwise['mean_agreement']:
    print("\n✓ BPR produces better friend matches!")
else:
    print("\n✗ Pointwise produces better friend matches")

print("=" * 70)
