#!/usr/bin/env python3
"""
Evaluate Popularity Filter for Friend-Finding

Tests whether dropping overly popular books improves friend-finding quality.

Hypothesis: Popular books (Harry Potter, 1984) don't help distinguish taste.
Dropping them should force the model to learn from more distinctive patterns.

Compares:
1. No filter (baseline): All books with 5+ users
2. With filter: Drop books read by >30% of users

Uses predictive evaluation: Do friend matches predict held-out liked books
better than random users?
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from collections import defaultdict
import os
import time

print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_FILE = os.path.join(DATA_DIR, "books_users.json")
USER_BOOKS_FILE = os.path.join(DATA_DIR, "user_books.json")

MIN_USERS_PER_BOOK = 5
NUM_FEATURES = 20
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
NEG_SAMPLES = 4
RANDOM_SEED = 42
SAMPLE_USERS = 300
HOLDOUT_FRACTION = 0.2

# Popularity thresholds to test
POPULARITY_THRESHOLDS = [1.0, 0.30, 0.20, 0.10]  # 1.0 = disabled

print("=" * 70)
print("POPULARITY FILTER EVALUATION")
print("=" * 70)
print("\nHypothesis: Dropping popular books improves friend-finding")
print("by forcing the model to learn from distinctive reading patterns.\n")

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("[1/5] Loading data...")
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

# Get total user count for percentage calculation
total_users = len(users_data['user_books'])
print(f"  Total users: {total_users}")
print(f"  Total books: {len(books_data['books'])}")
print(f"  Loaded in {time.time() - start:.1f}s")

# ============================================================================
# FILTERING FUNCTION
# ============================================================================

def filter_books(books_data, max_popularity_pct):
    """Filter books based on popularity threshold."""
    # First filter: minimum users
    filtered = [
        b for b in books_data['books']
        if get_user_count(b) >= MIN_USERS_PER_BOOK
    ]

    if max_popularity_pct < 1.0:
        max_readers = int(total_users * max_popularity_pct)
        before = len(filtered)

        # Track what we're dropping
        dropped_books = [
            (get_book_info(b)['title'], get_user_count(b))
            for b in filtered
            if get_user_count(b) > max_readers
        ]
        dropped_books.sort(key=lambda x: x[1], reverse=True)

        filtered = [b for b in filtered if get_user_count(b) <= max_readers]

        print(f"\n  Popularity filter (>{max_popularity_pct*100:.0f}% = >{max_readers} readers):")
        print(f"    Dropped {before - len(filtered)} books")
        if dropped_books[:5]:
            print(f"    Examples dropped: {', '.join(b[0] for b in dropped_books[:5])}")

    return filtered

# ============================================================================
# BUILD DATA FOR A GIVEN BOOK SET
# ============================================================================

def build_data(filtered_books, users_data):
    """Build interaction data for a filtered book set."""
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

    return num_users, num_books, user_liked_books

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def create_train_test_split(user_liked_books):
    """Split liked books into train and test sets."""
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

    return train_user_books, test_user_books

# ============================================================================
# TRAINING (BPR only - it's the best method)
# ============================================================================

def train_bpr(train_user_books, num_users, num_books):
    """Train BPR model."""
    pos_interactions = []
    for user_idx, book_set in train_user_books.items():
        for book_idx in book_set:
            pos_interactions.append((user_idx, book_idx))

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

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

        if (iter + 1) % 25 == 0:
            print(f"      Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy(), X.numpy()

# ============================================================================
# EVALUATION
# ============================================================================

def get_top_k_friends(W, user_idx, k=10):
    """Get top-k most similar users based on cosine similarity."""
    W_norm = normalize(W, norm='l2', axis=1)
    W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = W_norm @ W_norm[user_idx]
    similarities[user_idx] = -np.inf
    top_k = np.argsort(-similarities)[:k]
    return top_k

def compute_predicted_score(W, X, user_idx, book_idx):
    """Compute model's predicted score for a user-book pair."""
    score = np.dot(W[user_idx], X[book_idx])
    return 1 / (1 + np.exp(-score))

def evaluate_predictive_quality(W, X, test_user_books, train_user_books):
    """Evaluate friend-finding quality using predictive metrics."""
    valid_users = [u for u in test_user_books
                   if len(test_user_books[u]) > 0 and len(train_user_books.get(u, set())) >= 5]

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(SAMPLE_USERS, len(valid_users)), replace=False)

    friend_scores = []
    random_scores = []
    self_scores = []
    hit_rates_friends = []
    hit_rates_random = []

    all_users = list(train_user_books.keys())

    for user_idx in sample_users:
        holdout_books = test_user_books[user_idx]
        friends = get_top_k_friends(W, user_idx, k=10)

        random_users = []
        while len(random_users) < 10:
            r = np.random.choice(all_users)
            if r != user_idx and r not in friends:
                random_users.append(r)

        for book_idx in holdout_books:
            self_score = compute_predicted_score(W, X, user_idx, book_idx)
            self_scores.append(self_score)

            friend_preds = [compute_predicted_score(W, X, f, book_idx) for f in friends]
            friend_scores.append(np.mean(friend_preds))

            random_preds = [compute_predicted_score(W, X, r, book_idx) for r in random_users]
            random_scores.append(np.mean(random_preds))

            friend_hits = sum(1 for f in friends if book_idx in train_user_books.get(f, set()))
            random_hits = sum(1 for r in random_users if book_idx in train_user_books.get(r, set()))
            hit_rates_friends.append(friend_hits / 10)
            hit_rates_random.append(random_hits / 10)

    return {
        'self_score': np.mean(self_scores),
        'friend_score': np.mean(friend_scores),
        'random_score': np.mean(random_scores),
        'friend_vs_random': np.mean(friend_scores) / np.mean(random_scores) if np.mean(random_scores) > 0 else 0,
        'friend_hit_rate': np.mean(hit_rates_friends),
        'random_hit_rate': np.mean(hit_rates_random),
        'hit_rate_ratio': np.mean(hit_rates_friends) / np.mean(hit_rates_random) if np.mean(hit_rates_random) > 0 else 0,
        'n_evaluated': len(friend_scores)
    }

# ============================================================================
# RUN COMPARISON
# ============================================================================

results = {}

for threshold in POPULARITY_THRESHOLDS:
    label = "No filter" if threshold >= 1.0 else f"Drop >{threshold*100:.0f}%"
    print(f"\n{'='*70}")
    print(f"[2/5] Testing: {label}")
    print("=" * 70)

    # Filter books
    filtered_books = filter_books(books_data, threshold)
    print(f"  Books after filter: {len(filtered_books)}")

    # Build data
    num_users, num_books, user_liked_books = build_data(filtered_books, users_data)
    total_liked = sum(len(l) for l in user_liked_books.values())
    print(f"  Total liked interactions: {total_liked:,}")

    # Train/test split
    train_user_books, test_user_books = create_train_test_split(user_liked_books)

    # Train
    print(f"\n  Training BPR model...")
    start = time.time()
    W, X = train_bpr(train_user_books, num_users, num_books)
    train_time = time.time() - start
    print(f"  Trained in {train_time:.1f}s")

    # Evaluate
    print(f"\n  Evaluating...")
    metrics = evaluate_predictive_quality(W, X, test_user_books, train_user_books)

    results[label] = {
        'num_books': len(filtered_books),
        'metrics': metrics
    }

    print(f"  Friend/Random score: {metrics['friend_vs_random']:.2f}x")
    print(f"  Friend hit rate: {metrics['friend_hit_rate']*100:.2f}%")
    print(f"  Hit rate ratio: {metrics['hit_rate_ratio']:.2f}x")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[5/5] SUMMARY: POPULARITY FILTER COMPARISON")
print("=" * 70)

print(f"\n{'Configuration':<20} {'Books':<10} {'Friend/Rand':<15} {'Hit Ratio':<12}")
print("-" * 60)

for label, data in results.items():
    m = data['metrics']
    print(f"{label:<20} {data['num_books']:<10} {m['friend_vs_random']:<15.2f}x {m['hit_rate_ratio']:<12.2f}x")

print("\n" + "-" * 60)

# Determine winner
labels = list(results.keys())
if len(labels) >= 2:
    baseline = results[labels[0]]['metrics']
    filtered = results[labels[1]]['metrics']

    score_diff = filtered['friend_vs_random'] - baseline['friend_vs_random']
    hit_diff = filtered['hit_rate_ratio'] - baseline['hit_rate_ratio']

    print(f"\nDifference (filtered - baseline):")
    print(f"  Friend/Random score: {score_diff:+.2f}x")
    print(f"  Hit rate ratio: {hit_diff:+.2f}x")

    if score_diff > 0.05 or hit_diff > 0.1:
        print("\n=> Popularity filter HELPS friend-finding!")
    elif score_diff < -0.05 or hit_diff < -0.1:
        print("\n=> Popularity filter HURTS friend-finding!")
    else:
        print("\n=> No significant difference.")

print("=" * 70)
