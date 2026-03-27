#!/usr/bin/env python3
"""
Predictive Evaluation for Friend-Finding

The key question: "If user B liked book X, is user A more likely to enjoy X
if A and B are matched as friends?"

This measures the PREDICTIVE value of friend matching, not just descriptive overlap.

Method:
1. Train model on all data → get predicted scores for all user-book pairs
2. For each user A, hold out some liked books
3. Get A's top-10 friend matches
4. Compare: friends' predicted scores for held-out books vs random users' scores
5. If friends score higher → friend matching predicts taste

This is the right evaluation because:
- It uses predicted preferences (including unread books), not just observed
- It measures whether friends would recommend books you'd actually like
- It accounts for the model's learned patterns, not just reading history overlap
"""

import json
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from collections import defaultdict
import os

print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_DB = os.path.join(DATA_DIR, "books_users.db")

NUM_FEATURES = 30
LAMBDA = 1.0
ITERATIONS = 50
LEARNING_RATE = 0.1
NEG_SAMPLES = 4
RANDOM_SEED = 42
SAMPLE_USERS = 300
HOLDOUT_FRACTION = 0.2  # Hold out 20% of each user's liked books

print("=" * 70)
print("PREDICTIVE EVALUATION FOR FRIEND-FINDING")
print("=" * 70)
print("\nKey question: Do friend matches predict held-out liked books")
print("better than random users?")

# ============================================================================
# LOAD DATA FROM SQLITE
# ============================================================================
print("\n[1/7] Loading data from SQLite...")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
MAX_BOOK_POPULARITY_PCT = 0.10

conn = sqlite3.connect(BOOKS_USERS_DB)
cursor = conn.cursor()

# Get book user counts
cursor.execute("""
    SELECT book_id, COUNT(*) as user_count
    FROM book_users
    GROUP BY book_id
""")
book_user_counts = {row[0]: row[1] for row in cursor.fetchall()}

# Get unique users
cursor.execute("SELECT DISTINCT user_id FROM book_users")
all_user_ids = [row[0] for row in cursor.fetchall()]
total_users = len(all_user_ids)
max_readers = int(total_users * MAX_BOOK_POPULARITY_PCT)

# Filter books: 5+ users and <= 10% popularity
filtered_book_ids = [
    bid for bid, cnt in book_user_counts.items()
    if cnt >= MIN_USERS_PER_BOOK and cnt <= max_readers
]
book_id_to_idx = {bid: idx for idx, bid in enumerate(filtered_book_ids)}

# Get all book-user relationships for filtered books
placeholders = ','.join('?' * len(filtered_book_ids))
cursor.execute(f"""
    SELECT book_id, user_id, status_id, rating
    FROM book_users
    WHERE book_id IN ({placeholders})
""", filtered_book_ids)
book_users_data = cursor.fetchall()

# Count books per user (only filtered books)
user_book_counts = defaultdict(int)
for book_id, user_id, status_id, rating in book_users_data:
    user_book_counts[user_id] += 1

# Filter users with 20+ books
filtered_user_ids = [uid for uid, cnt in user_book_counts.items() if cnt >= MIN_RATINGS_PER_USER]
user_id_to_idx = {uid: idx for idx, uid in enumerate(filtered_user_ids)}

num_books = len(filtered_book_ids)
num_users = len(filtered_user_ids)

print(f"  {num_users} users, {num_books} books")
conn.close()

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[2/7] Building interaction data...")

user_ratings = defaultdict(dict)  # user_idx -> {book_idx: rating 0-5}
user_liked_books = defaultdict(list)  # user_idx -> [book_idx, ...]

for book_id, user_id, status_id, rating in book_users_data:
    if book_id not in book_id_to_idx:
        continue
    if user_id not in user_id_to_idx:
        continue

    book_idx = book_id_to_idx[book_id]
    user_idx = user_id_to_idx[user_id]

    if status_id == 3:  # Read
        if rating is not None:
            user_ratings[user_idx][book_idx] = rating
            if rating >= 3:
                user_liked_books[user_idx].append(book_idx)
        else:
            user_ratings[user_idx][book_idx] = 4  # Assume liked if no rating
            user_liked_books[user_idx].append(book_idx)
    elif status_id == 5:  # DNF
        user_ratings[user_idx][book_idx] = 1  # Low rating for DNF

total_ratings = sum(len(r) for r in user_ratings.values())
total_liked = sum(len(l) for l in user_liked_books.values())
print(f"  Total ratings: {total_ratings:,}")
print(f"  Total liked books: {total_liked:,}")

# ============================================================================
# CREATE TRAIN/TEST SPLIT
# ============================================================================
print("\n[3/7] Creating holdout sets...")

np.random.seed(RANDOM_SEED)

train_user_books = defaultdict(set)  # For training
test_user_books = defaultdict(set)   # Held out for evaluation

for user_idx, liked_books in user_liked_books.items():
    if len(liked_books) < 5:
        # Too few books, use all for training
        train_user_books[user_idx] = set(liked_books)
        continue

    # Shuffle and split
    shuffled = liked_books.copy()
    np.random.shuffle(shuffled)

    n_holdout = max(1, int(len(shuffled) * HOLDOUT_FRACTION))
    test_user_books[user_idx] = set(shuffled[:n_holdout])
    train_user_books[user_idx] = set(shuffled[n_holdout:])

n_test_users = len([u for u in test_user_books if len(test_user_books[u]) > 0])
n_test_books = sum(len(b) for b in test_user_books.values())
print(f"  Users with holdout: {n_test_users}")
print(f"  Total held-out books: {n_test_books}")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_pointwise(train_user_books, num_users, num_books):
    """Train pointwise model on training set only."""
    print("  Training Pointwise model...")

    users_list, books_list, labels_list = [], [], []
    for user_idx, book_set in train_user_books.items():
        for book_idx in book_set:
            users_list.append(user_idx)
            books_list.append(book_idx)
            labels_list.append(1.0)  # Liked

    # Add some negative samples (unread books)
    for user_idx, book_set in train_user_books.items():
        n_neg = len(book_set)
        for _ in range(n_neg):
            neg_book = np.random.randint(0, num_books)
            while neg_book in book_set:
                neg_book = np.random.randint(0, num_books)
            users_list.append(user_idx)
            books_list.append(neg_book)
            labels_list.append(0.0)  # Treat unread as negative

    users = tf.constant(users_list, dtype=tf.int32)
    books = tf.constant(books_list, dtype=tf.int32)
    labels = tf.constant(labels_list, dtype=tf.float32)

    tf.random.set_seed(RANDOM_SEED)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1))
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1))

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    for iter in range(ITERATIONS):
        with tf.GradientTape() as tape:
            user_emb = tf.nn.embedding_lookup(W, users)
            book_emb = tf.nn.embedding_lookup(X, books)
            logits = tf.reduce_sum(user_emb * book_emb, axis=1)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            reg_loss = LAMBDA * (tf.reduce_mean(W**2) + tf.reduce_mean(X**2))
            total_loss = loss + reg_loss

        grads = tape.gradient(total_loss, [W, X])
        optimizer.apply_gradients(zip(grads, [W, X]))

        if (iter + 1) % 25 == 0:
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy(), X.numpy()


def train_bpr(train_user_books, num_users, num_books):
    """Train BPR model on training set only."""
    print("  Training BPR model...")

    pos_interactions = []
    for user_idx, book_set in train_user_books.items():
        for book_idx in book_set:
            pos_interactions.append((user_idx, book_idx))

    print(f"    {len(pos_interactions):,} positive interactions")

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
            print(f"    Iter {iter+1}: loss={float(total_loss):.4f}")

    return W.numpy(), X.numpy()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def get_top_k_friends(W, user_idx, k=10):
    """Get top-k most similar users based on cosine similarity."""
    W_norm = normalize(W, norm='l2', axis=1)
    W_norm = np.nan_to_num(W_norm, nan=0.0, posinf=0.0, neginf=0.0)
    similarities = W_norm @ W_norm[user_idx]
    similarities[user_idx] = -np.inf  # Exclude self
    top_k = np.argsort(-similarities)[:k]
    return top_k


def compute_predicted_score(W, X, user_idx, book_idx):
    """Compute model's predicted score for a user-book pair."""
    score = np.dot(W[user_idx], X[book_idx])
    return 1 / (1 + np.exp(-score))  # Sigmoid


def evaluate_predictive_quality(W, X, test_user_books, train_user_books, sample_size=SAMPLE_USERS):
    """
    Evaluate: Do friend matches predict held-out liked books better than random?

    For each user with held-out books:
    1. Get their top-10 friends
    2. For each held-out book, compute average predicted score from friends
    3. Compare to average predicted score from random users

    If friends score higher → friend matching works
    """
    # Get users with holdout books
    valid_users = [u for u in test_user_books if len(test_user_books[u]) > 0 and len(train_user_books.get(u, set())) >= 5]

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    friend_scores = []
    random_scores = []
    self_scores = []
    hit_rates_friends = []
    hit_rates_random = []

    all_users = list(train_user_books.keys())

    for user_idx in sample_users:
        holdout_books = test_user_books[user_idx]
        friends = get_top_k_friends(W, user_idx, k=10)

        # Get 10 random users (not self, not friends)
        random_users = []
        while len(random_users) < 10:
            r = np.random.choice(all_users)
            if r != user_idx and r not in friends:
                random_users.append(r)

        for book_idx in holdout_books:
            # User's own predicted score (sanity check - should be high)
            self_score = compute_predicted_score(W, X, user_idx, book_idx)
            self_scores.append(self_score)

            # Average predicted score from friends
            friend_preds = [compute_predicted_score(W, X, f, book_idx) for f in friends]
            friend_scores.append(np.mean(friend_preds))

            # Average predicted score from random users
            random_preds = [compute_predicted_score(W, X, r, book_idx) for r in random_users]
            random_scores.append(np.mean(random_preds))

            # Hit rate: did friends/random actually read and like this book?
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
# RUN EVALUATION
# ============================================================================

print("\n[4/7] Training Pointwise model...")
W_pointwise, X_pointwise = train_pointwise(train_user_books, num_users, num_books)

print("\n[5/7] Training BPR model...")
W_bpr, X_bpr = train_bpr(train_user_books, num_users, num_books)

print("\n[6/7] Evaluating predictive quality...")

print("\n  Pointwise model:")
results_pointwise = evaluate_predictive_quality(W_pointwise, X_pointwise, test_user_books, train_user_books)
print(f"    Self predicted score: {results_pointwise['self_score']:.3f}")
print(f"    Friend predicted score: {results_pointwise['friend_score']:.3f}")
print(f"    Random predicted score: {results_pointwise['random_score']:.3f}")
print(f"    Friend/Random ratio: {results_pointwise['friend_vs_random']:.2f}x")
print(f"    Friend hit rate: {results_pointwise['friend_hit_rate']*100:.2f}%")
print(f"    Random hit rate: {results_pointwise['random_hit_rate']*100:.2f}%")
print(f"    Hit rate ratio: {results_pointwise['hit_rate_ratio']:.2f}x")

print("\n  BPR model:")
results_bpr = evaluate_predictive_quality(W_bpr, X_bpr, test_user_books, train_user_books)
print(f"    Self predicted score: {results_bpr['self_score']:.3f}")
print(f"    Friend predicted score: {results_bpr['friend_score']:.3f}")
print(f"    Random predicted score: {results_bpr['random_score']:.3f}")
print(f"    Friend/Random ratio: {results_bpr['friend_vs_random']:.2f}x")
print(f"    Friend hit rate: {results_bpr['friend_hit_rate']*100:.2f}%")
print(f"    Random hit rate: {results_bpr['random_hit_rate']*100:.2f}%")
print(f"    Hit rate ratio: {results_bpr['hit_rate_ratio']:.2f}x")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("[7/7] SUMMARY: PREDICTIVE EVALUATION")
print("=" * 70)

print("""
The key question: "If I hold out a book that user A liked, do A's friend
matches predict higher scores for that book than random users?"
""")

print(f"{'Metric':<30} {'Pointwise':<15} {'BPR':<15}")
print("-" * 60)
print(f"{'Self predicted score':<30} {results_pointwise['self_score']:<15.3f} {results_bpr['self_score']:<15.3f}")
print(f"{'Friend predicted score':<30} {results_pointwise['friend_score']:<15.3f} {results_bpr['friend_score']:<15.3f}")
print(f"{'Random predicted score':<30} {results_pointwise['random_score']:<15.3f} {results_bpr['random_score']:<15.3f}")
print(f"{'Friend/Random score ratio':<30} {results_pointwise['friend_vs_random']:<15.2f}x {results_bpr['friend_vs_random']:<15.2f}x")
print("-" * 60)
print(f"{'Friend hit rate':<30} {results_pointwise['friend_hit_rate']*100:<14.2f}% {results_bpr['friend_hit_rate']*100:<14.2f}%")
print(f"{'Random hit rate':<30} {results_pointwise['random_hit_rate']*100:<14.2f}% {results_bpr['random_hit_rate']*100:<14.2f}%")
print(f"{'Hit rate ratio':<30} {results_pointwise['hit_rate_ratio']:<15.2f}x {results_bpr['hit_rate_ratio']:<15.2f}x")

print("\nInterpretation:")
print("-" * 60)

if results_pointwise['friend_vs_random'] > 1.1 or results_bpr['friend_vs_random'] > 1.1:
    print("✓ Friend matches predict held-out books BETTER than random!")
    print("  The friend-finding system provides predictive value.")
else:
    print("✗ Friend matches don't predict much better than random.")
    print("  The system may need more data or different approach.")

if results_pointwise['hit_rate_ratio'] > 1.5 or results_bpr['hit_rate_ratio'] > 1.5:
    print("✓ Friends are more likely to have READ the held-out books!")
    print("  Friends share reading patterns, not just predicted taste.")

best_model = "Pointwise" if results_pointwise['friend_vs_random'] > results_bpr['friend_vs_random'] else "BPR"
print(f"\nBest model for friend-finding: {best_model}")

print("=" * 70)
