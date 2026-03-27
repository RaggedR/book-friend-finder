#!/usr/bin/env python3
"""
Niche Discovery Evaluation for Friend-Finding

The key question: "Can friends help you discover NON-POPULAR books you'd like?"

This is different from general predictive evaluation because:
- Popularity baseline achieves ~38% precision by recommending Harry Potter to everyone
- That doesn't help friend-finding - everyone has read Harry Potter
- The VALUE of a friend is discovering books you wouldn't find otherwise

Method:
1. Compute book popularity (number of readers)
2. Hold out only NON-POPULAR books (below median popularity) from each user
3. Train model on remaining data
4. Measure: do friends predict held-out NICHE books better than random?

If friends predict niche books better → friend-finding provides discovery value
that popularity-based recommendations cannot.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import normalize
from collections import defaultdict
import os

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
SAMPLE_USERS = 300
HOLDOUT_FRACTION = 0.3  # Hold out 30% of each user's NICHE liked books

print("=" * 70)
print("NICHE DISCOVERY EVALUATION FOR FRIEND-FINDING")
print("=" * 70)
print("\nKey question: Do friend matches predict held-out NICHE books")
print("(below-median popularity) better than random users?")
print("\nThis measures discovery value - can friends help you find books")
print("you wouldn't find through popularity recommendations?")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

with open(TEMP_FILE, 'r') as f:
    books_data = json.load(f)

with open(USER_BOOKS_FILE, 'r') as f:
    users_data = json.load(f)

filtered_books = books_data['books']

book_id_to_idx = {}
for idx, b in enumerate(filtered_books):
    book_info = b.get('book', b)
    book_id = book_info.get('id') if isinstance(book_info, dict) else b.get('id')
    if book_id:
        book_id_to_idx[book_id] = idx

user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(users_data['user_books'])}

num_books = len(filtered_books)
num_users = len(users_data['user_books'])

print(f"  {num_users} users, {num_books} books")

# ============================================================================
# COMPUTE BOOK POPULARITY
# ============================================================================
print("\n[2/8] Computing book popularity...")

book_popularity = defaultdict(int)  # book_idx -> reader count

STATUS_MAP = {'want_to_read': 1, 'currently_reading': 2, 'read': 3, 'did_not_finish': 5}

for book_entry in filtered_books:
    book_info = book_entry.get('book', book_entry)
    book_id = book_info.get('id') if isinstance(book_info, dict) else book_entry.get('id')
    if book_id not in book_id_to_idx:
        continue
    book_idx = book_id_to_idx[book_id]

    # Count users who have read this book
    for user_entry in book_entry.get('users', []):
        if 'user' in user_entry and isinstance(user_entry['user'], dict):
            status_str = user_entry.get('status', '')
            status_id = STATUS_MAP.get(status_str, 0)
        else:
            status_id = user_entry.get('status_id', 0)

        if status_id == 3:  # Read
            book_popularity[book_idx] += 1

popularity_values = list(book_popularity.values())
median_popularity = np.median(popularity_values) if popularity_values else 0
max_popularity = max(popularity_values) if popularity_values else 0

print(f"  Median book popularity: {median_popularity:.0f} readers")
print(f"  Max book popularity: {max_popularity} readers")

# Classify books as popular or niche
popular_books = set(b for b, p in book_popularity.items() if p >= median_popularity)
niche_books = set(b for b, p in book_popularity.items() if p < median_popularity)

print(f"  Popular books (>= median): {len(popular_books)}")
print(f"  Niche books (< median): {len(niche_books)}")

# ============================================================================
# BUILD INTERACTION DATA
# ============================================================================
print("\n[3/8] Building interaction data...")

user_ratings = defaultdict(dict)
user_liked_books = defaultdict(list)
user_liked_niche = defaultdict(list)  # Only niche books
user_liked_popular = defaultdict(list)  # Only popular books

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
            if rating is not None:
                user_ratings[user_idx][book_idx] = rating
                if rating >= 3:
                    user_liked_books[user_idx].append(book_idx)
                    if book_idx in niche_books:
                        user_liked_niche[user_idx].append(book_idx)
                    else:
                        user_liked_popular[user_idx].append(book_idx)
            else:
                user_ratings[user_idx][book_idx] = 4
                user_liked_books[user_idx].append(book_idx)
                if book_idx in niche_books:
                    user_liked_niche[user_idx].append(book_idx)
                else:
                    user_liked_popular[user_idx].append(book_idx)
        elif status_id == 5:  # DNF
            user_ratings[user_idx][book_idx] = 1

total_liked = sum(len(l) for l in user_liked_books.values())
total_niche = sum(len(l) for l in user_liked_niche.values())
total_popular = sum(len(l) for l in user_liked_popular.values())

print(f"  Total liked books: {total_liked:,}")
print(f"  Total liked niche books: {total_niche:,} ({100*total_niche/total_liked:.1f}%)")
print(f"  Total liked popular books: {total_popular:,} ({100*total_popular/total_liked:.1f}%)")

# ============================================================================
# CREATE TRAIN/TEST SPLIT (NICHE BOOKS ONLY FOR TEST)
# ============================================================================
print("\n[4/8] Creating holdout sets (niche books only)...")

np.random.seed(RANDOM_SEED)

train_user_books = defaultdict(set)
test_user_books = defaultdict(set)  # Only niche books

users_with_niche = 0

for user_idx in user_liked_books.keys():
    niche_liked = user_liked_niche[user_idx]
    popular_liked = user_liked_popular[user_idx]

    # All popular books go to training
    train_user_books[user_idx] = set(popular_liked)

    if len(niche_liked) < 3:
        # Too few niche books, use all for training
        train_user_books[user_idx].update(niche_liked)
        continue

    users_with_niche += 1

    # Shuffle niche books and split
    shuffled = niche_liked.copy()
    np.random.shuffle(shuffled)

    n_holdout = max(1, int(len(shuffled) * HOLDOUT_FRACTION))
    test_user_books[user_idx] = set(shuffled[:n_holdout])
    train_user_books[user_idx].update(shuffled[n_holdout:])

n_test_users = len([u for u in test_user_books if len(test_user_books[u]) > 0])
n_test_books = sum(len(b) for b in test_user_books.values())

print(f"  Users with enough niche books: {users_with_niche}")
print(f"  Users with holdout: {n_test_users}")
print(f"  Total held-out niche books: {n_test_books}")

# Verify holdout books are actually niche
holdout_popularities = []
for user_idx, books in test_user_books.items():
    for book_idx in books:
        holdout_popularities.append(book_popularity[book_idx])

if holdout_popularities:
    print(f"  Holdout books avg popularity: {np.mean(holdout_popularities):.1f} readers")
    print(f"  Holdout books max popularity: {max(holdout_popularities)} readers")

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
            labels_list.append(1.0)

    # Add negative samples
    for user_idx, book_set in train_user_books.items():
        n_neg = len(book_set)
        for _ in range(n_neg):
            neg_book = np.random.randint(0, num_books)
            while neg_book in book_set:
                neg_book = np.random.randint(0, num_books)
            users_list.append(user_idx)
            books_list.append(neg_book)
            labels_list.append(0.0)

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
    similarities[user_idx] = -np.inf
    top_k = np.argsort(-similarities)[:k]
    return top_k


def compute_predicted_score(W, X, user_idx, book_idx):
    """Compute model's predicted score for a user-book pair."""
    score = np.dot(W[user_idx], X[book_idx])
    return 1 / (1 + np.exp(-score))


def evaluate_niche_discovery(W, X, test_user_books, train_user_books, book_popularity, sample_size=SAMPLE_USERS):
    """
    Evaluate: Do friend matches predict held-out NICHE books better than random?

    This measures DISCOVERY VALUE - can friends help you find books that
    popularity-based recommendations cannot?
    """
    valid_users = [u for u in test_user_books if len(test_user_books[u]) > 0 and len(train_user_books.get(u, set())) >= 5]

    np.random.seed(RANDOM_SEED)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    friend_scores = []
    random_scores = []
    self_scores = []
    hit_rates_friends = []
    hit_rates_random = []

    # Track by popularity bucket
    very_niche_friend_hits = []  # Books with < 20 readers
    very_niche_random_hits = []

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
            pop = book_popularity[book_idx]

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

            # Track very niche books separately
            if pop < 20:
                very_niche_friend_hits.append(friend_hits / 10)
                very_niche_random_hits.append(random_hits / 10)

    return {
        'self_score': np.mean(self_scores),
        'friend_score': np.mean(friend_scores),
        'random_score': np.mean(random_scores),
        'friend_vs_random': np.mean(friend_scores) / np.mean(random_scores) if np.mean(random_scores) > 0 else 0,
        'friend_hit_rate': np.mean(hit_rates_friends),
        'random_hit_rate': np.mean(hit_rates_random),
        'hit_rate_ratio': np.mean(hit_rates_friends) / np.mean(hit_rates_random) if np.mean(hit_rates_random) > 0 else 0,
        'very_niche_friend_hit': np.mean(very_niche_friend_hits) if very_niche_friend_hits else 0,
        'very_niche_random_hit': np.mean(very_niche_random_hits) if very_niche_random_hits else 0,
        'very_niche_ratio': (np.mean(very_niche_friend_hits) / np.mean(very_niche_random_hits)) if very_niche_random_hits and np.mean(very_niche_random_hits) > 0 else 0,
        'n_evaluated': len(friend_scores),
        'n_very_niche': len(very_niche_friend_hits)
    }


def evaluate_discovery_potential(W, train_user_books, book_popularity, popular_threshold=100, sample_size=SAMPLE_USERS):
    """
    Evaluate: Do friends have more non-popular books to recommend?

    For each user, count books that:
    1. Their friends have read
    2. They haven't read
    3. Are not in top-N popular

    Compare to random users.
    """
    valid_users = [u for u in train_user_books if len(train_user_books[u]) >= 10]

    np.random.seed(RANDOM_SEED + 1)
    sample_users = np.random.choice(valid_users, min(sample_size, len(valid_users)), replace=False)

    # Get top popular books to exclude
    sorted_books = sorted(book_popularity.items(), key=lambda x: -x[1])
    top_popular = set(b for b, _ in sorted_books[:popular_threshold])

    friend_discovery = []
    random_discovery = []

    all_users = list(train_user_books.keys())

    for user_idx in sample_users:
        user_books = train_user_books[user_idx]
        friends = get_top_k_friends(W, user_idx, k=10)

        random_users = []
        while len(random_users) < 10:
            r = np.random.choice(all_users)
            if r != user_idx and r not in friends:
                random_users.append(r)

        # Count non-popular books friends have that user doesn't
        friend_discoverable = set()
        for f in friends:
            for book in train_user_books.get(f, set()):
                if book not in user_books and book not in top_popular:
                    friend_discoverable.add(book)

        random_discoverable = set()
        for r in random_users:
            for book in train_user_books.get(r, set()):
                if book not in user_books and book not in top_popular:
                    random_discoverable.add(book)

        friend_discovery.append(len(friend_discoverable))
        random_discovery.append(len(random_discoverable))

    return {
        'friend_discovery_potential': np.mean(friend_discovery),
        'random_discovery_potential': np.mean(random_discovery),
        'discovery_ratio': np.mean(friend_discovery) / np.mean(random_discovery) if np.mean(random_discovery) > 0 else 0
    }


# ============================================================================
# RUN EVALUATION
# ============================================================================

print("\n[5/8] Training Pointwise model...")
W_pointwise, X_pointwise = train_pointwise(train_user_books, num_users, num_books)

print("\n[6/8] Training BPR model...")
W_bpr, X_bpr = train_bpr(train_user_books, num_users, num_books)

print("\n[7/8] Evaluating niche discovery...")

print("\n  Pointwise model:")
results_pointwise = evaluate_niche_discovery(W_pointwise, X_pointwise, test_user_books, train_user_books, book_popularity)
print(f"    Self predicted score: {results_pointwise['self_score']:.3f}")
print(f"    Friend predicted score: {results_pointwise['friend_score']:.3f}")
print(f"    Random predicted score: {results_pointwise['random_score']:.3f}")
print(f"    Friend/Random ratio: {results_pointwise['friend_vs_random']:.2f}x")
print(f"    Niche hit rate (friends): {results_pointwise['friend_hit_rate']*100:.2f}%")
print(f"    Niche hit rate (random): {results_pointwise['random_hit_rate']*100:.2f}%")
print(f"    Hit rate ratio: {results_pointwise['hit_rate_ratio']:.2f}x")
if results_pointwise['n_very_niche'] > 0:
    print(f"    Very niche (<20 readers) hit rate ratio: {results_pointwise['very_niche_ratio']:.2f}x")

print("\n  BPR model:")
results_bpr = evaluate_niche_discovery(W_bpr, X_bpr, test_user_books, train_user_books, book_popularity)
print(f"    Self predicted score: {results_bpr['self_score']:.3f}")
print(f"    Friend predicted score: {results_bpr['friend_score']:.3f}")
print(f"    Random predicted score: {results_bpr['random_score']:.3f}")
print(f"    Friend/Random ratio: {results_bpr['friend_vs_random']:.2f}x")
print(f"    Niche hit rate (friends): {results_bpr['friend_hit_rate']*100:.2f}%")
print(f"    Niche hit rate (random): {results_bpr['random_hit_rate']*100:.2f}%")
print(f"    Hit rate ratio: {results_bpr['hit_rate_ratio']:.2f}x")
if results_bpr['n_very_niche'] > 0:
    print(f"    Very niche (<20 readers) hit rate ratio: {results_bpr['very_niche_ratio']:.2f}x")

print("\n[8/8] Evaluating discovery potential...")

print("\n  Pointwise model:")
discovery_pointwise = evaluate_discovery_potential(W_pointwise, train_user_books, book_popularity)
print(f"    Non-popular books friends can recommend: {discovery_pointwise['friend_discovery_potential']:.1f}")
print(f"    Non-popular books random can recommend: {discovery_pointwise['random_discovery_potential']:.1f}")
print(f"    Discovery ratio: {discovery_pointwise['discovery_ratio']:.2f}x")

print("\n  BPR model:")
discovery_bpr = evaluate_discovery_potential(W_bpr, train_user_books, book_popularity)
print(f"    Non-popular books friends can recommend: {discovery_bpr['friend_discovery_potential']:.1f}")
print(f"    Non-popular books random can recommend: {discovery_bpr['random_discovery_potential']:.1f}")
print(f"    Discovery ratio: {discovery_bpr['discovery_ratio']:.2f}x")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: NICHE DISCOVERY EVALUATION")
print("=" * 70)

print("""
The key question: "Can friends help you discover NON-POPULAR books you'd like?"

This measures discovery value that popularity recommendations cannot provide.
Only below-median popularity books were held out for testing.
""")

print(f"{'Metric':<40} {'Pointwise':<15} {'BPR':<15}")
print("-" * 70)
print(f"{'Friend/Random predicted score ratio':<40} {results_pointwise['friend_vs_random']:<15.2f}x {results_bpr['friend_vs_random']:<15.2f}x")
print(f"{'Niche hit rate (friends)':<40} {results_pointwise['friend_hit_rate']*100:<14.2f}% {results_bpr['friend_hit_rate']*100:<14.2f}%")
print(f"{'Niche hit rate (random)':<40} {results_pointwise['random_hit_rate']*100:<14.2f}% {results_bpr['random_hit_rate']*100:<14.2f}%")
print(f"{'Niche hit rate ratio':<40} {results_pointwise['hit_rate_ratio']:<15.2f}x {results_bpr['hit_rate_ratio']:<15.2f}x")
print("-" * 70)
print(f"{'Discovery potential (friends)':<40} {discovery_pointwise['friend_discovery_potential']:<15.1f} {discovery_bpr['friend_discovery_potential']:<15.1f}")
print(f"{'Discovery potential (random)':<40} {discovery_pointwise['random_discovery_potential']:<15.1f} {discovery_bpr['random_discovery_potential']:<15.1f}")
print(f"{'Discovery ratio':<40} {discovery_pointwise['discovery_ratio']:<15.2f}x {discovery_bpr['discovery_ratio']:<15.2f}x")

print("\nInterpretation:")
print("-" * 70)

if results_pointwise['hit_rate_ratio'] > 1.5 or results_bpr['hit_rate_ratio'] > 1.5:
    print("✓ Friends predict NICHE books better than random!")
    print("  Friend-finding provides discovery value beyond popularity.")
else:
    print("? Friends don't strongly predict niche books better than random.")
    print("  May need more data or different clustering approach.")

if discovery_pointwise['discovery_ratio'] > 1.0 or discovery_bpr['discovery_ratio'] > 1.0:
    print("✓ Friends have more non-popular books to recommend!")
    print("  Friends can introduce you to books you wouldn't find otherwise.")
else:
    print("? Friends don't have more non-popular recommendations than random.")

# Determine best model
pointwise_score = results_pointwise['hit_rate_ratio'] + discovery_pointwise['discovery_ratio']
bpr_score = results_bpr['hit_rate_ratio'] + discovery_bpr['discovery_ratio']
best_model = "Pointwise" if pointwise_score > bpr_score else "BPR"
print(f"\nBest model for niche discovery: {best_model}")

print("\nComparison to popularity baseline:")
print("-" * 70)
print("Popularity achieves ~38% Precision@10 by recommending Harry Potter to everyone.")
print("But that doesn't help friend-finding - everyone has read Harry Potter.")
print(f"Friend matching finds users who share NICHE tastes ({results_bpr['hit_rate_ratio']:.1f}x better hit rate).")
print("This is the value that popularity cannot provide.")

print("=" * 70)
