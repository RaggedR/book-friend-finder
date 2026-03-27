#!/usr/bin/env python3
"""
Pre-compute All Recommendations

Run this locally to generate all friend recommendations.
The output files can then be deployed to Cloud Run with no ML dependencies.

Usage:
    python3 precompute_all.py

Output:
    webapp/data/recommendations.json - All pre-computed friend matches
    webapp/data/users.json - User list for dropdown
"""

import json
import sqlite3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration
DATA_DIR = os.path.expanduser("~/data/hardcover/")
BOOKS_USERS_DB = os.path.join(DATA_DIR, "books_users.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "webapp", "data")

MIN_RATINGS_PER_USER = 20
MIN_USERS_PER_BOOK = 5
MAX_BOOK_POPULARITY_PCT = 0.10  # Drop books read by >X% of users (1.0 = disabled)
NUM_FEATURES = 30
LAMBDA = 1.0
ITERATIONS = 100
LEARNING_RATE = 0.1
NEG_SAMPLES_PER_POS = 4
USE_BPR = True

print("="*80)
print("PRE-COMPUTING FRIEND RECOMMENDATIONS")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA AND TRAIN MODEL
# ============================================================================
print("\n[1/3] Loading data and training model...")

# Connect to SQLite database
print("  Loading data from SQLite...")
conn = sqlite3.connect(BOOKS_USERS_DB)
cursor = conn.cursor()

# Get book user counts
cursor.execute("""
    SELECT book_id, COUNT(*) as user_count
    FROM book_users
    GROUP BY book_id
""")
book_user_counts = {row[0]: row[1] for row in cursor.fetchall()}

# Get book metadata
cursor.execute("SELECT id, title, slug, image_url, cached_contributors FROM books")
books_metadata = {
    row[0]: {'id': row[0], 'title': row[1], 'slug': row[2], 'image_url': row[3], 'cached_contributors': row[4]}
    for row in cursor.fetchall()
}

# Get all book-user relationships
cursor.execute("""
    SELECT book_id, user_id, username, name, status, status_id, rating, review_raw
    FROM book_users
""")
book_users_data = cursor.fetchall()

print(f"  Loaded {len(books_metadata)} books, {len(book_users_data)} book-user relationships")

# Load unique users from SQLite
cursor.execute("""
    SELECT DISTINCT user_id, username, name
    FROM book_users
    ORDER BY user_id
""")
unique_users_raw = cursor.fetchall()
print(f"  Loaded {len(unique_users_raw)} unique users")

# Build users_data structure (compatible with old JSON format)
users_data = {
    'user_books': [
        {'user': {'id': row[0], 'username': row[1], 'name': row[2]}}
        for row in unique_users_raw
    ]
}

# Build books_data structure (compatible with old JSON format)
books_by_id = defaultdict(lambda: {'users': []})
for row in book_users_data:
    book_id, user_id, username, name, status, status_id, rating, review_raw = row
    books_by_id[book_id]['users'].append({
        'user_id': user_id,
        'username': username,
        'name': name,
        'status': status,
        'status_id': status_id,
        'rating': rating,
        'review_raw': review_raw
    })

# Create books list with metadata
books_list = []
for book_id, book_data in books_by_id.items():
    if book_id in books_metadata:
        books_list.append({
            'book': books_metadata[book_id],
            'users': book_data['users'],
            'user_count': len(book_data['users'])
        })

books_data = {'books': books_list}
print(f"  Built books_data with {len(books_list)} books")

conn.close()

# Helper functions
def get_user_count(b):
    if 'user_count' in b:
        return b['user_count']
    return len(b.get('users', []))

def get_book_info(b):
    if 'book' in b and isinstance(b['book'], dict):
        return b['book']
    return b

STATUS_MAP = {'want_to_read': 1, 'currently_reading': 2, 'read': 3, 'did_not_finish': 5}

def get_user_entry_info(user_entry):
    return user_entry.get('user_id'), user_entry.get('status_id', 0), user_entry.get('rating')

# Deduplicate users
seen_user_ids = set()
unique_user_books = []
for u in users_data['user_books']:
    uid = u['user']['id']
    if uid not in seen_user_ids:
        seen_user_ids.add(uid)
        unique_user_books.append(u)

print(f"  Deduplicated: {len(users_data['user_books'])} -> {len(unique_user_books)} users")

# Filter books (need 5+ users)
filtered_books = [
    b for b in books_data['books']
    if get_user_count(b) >= MIN_USERS_PER_BOOK
]
print(f"  {len(filtered_books)} books with {MIN_USERS_PER_BOOK}+ users")

# Filter out overly popular books
if MAX_BOOK_POPULARITY_PCT < 1.0:
    total_users = len(unique_user_books)
    max_readers = int(total_users * MAX_BOOK_POPULARITY_PCT)
    before_count = len(filtered_books)
    filtered_books = [b for b in filtered_books if get_user_count(b) <= max_readers]
    dropped = before_count - len(filtered_books)
    print(f"  Dropped {dropped} books with >{max_readers} readers (>{MAX_BOOK_POPULARITY_PCT*100:.0f}% of users)")
    print(f"  {len(filtered_books)} books remaining after popularity filter")

# Count book interactions per user
print("  Counting book interactions per user...")
user_book_counts = defaultdict(int)
all_user_ids = {u['user']['id'] for u in unique_user_books}

for book_entry in filtered_books:
    for user_entry in book_entry['users']:
        user_id, status_id, rating = get_user_entry_info(user_entry)
        if user_id not in all_user_ids:
            continue
        if status_id in (1, 2, 3):
            user_book_counts[user_id] += 1

# Filter users
filtered_users = [
    u for u in unique_user_books
    if user_book_counts[u['user']['id']] >= MIN_RATINGS_PER_USER
]
print(f"  {len(filtered_users)} users with {MIN_RATINGS_PER_USER}+ books")

# Build mappings
user_id_to_idx = {u['user']['id']: idx for idx, u in enumerate(filtered_users)}
user_idx_to_id = {idx: u['user']['id'] for idx, u in enumerate(filtered_users)}
book_id_to_idx = {get_book_info(b)['id']: idx for idx, b in enumerate(filtered_books)}
book_idx_to_title = {idx: get_book_info(b)['title'] for idx, b in enumerate(filtered_books)}
book_idx_to_slug = {idx: get_book_info(b).get('slug', '') for idx, b in enumerate(filtered_books)}

# Book popularity weights (IDF-style)
max_book_popularity = max(get_user_count(b) for b in filtered_books)
book_idx_to_weight = {
    book_id_to_idx[get_book_info(b)['id']]: np.log(max_book_popularity / get_user_count(b)) + 1
    for b in filtered_books
}

num_books = len(filtered_books)
num_users = len(filtered_users)
print(f"  Final: {num_users} users, {num_books} books")

# Build interaction data
Y_raw = np.full((num_books, num_users), np.nan)
R = np.zeros((num_books, num_users))
user_books_dict = defaultdict(lambda: {'read': [], 'want': [], 'current': []})

for book_entry in filtered_books:
    book_info = get_book_info(book_entry)
    book_idx = book_id_to_idx[book_info['id']]

    for user_entry in book_entry['users']:
        user_id, status_id, rating = get_user_entry_info(user_entry)
        if user_id not in user_id_to_idx:
            continue

        user_idx = user_id_to_idx[user_id]

        if status_id == 3:
            user_books_dict[user_idx]['read'].append(book_idx)
            Y_raw[book_idx, user_idx] = rating / 5.0 if rating is not None else 0.7
            R[book_idx, user_idx] = 1
        elif status_id == 2:
            user_books_dict[user_idx]['current'].append(book_idx)
            Y_raw[book_idx, user_idx] = 0.7
            R[book_idx, user_idx] = 1
        elif status_id == 1:
            user_books_dict[user_idx]['want'].append(book_idx)
            Y_raw[book_idx, user_idx] = 0.3
            R[book_idx, user_idx] = 1
        elif status_id == 5:
            Y_raw[book_idx, user_idx] = 0.0
            R[book_idx, user_idx] = 1

Y = np.nan_to_num(Y_raw, nan=0.5)

# Train model
print("  Training collaborative filtering (BPR)...")

def train_bpr(num_users, num_books, user_positives, iterations=ITERATIONS, neg_samples=NEG_SAMPLES_PER_POS):
    tf.random.set_seed(42)
    W = tf.Variable(tf.random.normal((num_users, NUM_FEATURES), stddev=0.1), name='W')
    X = tf.Variable(tf.random.normal((num_books, NUM_FEATURES), stddev=0.1), name='X')

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    pos_interactions = []
    for user_idx, book_indices in user_positives.items():
        for book_idx in book_indices:
            pos_interactions.append((user_idx, book_idx))

    print(f"    Training on {len(pos_interactions):,} positive interactions")

    for iter in range(iterations):
        np.random.seed(42 + iter)
        users_list, pos_list, neg_list = [], [], []

        for u, pos_b in pos_interactions:
            user_pos = user_positives[u]
            for _ in range(neg_samples):
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
            print(f"    Iteration {iter + 1}/{iterations}, loss={float(total_loss):.4f}")

    return W, X

# Build user_positives
user_positives = defaultdict(set)
for user_idx in range(num_users):
    for book_idx in user_books_dict[user_idx]['read']:
        user_positives[user_idx].add(book_idx)
    for book_idx in user_books_dict[user_idx]['current']:
        user_positives[user_idx].add(book_idx)

W, X = train_bpr(num_users, num_books, user_positives)

# ============================================================================
# 2. COMPUTE SIMILARITIES AND RECOMMENDATIONS
# ============================================================================
print("\n[2/3] Computing friend recommendations for all users...")

user_features = W.numpy()
user_features_normalized = normalize(user_features, norm='l2', axis=1)
# Handle any NaN/inf values
user_features_normalized = np.nan_to_num(user_features_normalized, nan=0.0, posinf=0.0, neginf=0.0)

# Compute similarity matrix for ALL users (no cluster restriction)
similarity_matrix = cosine_similarity(user_features_normalized)

recommendations = {}

for user_idx in range(num_users):
    user_id = user_idx_to_id[user_idx]
    user = filtered_users[user_idx]['user']

    # Get similarities to ALL other users
    user_similarities = similarity_matrix[user_idx].copy()
    user_similarities[user_idx] = -np.inf  # Exclude self

    # Get top 10 most similar users
    top_indices = np.argsort(-user_similarities)[:10]

    matches = []
    for other_idx in top_indices:
        other_user = filtered_users[other_idx]['user']
        similarity = float(user_similarities[other_idx])

        # Find shared books
        user_read = set(user_books_dict[user_idx]['read'])
        other_read = set(user_books_dict[other_idx]['read'])
        shared_books = user_read & other_read

        # Sort by rarity (IDF weight)
        shared_books_sorted = sorted(shared_books, key=lambda idx: book_idx_to_weight[idx], reverse=True)
        shared_book_info = [
            {'title': book_idx_to_title[idx], 'slug': book_idx_to_slug[idx]}
            for idx in shared_books_sorted[:10]
        ]

        # Weighted score
        shared_books_score = sum(book_idx_to_weight[idx] for idx in shared_books)

        # Books they've read that you want
        user_want = set(user_books_dict[user_idx]['want'])
        can_recommend = (other_read - user_read) & user_want
        can_recommend_sorted = sorted(can_recommend, key=lambda idx: book_idx_to_weight[idx], reverse=True)
        recommend_info = [
            {'title': book_idx_to_title[idx], 'slug': book_idx_to_slug[idx]}
            for idx in can_recommend_sorted[:5]
        ]

        matches.append({
            'user_id': int(user_idx_to_id[other_idx]),
            'name': other_user['name'],
            'username': other_user['username'],
            'similarity': round(similarity * 100, 1),
            'shared_books': shared_book_info,
            'shared_books_total': len(shared_books),
            'shared_books_score': round(shared_books_score, 1),
            'can_recommend': recommend_info,
            'num_read': len(other_read)
        })

    recommendations[str(user_id)] = {
        'user': {
            'id': int(user_id),
            'name': user['name'],
            'username': user['username']
        },
        'num_read': len(user_books_dict[user_idx]['read']),
        'num_want': len(user_books_dict[user_idx]['want']),
        'num_current': len(user_books_dict[user_idx]['current']),
        'matches': matches
    }

    if (user_idx + 1) % 100 == 0:
        print(f"    Processed {user_idx + 1}/{num_users} users")

print(f"  ✓ Pre-computed recommendations for {num_users} users")

# ============================================================================
# 3. SAVE OUTPUT FILES
# ============================================================================
print("\n[3/3] Saving output files...")

# Save recommendations
recommendations_file = os.path.join(OUTPUT_DIR, "recommendations.json")
with open(recommendations_file, 'w') as f:
    json.dump(recommendations, f, indent=2)
print(f"  ✓ Saved recommendations.json ({os.path.getsize(recommendations_file) / 1024 / 1024:.1f}MB)")

# Save users list
users_list = [
    {
        'id': int(u['user']['id']),
        'name': u['user']['name'],
        'username': u['user']['username']
    }
    for u in filtered_users
]
users_list.sort(key=lambda x: x['name'].lower())

users_file = os.path.join(OUTPUT_DIR, "users.json")
with open(users_file, 'w') as f:
    json.dump(users_list, f, indent=2)
print(f"  ✓ Saved users.json ({len(users_list)} users)")

print("\n" + "="*80)
print("✓ PRE-COMPUTATION COMPLETE!")
print("="*80)
print(f"\nOutput files in: {OUTPUT_DIR}/")
print(f"  - recommendations.json")
print(f"  - users.json")
print(f"\nNext steps:")
print(f"  1. Review the generated files")
print(f"  2. Deploy webapp/ directory to Cloud Run")
print("="*80)
