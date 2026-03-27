# Implementation Details

This document provides in-depth technical documentation of the book recommendation system, including the algorithms, design decisions, and hyperparameters.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Pipeline](#data-pipeline)
3. [Matrix Factorization Algorithm](#matrix-factorization-algorithm)
4. [Friend Matching](#friend-matching)
5. [Hyperparameters](#hyperparameters)
6. [Design Decisions](#design-decisions)
7. [Algorithm Improvements](#algorithm-improvements)
8. [Class Imbalance Solution: BPR Loss](#class-imbalance-solution-bpr-loss)
9. [The Sparse Matrix Problem](#the-sparse-matrix-problem)

---

## System Overview

The system finds "book friends" - users with similar reading tastes - using collaborative filtering. The core idea: users who have rated the same books similarly will likely enjoy similar books in the future.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION                                  │
│  scripts/process_batch.py → Hardcover GraphQL API                       │
│  Output: ~/data/hardcover/{users.json, books_users.db}                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                                   │
│  precompute_all.py                                                       │
│  1. Load and filter data                                                 │
│  2. Build interaction matrix with implicit feedback                      │
│  3. Train BPR matrix factorization model (TensorFlow)                    │
│  4. Compute all-pairs similarity and pre-compute friend matches          │
│  Output: webapp/data/{recommendations.json, users.json}                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEB APPLICATION                                  │
│  webapp/app.py (Flask)                                                   │
│  - Serves pre-computed recommendations (no ML at runtime)                │
│  - Chat functionality between matched users                              │
│  - Deployed to Google Cloud Run                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principle: Offline ML

All machine learning happens locally via `precompute_all.py`. The production webapp only serves pre-computed JSON files. This means:

- No TensorFlow/scikit-learn dependencies in production
- Fast response times (simple JSON lookups)
- Easy deployment (just static files + Flask)
- Reproducible results (same input data → same output)

---

## Data Pipeline

### Input Files

| File | Description | Size |
|------|-------------|------|
| `~/data/hardcover/users.json` | User profiles | ~4MB |
| `~/data/hardcover/books_users.db` | SQLite database: books → users who read them | ~1GB |

The SQLite database contains:
- `books` table: book metadata (id, title, slug, image_url, cached_contributors)
- `book_users` table: relationships (book_id, user_id, status, rating, etc.)
- Indexes for fast queries by user_id and book_id

### Data Filtering

Before training, we filter to ensure quality:

```python
MIN_RATINGS_PER_USER = 20  # Users with fewer interactions are unreliable
MIN_USERS_PER_BOOK = 5     # Books with fewer readers don't help find patterns
```

**Rationale:**
- Users with very few ratings don't provide enough signal for meaningful embeddings
- Obscure books (< 5 readers) add noise without helping similarity detection
- These thresholds balance data quality vs. coverage

### Book Status IDs (from Hardcover API)

| Status ID | Meaning | Implicit Feedback Weight |
|-----------|---------|-------------------------|
| 1 | Want to read | 0.3 (mild positive signal) |
| 2 | Currently reading | 0.7 (strong positive signal) |
| 3 | Read | Rating/5.0 or 0.7 if no rating |
| 5 | Did not finish | 0.0 (negative signal) |

---

## Matrix Factorization Algorithm

### Overview

We use collaborative filtering with matrix factorization, similar to the Netflix Prize approach. The goal is to decompose the user-book interaction matrix into two low-rank matrices:

```
R ≈ sigmoid(X × W^T + b)

Where:
- R: (num_books × num_users) interaction matrix
- X: (num_books × k) book feature matrix (latent factors)
- W: (num_users × k) user feature matrix (latent factors)
- b: (1 × num_users) user bias terms
- k: number of latent features (NUM_FEATURES = 30)
```

### Implicit Feedback

Unlike explicit ratings (1-5 stars), we use implicit feedback that captures user intent:

```python
# Read books: use actual rating normalized to 0.2-1.0
if status_id == 3:
    if rating is not None:
        # 1-star → 0.2, 5-star → 1.0
        weight = rating / 5.0
    else:
        # No rating but marked read → assume positive
        weight = 0.7

# Currently reading: strong positive signal
elif status_id == 2:
    weight = 0.7

# Want to read: mild positive signal
elif status_id == 1:
    weight = 0.3

# Did not finish: negative signal
elif status_id == 5:
    weight = 0.0

# Unrated/unknown: neutral (0.5, masked during training)
```

**Rationale for weights:**
- "Read" is the strongest signal, but we now use actual ratings for granularity
- "Currently reading" shows active interest
- "Want to read" shows intent but not confirmation of taste
- "DNF" is a clear negative signal

### Cost Function

```python
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    # Mask out unrated items (Y == 0.5 means unknown)
    R_mask = tf.where(tf.equal(Y, 0.5),
                      tf.constant(0.0, dtype=tf.float32),
                      tf.constant(1.0, dtype=tf.float32))

    # Predictions through sigmoid (outputs 0-1)
    logits = tf.matmul(X, W, transpose_b=True) + b
    probs = tf.sigmoid(logits)

    # Squared error loss (only on known interactions)
    loss = (probs - Y) ** 2
    masked_loss = loss * R_mask

    # L2 regularization on feature matrices
    total_cost = tf.reduce_sum(masked_loss) + (lambda_ / 2) * (
        tf.reduce_sum(X**2) + tf.reduce_sum(W**2)
    )
    return total_cost
```

**Key points:**
- **Sigmoid activation**: Keeps predictions in [0, 1] range
- **Squared error**: Simple, works well for implicit feedback
- **Masking**: Only trains on known interactions (not the 0.5 neutral values)
- **L2 regularization**: Prevents overfitting, controlled by `LAMBDA`

### Training Process

```python
optimizer = keras.optimizers.Adam(learning_rate=0.1)

for iter in range(100):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Y, R, LAMBDA)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

**Why Adam optimizer:**
- Adaptive learning rates per parameter
- Handles sparse gradients well (many zero entries in R)
- Converges faster than vanilla SGD

---

## Friend Matching

### Similarity Metric

We compute cosine similarity between ALL user pairs using the learned embeddings:

```python
# L2 normalize user embeddings
user_features_normalized = normalize(user_features, norm='l2', axis=1)

# Compute full similarity matrix
similarity_matrix = cosine_similarity(user_features_normalized)
```

Cosine similarity measures the angle between user vectors:
- 1.0 = identical taste profiles
- 0.0 = orthogonal (no overlap)
- -1.0 = opposite tastes (rare in practice)

### Match Selection

For each user:
1. Compute cosine similarity with ALL other users
2. Sort by similarity (descending)
3. Return top 10 matches

**Note:** Earlier versions used K-means clustering to limit comparisons to within-cluster matches only. This was removed in Jan 2025 because:
1. Users near cluster boundaries could miss good matches in adjacent clusters
2. With ~4K users, O(n²) computation is still fast (~16M comparisons)
3. Simpler code with better match quality

### Shared Book Analysis

For each match, we identify:

1. **Shared books**: Books both users have read (with rating ≥ 3 or no rating)
2. **Can recommend**: Books the match has read that you want to read

Both lists are now sorted by IDF weight (rarest books first) to surface the most meaningful connections.

### What We Want From Friend-Finding

Friend-finding is fundamentally different from book recommendation. Recommending popular books achieves ~38% precision - but that doesn't help you find friends. Everyone has read Harry Potter; that's not a meaningful connection.

**What makes a good book friend:**

1. **Discovery potential**: They've read books you'd like that you wouldn't have found otherwise. A friend who only recommends bestsellers is no better than a "most popular" list.

2. **Shared niche tastes**: You both loved the same obscure books. Agreement on popular books is noise; agreement on rare books is signal.

3. **Conversation potential**: You could have a meaningful discussion about books you've both read. This requires shared reads that aren't universally read.

4. **Complementary reading**: They've read books on your "want to read" list (they can tell you if it's worth it) or books you've never heard of in genres you like.

**Why popularity-based metrics are misleading for friend-finding:**

The popularity baseline (~38% Precision@10) measures "will you like this book?" But friend-finding should measure "will this person help you discover books you wouldn't find otherwise?" These are different questions:

| Question | Best strategy | Metric |
|----------|---------------|--------|
| Will you like this book? | Recommend popular books | Precision@K |
| Will this friend help you discover? | Match on niche tastes | Niche hit rate |

**Metrics we should use for friend-finding:**

1. **Niche hit rate**: Hold out non-popular books (below median popularity). Do friends predict these better than random? This directly measures discovery value.

2. **Non-popular discovery rate**: Of a friend's books you haven't read, what % are outside the top 100? Higher = more discovery potential.

3. **Shared rare books**: Count books both users rated 4+ with <500 readers. These are "we both loved this obscure gem" moments - the foundation of book friendships.

4. **Recommendation pool depth**: Books they've read that you haven't, aren't popular, and they rated highly. This is actionable: "here's something you'd probably like that you've never heard of."

The current predictive evaluation (3x hit rate) is promising but doesn't exclude popular books. A friend who recommends Harry Potter isn't providing value. We need metrics that specifically measure non-obvious discovery.

---

## Hyperparameters

All tunable constants are defined at the top of `precompute_all.py`:

| Parameter | Value | Purpose | Tuning Notes |
|-----------|-------|---------|--------------|
| `MIN_RATINGS_PER_USER` | 20 | Filter inactive users | Lower = more users but noisier embeddings |
| `MIN_USERS_PER_BOOK` | 5 | Filter obscure books | Lower = more books but less signal |
| `MAX_BOOK_POPULARITY_PCT` | 0.10 | Drop overly popular books | Lower = more aggressive filtering, better match quality |
| `NUM_FEATURES` | 30 | Latent dimensions | Tuned: 30 beats 20, 50 overfits |
| `LAMBDA` | 1.0 | L2 regularization strength | Higher = simpler model, less overfitting |
| `ITERATIONS` | 100 | Training epochs | More = better fit but diminishing returns |
| `LEARNING_RATE` | 0.1 | Adam optimizer LR | Higher = faster but may overshoot |

---

## Design Decisions

### 1. Offline Pre-computation vs. Real-time

**Decision:** Pre-compute all recommendations offline.

**Why:**
- Production webapp needs no ML dependencies
- Sub-millisecond response times
- Reproducible, testable results
- Easy deployment (just JSON files)

**Trade-off:** Recommendations don't update until re-running `precompute_all.py`.

### 2. Global Matching vs. Cluster-based Matching

**Decision:** Compare all users against all users for friend matching.

**Why:**
- Users near cluster boundaries don't miss good matches in adjacent clusters
- With ~4K users, O(n²) = 16M comparisons is still fast
- Simpler code, no clustering hyperparameters to tune
- Better match quality (global optimum, not local within cluster)

**Trade-off:** Computation scales as O(n²). May need clustering for >50K users.

### 3. Matrix Factorization vs. Neural Collaborative Filtering

**Decision:** Use traditional matrix factorization with gradient descent.

**Why:**
- Works well with sparse data (users rate ~20-100 books out of millions)
- Fewer parameters, less overfitting risk
- Faster training
- More interpretable (latent factors)
- Research shows well-tuned MF often matches neural approaches on pure collaborative filtering

**Trade-off:** Can't easily incorporate side features (user demographics, book metadata).

### 4. Top 10 Matches Only

**Decision:** Store and display only top 10 matches per user.

**Why:**
- Reduces output file size significantly
- Users won't meaningfully interact with 100+ suggestions
- Focuses attention on best matches

**Trade-off:** Can't explore "medium quality" matches.

### 5. Chat Access Control

**Decision:** Users can only message their top 10 matches.

**Why:**
- Prevents spam
- Ensures conversations are between genuinely similar readers
- Creates natural scarcity/value

---

## Algorithm Improvements

### Improvement 1: Book Popularity Weighting (IDF-style)

**Problem:** Two users both reading Harry Potter is less meaningful than both reading an obscure novel. Popular books are weak signals for similarity.

**Solution:** Weight shared books using inverse document frequency:

```python
# IDF weight: log(max_popularity / book_popularity) + 1
max_book_popularity = max(b['user_count'] for b in filtered_books)
book_idx_to_weight = {
    idx: np.log(max_book_popularity / book['user_count']) + 1
    for idx, book in enumerate(filtered_books)
}
```

**Effect:**
- Popular books (e.g., 1000 readers): weight ≈ 1.0
- Medium books (e.g., 100 readers): weight ≈ 3.3
- Rare books (e.g., 10 readers): weight ≈ 5.6

**Usage:**
1. Shared books are sorted by weight (rarest first) in the display
2. A `shared_books_score` field sums IDF weights for each match
3. This surfaces more meaningful connections (obscure shared interests)

### Improvement 2: Granular Rating Values

**Problem:** Original implementation treated all ratings ≥ 3 as 1.0 (positive) and < 3 as 0.0 (negative). This loses information: a 5-star rating is stronger signal than a 3-star.

**Solution:** Use normalized actual ratings:

```python
# Before (binary):
Y_raw[book_idx, user_idx] = 1.0 if (rating >= 3) else 0.0

# After (granular):
if rating is not None:
    Y_raw[book_idx, user_idx] = rating / 5.0  # 1→0.2, 5→1.0
else:
    Y_raw[book_idx, user_idx] = 0.7  # No rating = assume positive
```

**Effect:**
- 5-star: 1.0 (strong positive)
- 4-star: 0.8
- 3-star: 0.6
- 2-star: 0.4
- 1-star: 0.2 (weak positive, still read it)
- No rating: 0.7 (default positive assumption)

**Rationale:** Users who both 5-starred a book share stronger taste alignment than users who gave it 3 stars. This gradient provides more training signal.

### Improvement 3: Popular Book Filtering

**Problem:** Extremely popular books (Harry Potter, 1984, Dune) are read by 30-45% of all users. When the model learns user embeddings, these books dominate the signal. Two users who both read Harry Potter doesn't indicate similar taste - it just means they're both readers. The model wastes capacity learning "most people read popular books" instead of distinctive taste patterns.

**Solution:** Before training, drop books read by more than 10% of users:

```python
MAX_BOOK_POPULARITY_PCT = 0.10  # Drop books read by >10% of users

if MAX_BOOK_POPULARITY_PCT < 1.0:
    max_readers = int(total_users * MAX_BOOK_POPULARITY_PCT)
    filtered_books = [b for b in filtered_books if get_user_count(b) <= max_readers]
```

**Experimental Results (2025-01-25):**

| Threshold | Books Dropped | Friend/Random Score | Hit Rate Ratio |
|-----------|---------------|---------------------|----------------|
| No filter | 0             | 1.17x               | 2.79x          |
| Drop >30% | 15            | 1.17x               | 3.69x (+32%)   |
| Drop >20% | 43            | 1.20x               | 4.17x (+49%)   |
| Drop >10% | 217           | 1.21x               | **5.50x (+97%)** |

**What Hit Rate Ratio measures:** For each user, we hold out 20% of their liked books. We then check if their matched friends have actually read those held-out books. The ratio compares friends vs random users. A ratio of 5.50x means friends are 5.5x more likely to have read books you'd like.

**Why this works:**
1. Forces the model to learn from distinctive reading patterns, not mainstream popularity
2. Friends matched on niche interests share more meaningful taste overlap
3. The absolute hit rate drops (8.38% → 5.36%) because there are fewer books to match on, but the *quality* of matches improves dramatically

**Trade-off:** Users who have only read popular books will have less training data. However, with ~52,000 books remaining after filtering, this is rarely a problem in practice.

**Configuration:** Set `MAX_BOOK_POPULARITY_PCT = 1.0` to disable filtering.

### Improvement 4: Hyperparameter Tuning

**Problem:** The default values for `NUM_FEATURES` (latent dimensions) were chosen heuristically. Better values might improve friend-finding quality.

**Experiment (2025-01-25):** Tested multiple values using predictive evaluation (hit rate ratio).

**NUM_FEATURES Results:**

| Features | Friend/Random | Hit Rate Ratio |
|----------|---------------|----------------|
| 10       | 1.15x         | 4.94x          |
| 20       | 1.21x         | 5.32x          |
| **30**   | **1.24x**     | **5.50x**      |
| 50       | 1.25x         | 5.29x          |

**Finding:** 30 features is optimal. Fewer features (10) lack expressiveness. More features (50) start to overfit on sparse data.

**Changes made:**
- `NUM_FEATURES`: 20 → 30

**Note:** Clustering was also evaluated but later removed (Jan 2025) in favor of global all-pairs matching for better match quality.

**Evaluation script:** `evaluate_hyperparams.py`

---

## Class Imbalance Solution: BPR Loss

### The Problem

Our dataset has severe class imbalance: **94.7% positive** (liked) vs **5.3% negative** (disliked/DNF).

This happens because:
1. Users finish and rate books they enjoy
2. Users silently abandon books they dislike (survivorship bias)
3. DNF (Did Not Finish) is rarely explicitly marked

With the original pointwise sigmoid cross-entropy loss, the model learns that predicting "like" for everything achieves ~95% accuracy. This is useless for recommendations.

### Solution: BPR (Bayesian Personalized Ranking)

Based on literature review of arXiv papers on class imbalance in recommendation systems, we implemented **BPR (Bayesian Personalized Ranking)** from [Rendle et al. 2009](https://arxiv.org/abs/1205.2618).

Instead of treating recommendation as **classification** (will user like this book?), BPR treats it as **ranking** (does user prefer book A over book B?).

**Pointwise loss** (original):
```python
L = sigmoid_cross_entropy(score(user, book), label)
```

**BPR pairwise loss** (new):
```python
L = -log(sigmoid(score(user, positive_book) - score(user, negative_book)))
```

BPR samples random unobserved books as negatives, avoiding the class imbalance problem entirely.

### Experimental Results (2025-01-20)

*Note: These benchmarks were measured on an earlier dataset. Current dataset (Feb 2026) has ~10K users and ~125K books.*

Original comparison with:
- 2,682 users, 36,155 books
- 385,688 interactions (94.7% positive)
- 80/20 train/test split
- 50 training iterations, 20 latent features, λ=1.0

| Metric | Pointwise | BPR | Improvement |
|--------|-----------|-----|-------------|
| **Precision@5** | 9.80% | **19.36%** | +97% |
| **Precision@10** | 8.22% | **15.20%** | +85% |
| **Precision@20** | 6.56% | **11.85%** | +81% |
| **NDCG@5** | 0.1045 | **0.2101** | +101% |
| **NDCG@10** | 0.0970 | **0.1835** | +89% |
| **NDCG@20** | 0.0966 | **0.1738** | +80% |

**Key finding**: BPR nearly doubles recommendation quality compared to the old pointwise method.

### Baseline Comparisons

How do our models compare to simpler approaches?

| Method | Precision@10 | Description |
|--------|-------------|-------------|
| **Random** | 0.75% | Recommend 10 random books |
| **Old Pointwise** | 8.22% | Original sigmoid cross-entropy model |
| **BPR** | 15.20% | New pairwise ranking model |
| **Popularity** | ~38% | Recommend the 10 most popular books to everyone |

**Top 10 most popular books in the dataset:**
1. 1984 (45% of users)
2. Harry Potter and the Sorcerer's Stone (45%)
3. Project Hail Mary (40%)
4. The Hunger Games (37%)
5. Dune (37%)
6. The Hobbit (35%)
7. Harry Potter and the Chamber of Secrets (35%)
8. Animal Farm (35%)
9. Harry Potter and the Prisoner of Azkaban (35%)
10. Harry Potter and the Goblet of Fire (34%)

### Why Popularity is Hard to Beat

With ~2,800 users, the popularity baseline (~38%) still outperforms BPR (15.2%). This is a known challenge in recommendation systems:

1. **Sparse data**: Each user has rated only ~0.3% of books. Not enough overlap between users to learn personalization patterns.
2. **Popular books are popular for a reason**: Recommending bestsellers works because many people genuinely enjoy them.
3. **Cold start**: With few users, the model can't distinguish "likes sci-fi" from "likes popular books."

**When personalization beats popularity:**

| User Count | Expected Outcome |
|------------|------------------|
| <5,000 | Popularity often wins - insufficient data |
| 5,000-20,000 | Models start competing with popularity |
| 20,000+ | Personalization clearly wins |

### What BPR *Does* Improve

Despite not beating popularity for book recommendations, BPR provides value:

1. **2x better than old model**: 8.22% → 15.20% is meaningful progress
2. **Foundation for scale**: As more users are collected, BPR will outperform popularity

### Friend-Finding Quality Evaluation

We measured whether BPR produces better friend matches using **IDF-weighted rating agreement**:

**What is IDF-weighted agreement?**

IDF (Inverse Document Frequency) weights rare books higher than popular ones:
- Harry Potter (45% of users read it): IDF ≈ 1 (low weight)
- Obscure sci-fi (0.5% of users): IDF ≈ 5 (high weight)

For each friend pair, we compute:
```
Agreement Score = Σ IDF(book) for books where both users agree (both liked or both disliked)
Total Score = Σ IDF(book) for all books both users rated
Agreement Ratio = Agreement Score / Total Score
```

This rewards matches who agree on rare books, not just bestsellers.

**Results (2025-01-20):**

| Method | IDF-Weighted Agreement | vs Random |
|--------|----------------------|-----------|
| Random | 90.7% | - |
| BPR | 93.9% | 1.03x |
| Pointwise | 96.9% | 1.07x |

**Surprising finding**: Pointwise produces slightly better friend matches than BPR.

**Why agreement is high for everyone (~90%+):**

With 95% positive ratings (class imbalance), any two random users agree most of the time. If user A likes 95% of books and user B likes 95% of books, they'll agree on ~90% of shared books by chance.

**Why Pointwise beats BPR for friend-finding:**

- Pointwise directly optimizes P(user likes book), which is exactly what agreement measures
- BPR optimizes ranking (user prefers A over B), not binary like/dislike prediction
- BPR's advantage is in *ranking* recommendations, not in *classification*

**Implications:**

1. For **book recommendations**: Use BPR (better ranking, 2x precision)
2. For **friend-finding**: Pointwise may be slightly better, but the difference is small (3%)
3. The real bottleneck is **class imbalance** - with 95% likes, all methods perform similarly

**Evaluation script:** `evaluate_friend_quality.py`

### IDF-Weighted Training Experiment

**Hypothesis:** If we weight rare books higher during training, the model should learn more from "both liked obscure sci-fi" than "both liked Harry Potter", producing better friend matches.

**Implementation:**
```python
# Standard loss (all books equal):
loss = (prediction - actual)²

# IDF-weighted loss (rare books matter more):
loss = IDF(book) × (prediction - actual)²
```

Where IDF(book) = log(max_readers / book_readers) + 1

**Results (2025-01-20):**

| Method | Agreement | vs Base |
|--------|-----------|---------|
| Random | 90.7% | - |
| **Pointwise** | **96.9%** | baseline |
| Pointwise + IDF | 96.2% | -0.7% |
| BPR | 93.9% | baseline |
| BPR + IDF | 93.6% | -0.3% |

**Surprising finding:** IDF weighting during training slightly HURTS friend-finding quality.

**Why IDF weighting doesn't help:**

1. **Popular books are still valid signal**: Two users who both read Harry Potter and both liked it DO share some taste - it's not pure noise.

2. **Rare books have less reliable signal**: With fewer ratings, rare book preferences may be noisier (one person's 5-star obscure book is another's DNF).

3. **Agreement metric already uses IDF**: We measure friend quality with IDF-weighted agreement, so the evaluation already downweights popular book matches. Adding IDF to training is double-counting.

4. **Class imbalance dominates**: With 95% positive ratings, the high baseline agreement (~90%) swamps any IDF effect.

**Conclusion:** Keep training weights uniform. Use IDF only for display (sorting shared books by rarity).

**Evaluation script:** `evaluate_idf_training.py`

### Predictive Evaluation (The Right Metric)

Previous metrics measured "agreement on books both users read" - but with 95% positive ratings, this gives a ~90% baseline that's hard to beat.

**The right question:** "If I hold out a book that user A liked, do A's friend matches predict higher scores for that book than random users?"

This measures the **predictive value** of friend matching - whether friends can help you discover books you'd actually like.

**Method:**
1. Hold out 20% of each user's liked books
2. Train model on remaining 80%
3. Get each user's top-10 friend matches
4. Compare: friends' predicted scores for held-out books vs random users' scores

**Results (2025-01-20):**

| Metric | Pointwise | BPR |
|--------|-----------|-----|
| Self predicted score | 0.725 | 0.815 |
| Friend predicted score | 0.667 | 0.758 |
| Random predicted score | 0.589 | 0.641 |
| **Friend/Random score ratio** | **1.13x** | **1.18x** |
| Friend hit rate | 6.70% | 8.47% |
| Random hit rate | 2.76% | 2.75% |
| **Friend/Random hit rate** | **2.42x** | **3.07x** |

**Key findings:**

1. **✓ Friend matches DO predict held-out books better than random**
   - Friends predict 13-18% higher scores than random users
   - This is statistically meaningful - friend matching has predictive value

2. **✓ Friends share actual reading patterns**
   - Friends are 2.4-3x more likely to have actually READ the held-out books
   - This isn't just predicted taste - friends genuinely read similar books

3. **BPR is better for friend-finding**
   - BPR: 3.07x hit rate, 1.18x predicted score
   - Pointwise: 2.42x hit rate, 1.13x predicted score
   - BPR captures ranking preferences that transfer to friend similarity

**Why this metric is better:**

| Old Metric (Agreement) | New Metric (Prediction) |
|------------------------|-------------------------|
| Measures past overlap | Measures future prediction |
| ~90% baseline (class imbalance) | Meaningful baseline |
| Can't beat "everyone agrees" | Shows real predictive value |
| Descriptive | Predictive |

**Conclusion:** The friend-finding system works. Friends predict your taste 13-18% better than random, and are 3x more likely to have read books you'd like. BPR is the best model.

**Evaluation script:** `evaluate_predictive.py`

### Training Time Trade-off

- Pointwise: 2.6s
- BPR: 122.9s (slower due to negative sampling)

The extra training time is worth the 2x improvement in recommendation quality.

### Implementation

- `precompute_all.py` - Now uses BPR by default (`USE_BPR = True`). Set to `False` for old pointwise loss.
- `evaluate_bpr.py` - Comparison script showing BPR vs Pointwise results.

Both training methods are preserved in `precompute_all.py` - just toggle `USE_BPR`.

### References

- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) - Rendle et al. 2009
- [Negative Sampling in Recommendation: A Survey](https://arxiv.org/html/2409.07237v1) - 2024 survey

---

## The Sparse Matrix Problem

### The Fundamental Challenge

Recommendation systems face a deceptively simple problem: predict what users will like. But the data we have to work with is almost entirely empty.

Consider our dataset:
- **~10,000 users**
- **~125,000 books** (after filtering)
- **~1.5 million interactions**

At first glance, 1.5 million interactions sounds like a lot. But the full user-book matrix has 10,000 × 125,000 = **1.25 billion cells**. We have data for only **0.12%** of possible user-book pairs. The remaining 99.88% are unknown.

```
User-Book Matrix (simplified view):

              Book1  Book2  Book3  Book4  Book5  ...  Book36155
User1           5      -      -      4      -    ...     -
User2           -      3      -      -      -    ...     -
User3           4      -      5      -      -    ...     2
...            ...    ...    ...    ...    ...   ...    ...
User2682        -      -      -      -      3    ...     -

"-" = unknown (99.6% of cells)
```

This sparsity creates three interrelated problems:

1. **Cold start**: New users have zero interactions. New books have zero readers. How do you recommend anything?

2. **Weak signal**: Even active users have rated only ~140 books on average. That's 0.4% of the catalog. Two users might both love obscure sci-fi, but if they've read different obscure sci-fi books, we can't detect their similarity.

3. **Class imbalance**: Of the interactions we DO have, 95% are positive (user liked the book). Users don't bother recording books they abandoned. This makes "dislike" nearly invisible in the data.

### Naive Strategy 1: Recommend Everything

The simplest "recommendation" system: show users all books they haven't read.

```python
def recommend_all(user):
    return [book for book in all_books if book not in user.read_books]
```

**Precision**: Technically undefined. If we count any book the user eventually likes as a "hit," this achieves 100% recall - every book they'd like is in the list.

**Why it's useless**:
- No ranking. The user sees 35,000+ books with no guidance.
- No personalization. Every user gets the same overwhelming list.
- The "recommendation" provides zero information gain over browsing randomly.

This is the degenerate case that proves a recommendation system must do more than enumerate possibilities.

### Naive Strategy 2: Recommend Popular Books

A smarter baseline: recommend whatever most people liked.

```python
def recommend_popular(user, k=10):
    popular = sorted(all_books, key=lambda b: b.reader_count, reverse=True)
    return [b for b in popular if b not in user.read_books][:k]
```

Our top 10 most popular books:
1. 1984 (45% of users read it)
2. Harry Potter and the Sorcerer's Stone (45%)
3. Project Hail Mary (40%)
4. The Hunger Games (37%)
5. Dune (37%)
6. The Hobbit (35%)
7. Harry Potter and the Chamber of Secrets (35%)
8. Animal Farm (35%)
9. Harry Potter and the Prisoner of Azkaban (35%)
10. Harry Potter and the Goblet of Fire (34%)

**Precision@10: ~38%**

This is shockingly good. Recommend Harry Potter and 1984 to everyone, and you'll be right about 4 out of 10 times. Why?

1. **Popular books are popular for a reason**. They're genuinely good, broadly appealing, and culturally significant.

2. **Survivorship bias in popularity**. Books that 45% of users have read are books that appeal to diverse tastes - they're "safe" recommendations.

3. **Base rate dominates with sparse data**. When you know almost nothing about a user, the prior probability (popularity) is your best guess.

**Why it's still inadequate**:

1. **No personalization**. A romance reader and a hard sci-fi fan both get recommended 1984. One will love it; the other might not.

2. **Filter bubble of the mainstream**. Users will never discover niche books they'd love. The system reinforces what's already popular.

3. **Diminishing returns**. After a user reads the top 50 popular books, what then? The system has nothing left to offer.

4. **No community**. "You should read Harry Potter because everyone read it" is not a meaningful connection. You can't build friendships on universally shared experiences.

### Why Collaborative Filtering is Better

Our approach - matrix factorization with BPR loss - learns latent features that capture taste profiles. Instead of "recommend what's popular," we ask "recommend what similar people liked."

**BPR Precision@10: 15.2%**

Wait - that's *lower* than popularity's 38%. Have we made things worse?

No. The metrics measure different things:

| Metric | Popularity | BPR | What it measures |
|--------|------------|-----|------------------|
| Precision@10 | 38% | 15.2% | Will user like a recommended book? |
| Personalization | 0% | High | Do different users get different recommendations? |
| Discovery | 0% | High | Does user find books they wouldn't have found otherwise? |
| Niche coverage | 0% | High | Can system recommend obscure books? |

**The precision paradox**: Popularity wins on precision *because* it only recommends safe bets. It's a high-precision, zero-discovery system. Our system trades some precision for personalization and discovery.

### The Real Value: Friend-Finding

Here's where our approach definitively wins. Recommendation systems don't just recommend items - they can recommend *people*.

Our predictive evaluation shows:

| Metric | Random Users | Matched Friends | Improvement |
|--------|--------------|-----------------|-------------|
| Predicted score for held-out books | 0.641 | 0.758 | **+18%** |
| Actually read held-out books | 2.75% | 8.47% | **3.07x** |

Friends matched by our BPR embeddings:
- Predict your taste 18% better than random users
- Are 3x more likely to have read books you'll eventually like
- Share meaningful niche interests (weighted by IDF)

**Popularity can't do this**. You can't find "book friends" by looking at who else read Harry Potter - that's everyone. The signal is in the obscure shared interests, and finding those requires learning latent taste dimensions.

### When Personalization Beats Popularity

The crossover point depends on data density:

| User Count | Data Density | Winner |
|------------|--------------|--------|
| <5,000 | Very sparse | Popularity often wins |
| 5,000-20,000 | Sparse | Models start competing |
| 20,000-100,000 | Moderate | Personalization wins |
| 100,000+ | Dense | Personalization dominates |

Netflix, Spotify, and Amazon operate in the "dense" regime with millions of users. At that scale, personalization crushes popularity. With 10K users (Feb 2026), we're now in the "sparse" regime where personalization starts competing with popularity.

But even now, our approach provides value that popularity cannot:
1. **Friend discovery** (3x hit rate)
2. **Niche recommendations** (books with <100 readers)
3. **Taste profiling** (user embeddings capture reading preferences)
4. **Scalable foundation** (same algorithm works better with more data)

### Strategies for Sparse Data

Given the sparsity challenge, here's what works:

**1. BPR (Bayesian Personalized Ranking)** ✓ Implemented

Treats unobserved items as negatives and optimizes ranking rather than classification. This sidesteps the "95% positive" class imbalance by asking "does user prefer A over B?" rather than "does user like A?"

**2. Negative Sampling** ✓ Implemented

Sample random unobserved items as negatives during training. We use 4 negatives per positive, which balances training signal against computational cost.

**3. Regularization** ✓ Implemented

L2 regularization (λ=1.0) prevents overfitting to the sparse observed data. Without it, the model memorizes training examples rather than learning generalizable patterns.

**4. Implicit Feedback Weighting** ✓ Implemented

"Want to read" (0.3) is weaker signal than "Read" (1.0). This extracts more information from the sparse interactions we do have.

**5. IDF Weighting for Display** ✓ Implemented

Rare shared books are more meaningful than common ones. We use IDF to surface obscure connections.

**6. Content-Based Hybrid** ○ Future Work

When collaborative data is sparse, augment with book metadata (genre, author, tags). This helps with cold-start items.

**7. Collect More Data** ○ Ongoing

The fundamental fix for sparsity is more data. Target: 10K+ users to reliably beat the popularity baseline.

### Conclusion

The sparse matrix problem is why recommendation systems are hard. Naive approaches either provide no value (recommend everything) or provide generic value (recommend popular).

Collaborative filtering with BPR provides *personalized* value by learning latent taste dimensions from sparse interactions. It doesn't beat popularity on raw precision yet - we need more users for that. But it already provides something popularity cannot: meaningful connections between readers who share obscure interests.

The friend-matching results prove the approach works. Friends predict held-out books 18% better than random and are 3x more likely to have read them. That's the value of learning from sparse data rather than falling back to popularity.

---

## Summary: Friend-Finding Quality

### Current Dataset (Feb 2026)

| Metric | Value |
|--------|-------|
| Total users | 10,023 |
| Users after filtering | 9,998 |
| Total books | 924,269 |
| Books after filtering | 125,051 |
| Book-user relationships | 4,672,476 |
| Training interactions | 1,533,675 |

### How We Measure Quality

**Predictive evaluation**: Hold out 20% of each user's liked books, train on the rest, then check if matched friends predict those held-out books better than random users.

**Key metric - Hit Rate Ratio**: Are friends more likely to have *actually read* held-out books compared to random users? A ratio of 5x means friends are 5x more likely to share your taste.

### Progress Over Time

| Configuration | Hit Rate Ratio | vs Random |
|---------------|----------------|-----------|
| Random baseline | 1.00x | - |
| Pointwise loss (original) | 2.42x | +142% |
| BPR loss | 3.07x | +207% |
| **BPR + 10% popularity filter** | **5.50x** | **+450%** |

### Current Best Configuration

```python
USE_BPR = True                    # Pairwise ranking loss
MAX_BOOK_POPULARITY_PCT = 0.10    # Drop books read by >10% of users
NUM_FEATURES = 30                 # Latent dimensions (tuned)
# No clustering - compare all users against all users
```

**Result**: Matched friends are 5.5x more likely to have read books you'd enjoy compared to random users.

### What Each Improvement Contributed

1. **BPR loss** (+27% over Pointwise): Better handles class imbalance by optimizing ranking rather than classification
2. **Popularity filter** (+79% over BPR alone): Forces model to learn from distinctive reading patterns, not mainstream popularity
3. **Hyperparameter tuning** (+3% fine-tuning): Optimal feature dimensions for current data size
4. **Cluster removal** (Jan 2025): Global matching instead of cluster-based ensures no good matches are missed at boundaries

---

## Future Improvement Ideas

1. ~~**Integrate BPR into precompute_all.py**~~: Done - BPR is now the default
2. ~~**Popular book filtering**~~: Done - Drop books read by >10% of users (+79% improvement)
3. ~~**Hyperparameter tuning**~~: Done - NUM_FEATURES=30 (+3% improvement)
4. ~~**Cross-cluster matching**~~: Done - Removed clustering entirely, now compare all users
5. ~~**Collect more users**~~: Done - Now at 10K+ users (Feb 2026). Target 20K+ for next milestone.
6. **Hybrid popularity-personalization**: Blend scores: `α * popularity + (1-α) * BPR_score`
7. **Temporal decay**: Weight recent reads higher than old reads
8. **Content-based hybrid**: Incorporate book genres/tags for cold-start users
9. **Hard negative mining**: Sample negatives similar to positives for stronger training signal

---

## Output Files

After running `precompute_all.py`:

| File | Contents | Size |
|------|----------|------|
| `webapp/data/recommendations.json` | Per-user: stats, top 10 matches with shared books | ~98MB |
| `webapp/data/users.json` | List of all users (for dropdown selection) | ~600KB |
