# Book Friend Finder

A book recommendation system that helps readers find their "book friends" - people with similar reading tastes. Uses the [Hardcover](https://hardcover.app) API for data and BPR matrix factorization (TensorFlow) for collaborative filtering.

**Live demo:** https://book-friend-finder-954510692982.us-central1.run.app

## Features

- **Friend Matching**: Find readers with similar taste using ML-based collaborative filtering
- **Shared Books**: See exactly which books you and your matches both loved (sorted by rarity)
- **Conversation Starters**: Books they've read that are on your "want to read" list
- **Chat**: Message your top 10 matches directly

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Hardcover GraphQL API                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌─────────────────┐     Rate limited: 60 req/min                      │
│   │ process_batch.py │     25 users per batch                           │
│   └────────┬────────┘                                                   │
│            │                                                             │
│            ▼                                                             │
│   ~/data/hardcover/                                                      │
│   ├── users.json          (user profiles, ~4MB)                         │
│   └── books_users.db      (SQLite: books → users, ~1GB)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ML TRAINING (Offline)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │ precompute_all.py │                                                 │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│   1. Build user-book interaction matrix                                  │
│   2. Filter out popular books (>10% of users) to focus on niche taste   │
│   3. Train BPR matrix factorization model (TensorFlow)                  │
│   4. Compute all-pairs cosine similarity                                │
│   5. Pre-compute top 10 matches for each user                           │
│            │                                                             │
│            ▼                                                             │
│   webapp/data/                                                           │
│   ├── recommendations.json  (pre-computed friend matches, ~98MB)        │
│   └── users.json            (filtered active users)                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WEB APP (Runtime)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Flask app - NO ML dependencies at runtime                             │
│   Just serves pre-computed JSON files                                   │
│                                                                          │
│   Routes:                                                                │
│   /                    User selection                                   │
│   /find_friends        Get friend matches                               │
│   /chat                Messaging inbox                                  │
│   /chat/<id>           Conversation with a match                        │
│                                                                          │
│   Deployed on Google Cloud Run (serverless)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Algorithm

#### 1. Implicit Feedback Matrix

Unlike Netflix ratings, most book tracking is implicit. We convert reading statuses to preference scores:

| Status | Score | Rationale |
|--------|-------|-----------|
| Read with rating | rating/5.0 | 5-star = 1.0, 3-star = 0.6 |
| Read (unrated) | 0.7 | Completed = probably liked |
| Currently reading | 0.7 | Actively engaged |
| Want to read | 0.3 | Weak positive signal |
| Did not finish | 0.0 | Negative signal |

#### 2. Popular Book Filtering

Books read by >10% of users (Harry Potter, 1984, etc.) are dropped before training. This forces the model to learn from distinctive reading patterns rather than mainstream popularity. This nearly doubles match quality.

#### 3. BPR Matrix Factorization

We use Bayesian Personalized Ranking (BPR) which treats recommendation as a ranking problem:

```
Loss = -log(sigmoid(score(user, liked_book) - score(user, random_book)))
```

This handles the severe class imbalance (95% positive ratings) better than classification approaches.

#### 4. IDF-Weighted Similarity

Shared books are weighted by rarity using inverse document frequency:

```
weight(book) = log(max_popularity / book_popularity) + 1
```

Sharing a rare book means more than sharing a bestseller.

### Why This Works

Traditional book recommendations answer "what should I read next?" We answer a different question: **"who should I read with?"**

Matched friends are **5.5x more likely** to have read books you'd enjoy compared to random users.

## Quick Start

### Prerequisites

```bash
mkdir -p ~/data/hardcover
pip install requests python-dotenv tensorflow numpy scikit-learn flask gunicorn

# Get API token from https://hardcover.app/account/api
echo "HARDCOVER_API_TOKEN=your_token_here" > .env
```

### Collect Data

```bash
# Fetch one batch (25 users, ~30 sec due to rate limiting)
python3 scripts/process_batch.py

# Fetch multiple batches
for i in {1..20}; do
    echo "=== Batch $i/20 ==="
    python3 scripts/process_batch.py
    sleep 15
done
```

### Train Model

```bash
python3 precompute_all.py
```

### Run Locally

```bash
cd webapp && python3 app.py
# Visit http://localhost:8001
```

### Deploy

```bash
cd webapp && gcloud run deploy book-friend-finder \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

## Project Structure

```
book-friend-finder/
├── precompute_all.py         # ML training pipeline
├── IMPLEMENTATION.md         # Algorithm details and evaluation results
├── DEPLOYMENT.md             # Cloud Run deployment guide
├── scripts/
│   ├── process_batch.py      # Data collection (25 users/batch)
│   ├── invert_to_books.py    # Rebuild inverted index from user_books
│   └── migrate_to_sqlite.py  # Migrate JSON data to SQLite
├── evaluation/               # Model evaluation scripts
│   ├── evaluate_bpr.py       # BPR vs Pointwise comparison
│   ├── evaluate_predictive.py    # Predictive friend quality
│   ├── evaluate_popularity_filter.py  # Popular book filtering
│   └── ...                   # 5 more evaluation scripts
├── presentations/            # LaTeX slides (.tex + .pdf)
├── webapp/
│   ├── app.py                # Flask application
│   ├── chat_db.py            # SQLite chat storage
│   ├── templates/
│   │   ├── index.html        # User selection page
│   │   ├── results.html      # Friend match results
│   │   ├── chat_inbox.html   # Messaging inbox
│   │   └── chat_conversation.html  # Chat with a match
│   └── data/
│       └── users.json        # Filtered active users (~750KB)
└── ~/data/hardcover/         # Raw data (local only, not deployed)
    ├── users.json            # User profiles (~4MB)
    └── books_users.db        # SQLite: books → users (~1GB)
```

## Model Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NUM_FEATURES` | 30 | Latent dimensions |
| `MAX_BOOK_POPULARITY_PCT` | 0.10 | Drop books read by >10% of users |
| `LAMBDA` | 1.0 | L2 regularization |
| `ITERATIONS` | 100 | Training epochs |
| `LEARNING_RATE` | 0.1 | Adam optimizer |
| `NEG_SAMPLES_PER_POS` | 4 | BPR negative sampling ratio |
| `MIN_RATINGS_PER_USER` | 20 | Filter inactive users |
| `MIN_USERS_PER_BOOK` | 5 | Filter rare books |

## Current Dataset (Feb 2026)

| Metric | Value |
|--------|-------|
| Total users | ~10,000 |
| Total books | ~925,000 |
| Books after filtering | ~125,000 |
| Interactions | ~4.7 million |
| Matrix sparsity | 99.88% (the fundamental challenge) |

## Data Sources

All data comes from [Hardcover](https://hardcover.app), an indie book tracking platform. We use their public GraphQL API with appropriate rate limiting (60 requests/minute). Data is stored locally in SQLite (`books_users.db`, ~1GB).

**Book status codes:** 1 = Want to read, 2 = Currently reading, 3 = Read, 5 = Did not finish

## Limitations

- **No authentication**: Users select themselves from a dropdown; no login system yet (OAuth with Hardcover would solve this)
- **Cold start**: New users need ~20 books tracked for good matches
- **No content features**: Pure collaborative filtering; doesn't use genres/descriptions
- **Static recommendations**: Recompute needed when data updates significantly
- **Scale ceiling**: All-pairs similarity is O(n²); may need approximation above ~50K users

## License

Personal project - not licensed for redistribution.

## Acknowledgments

- [Hardcover](https://hardcover.app) for the excellent API and book data
- [BPR paper](https://arxiv.org/abs/1205.2618) by Rendle et al. for the ranking approach
