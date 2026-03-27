# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Book recommendation system using the Hardcover API (hardcover.app). Collects user/book data, builds recommendations using matrix factorization (TensorFlow), and provides a Flask web app for finding "book friends" with similar reading tastes.

## Quick Start Commands

```bash
# Setup
mkdir -p ~/data/hardcover
pip install requests python-dotenv tensorflow numpy scikit-learn flask gunicorn

# Data collection (25 users per batch, ~30 sec each due to rate limiting)
python3 scripts/process_batch.py

# Download multiple batches
for i in {1..20}; do echo "=== Batch $i/20 ==="; python3 scripts/process_batch.py; sleep 15; done

# Train model and generate webapp data
python3 precompute_all.py

# Run webapp locally
cd webapp && python3 app.py  # http://localhost:8001

# Deploy to Cloud Run
cd webapp && gcloud run deploy book-friend-finder --source . --region us-central1 --allow-unauthenticated
```

## Architecture

### Data Flow

```
Hardcover GraphQL API
         ↓
scripts/process_batch.py (25 users/batch, 1s delay between API calls)
         ↓
~/data/hardcover/
├── users.json         (user profiles)
├── user_books.json    (users → books, ~100MB+)
└── books_users.json   (books → users inverted index, ~60MB+)
         ↓
precompute_all.py (TensorFlow BPR matrix factorization)
         ↓
webapp/data/
├── recommendations.json (pre-computed friend matches)
├── users.json            (filtered active users)
└── chat.db              (SQLite message storage)
         ↓
webapp/app.py (Flask, no ML deps at runtime)
```

### Key Design Decisions

- **Offline ML**: All TensorFlow training happens locally via `precompute_all.py`. The webapp only serves pre-computed JSON - no ML libraries needed in production.
- **Atomic batch processing**: `scripts/process_batch.py` saves progress only after all 4 phases complete (fetch users → fetch books → update inverted index → stats). Safe to interrupt mid-batch.
- **Chat access control**: Users can only message their top 10 matches or people who messaged them first (enforced in `webapp/app.py:chat_conversation`, lines 150-153).
- **SQLite chat storage**: `webapp/chat_db.py` stores messages in `webapp/data/chat.db`.
- **Progress tracking**: `progress.json` in repo root tracks batches processed. After forced shutdowns, verify all files are consistent (users.json, user_books.json, books_users.json counts should match). If corrupted, `scripts/invert_to_books.py` can rebuild books_users.json from user_books.json.

### Recommendation Algorithm

**BPR (Bayesian Personalized Ranking)** is the default training method. It treats recommendation as a ranking problem rather than classification, which handles the severe class imbalance (95% positive ratings) better than pointwise loss. BPR doubles precision compared to the old pointwise method.

Implicit feedback weights:
- Read with rating: rating/5.0 (e.g., 5-star = 1.0, 3-star = 0.6)
- Read without rating: 0.7
- Currently reading: 0.7
- Want to read: 0.3
- Did not finish: 0.0

Friend matching uses cosine similarity on L2-normalized user feature vectors against ALL users (no clustering). See IMPLEMENTATION.md for the full algorithm details and evaluation results.

**Tunable constants in `precompute_all.py`:**
- `USE_BPR = True` - Use BPR loss (default, 2x better precision). Set False for old pointwise loss.
- `MIN_RATINGS_PER_USER = 20`, `MIN_USERS_PER_BOOK = 5` (data filtering)
- `MAX_BOOK_POPULARITY_PCT = 0.10` - Drop books read by >10% of users. **Improves hit rate ratio by 97%**
- `NUM_FEATURES = 30` (latent dimensions, tuned)
- `LAMBDA = 1.0` (L2 regularization)
- `ITERATIONS = 100`, `LEARNING_RATE = 0.1`
- `NEG_SAMPLES_PER_POS = 4` (BPR negative sampling ratio)

**Evaluation scripts:**
- `evaluate_recommendations.py` - Original pointwise evaluation (Precision@K, Recall@K, NDCG)
- `evaluate_bpr.py` - BPR vs Pointwise book recommendation comparison. **BPR doubles precision**
- `evaluate_friend_quality.py` - Friend-finding quality using IDF-weighted agreement
- `evaluate_idf_training.py` - Tests IDF-weighted training. **Does not help**
- `evaluate_predictive.py` - Predictive friend quality: do friends predict held-out books? **BPR is best (3x hit rate)**
- `evaluate_popularity_filter.py` - Tests popular book filtering. **10% threshold doubles hit rate ratio**
- `evaluate_hyperparams.py` - Tests NUM_FEATURES. **30 features optimal**

See IMPLEMENTATION.md for full results and analysis.

## API Configuration

Create `.env`:
```
HARDCOVER_API_TOKEN=your_token_here
```
Get token: https://hardcover.app/account/api

**Book status IDs:** 1=Want to read, 2=Currently reading, 3=Read, 5=Did not finish

## Webapp Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | User selection |
| `/find_friends` | POST | Get friend recommendations |
| `/chat` | GET | Chat inbox |
| `/chat/<other_id>` | GET/POST | Conversation (top 10 matches only) |
| `/chat/<other_id>/messages` | GET | Poll for new messages (JSON) |
| `/chat/unread_count` | GET | Unread count (JSON) |

## Production

**URL:** https://book-friend-finder-i2nrrpteiq-uc.a.run.app

See `DEPLOYMENT.md` for full Cloud Run setup guide.

