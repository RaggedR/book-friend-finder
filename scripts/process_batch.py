#!/usr/bin/env python3
"""
Hardcover API - Batch Processor
Coordinates all 4 phases of processing a batch of 25 users:
1. Download 25 users
2. Get books for those 25 users
3. Create inverted JSON (books -> users)
4. Print statistics
Only updates progress after ALL 4 phases complete.
"""

import json
import os
import sys
import time
import sqlite3
from datetime import datetime, UTC
from collections import defaultdict
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_ENDPOINT = "https://api.hardcover.app/v1/graphql"
API_TOKEN = os.getenv("HARDCOVER_API_TOKEN", "YOUR_API_TOKEN_HERE")
USERS_FILE = os.path.expanduser("~/data/hardcover/users.json")
BOOKS_USERS_DB = os.path.expanduser("~/data/hardcover/books_users.db")
PROGRESS_FILE = os.path.expanduser("~/git/hardcover-live/progress.json")
BATCH_SIZE = 25
REQUEST_DELAY = 1.0


def load_progress():
    """Load progress file."""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            progress = json.load(f)
            # Ensure new fields exist
            if 'seen_user_ids' not in progress:
                progress['seen_user_ids'] = []
            if 'cursor_created_before' not in progress:
                progress['cursor_created_before'] = None
            # Convert to set for O(1) lookup
            progress['seen_user_ids_set'] = set(progress['seen_user_ids'])
            return progress
    except FileNotFoundError:
        return {
            "batches_processed": 0,
            "total_users": 0,
            "total_books": 0,
            "last_updated": None,
            "seen_user_ids": [],
            "seen_user_ids_set": set(),
            "cursor_created_before": None
        }


def save_progress(progress):
    """Save progress file."""
    progress["last_updated"] = datetime.now(UTC).isoformat()
    # Remove the set before saving (not JSON serializable)
    save_data = {k: v for k, v in progress.items() if k != 'seen_user_ids_set'}
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(save_data, f, indent=2)


def load_json_file(filepath, default):
    """Load JSON file or return default."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def save_json_file(filepath, data):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================================
# PHASE 1: Download 25 users
# ============================================================================

MIN_BOOKS_FILTER = 20  # Only fetch detailed books for users with at least this many


def fetch_users(cursor_created_before=None):
    """Fetch 25 users from API using cursor-based pagination.

    Args:
        cursor_created_before: ISO timestamp. Fetch users created before this time.
                              If None, fetches the newest users.

    Only fetches users who have at least 1 book (skips empty accounts).
    Returns book count so we can filter for MIN_BOOKS_FILTER in Phase 2.
    """
    if cursor_created_before:
        # Cursor-based: get users created before the cursor, who have at least 1 book
        query = """
        query GetUsers($limit: Int!, $cursor: timestamptz!) {
          users(
            limit: $limit,
            where: {
              created_at: {_lt: $cursor},
              user_books: {}
            },
            order_by: {created_at: desc}
          ) {
            id
            name
            username
            bio
            image { url }
            created_at
            updated_at
            user_books_aggregate {
              aggregate { count }
            }
          }
        }
        """
        variables = {"limit": BATCH_SIZE, "cursor": cursor_created_before}
    else:
        # No cursor: get newest users who have at least 1 book
        query = """
        query GetUsers($limit: Int!) {
          users(
            limit: $limit,
            where: {
              user_books: {}
            },
            order_by: {created_at: desc}
          ) {
            id
            name
            username
            bio
            image { url }
            created_at
            updated_at
            user_books_aggregate {
              aggregate { count }
            }
          }
        }
        """
        variables = {"limit": BATCH_SIZE}

    payload = {
        "query": query,
        "variables": variables
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")

    return data.get("data", {}).get("users", [])


# ============================================================================
# PHASE 2: Get books for each user
# ============================================================================

def fetch_user_books(user_id):
    """Fetch books for a specific user."""
    query = """
    query GetUserBooks($user_id: Int!) {
      user_books(where: {user_id: {_eq: $user_id}}) {
        id
        status_id
        rating
        review_raw
        book {
          id
          title
          slug
          image { url }
          cached_contributors
        }
      }
    }
    """

    payload = {
        "query": query,
        "variables": {"user_id": user_id}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()

    if "errors" in data:
        return []

    user_books = data.get("data", {}).get("user_books", [])

    # Organize by status
    result = {
        "want_to_read": [],
        "currently_reading": [],
        "read": [],
        "other": []
    }

    for entry in user_books:
        status_id = entry.get("status_id")
        if status_id == 1:
            result["want_to_read"].append(entry)
        elif status_id == 2:
            result["currently_reading"].append(entry)
        elif status_id == 3:
            result["read"].append(entry)
        else:
            result["other"].append(entry)

    return result


# ============================================================================
# PHASE 3: Update SQLite database (books -> users) - INCREMENTAL
# ============================================================================

def get_db_connection():
    """Get a connection to the SQLite database, creating schema if needed."""
    conn = sqlite3.connect(BOOKS_USERS_DB)

    # Create schema if it doesn't exist
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            slug TEXT,
            image_url TEXT,
            cached_contributors TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS book_users (
            book_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT,
            name TEXT,
            status TEXT,
            status_id INTEGER,
            rating REAL,
            review_raw TEXT,
            PRIMARY KEY (book_id, user_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_book_users_user ON book_users(user_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_book_users_book ON book_users(book_id)
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()

    return conn


def update_books_users_db(conn, new_user_books_data):
    """
    Update SQLite database with NEW users only (incremental).

    Args:
        conn: SQLite connection
        new_user_books_data: Only the NEW users from this batch

    Returns:
        Tuple of (total_books, total_relationships)
    """
    cursor = conn.cursor()

    STATUS_NAMES = {
        1: "want_to_read",
        2: "currently_reading",
        3: "read",
        5: "did_not_finish"
    }

    # Process ONLY the new users
    for user_data in new_user_books_data:
        user_info = user_data["user"]
        books = user_data.get("books")

        if not books:
            continue

        for status_id, status_name in STATUS_NAMES.items():
            status_books = books.get(status_name, [])

            for book_entry in status_books:
                book = book_entry.get("book")
                if not book:
                    continue

                book_id = book["id"]

                # Insert or update book
                image = book.get("image")
                image_url = image.get("url") if image else None
                cached_contrib = book.get("cached_contributors")

                cursor.execute("""
                    INSERT OR REPLACE INTO books (id, title, slug, image_url, cached_contributors)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    book_id,
                    book.get("title"),
                    book.get("slug"),
                    image_url,
                    json.dumps(cached_contrib) if cached_contrib else None
                ))

                # Insert book-user relationship
                cursor.execute("""
                    INSERT OR REPLACE INTO book_users
                    (book_id, user_id, username, name, status, status_id, rating, review_raw)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    book_id,
                    user_info["id"],
                    user_info["username"],
                    user_info["name"],
                    status_name,
                    status_id,
                    book_entry.get("rating"),
                    book_entry.get("review_raw")
                ))

    conn.commit()

    # Get counts
    cursor.execute("SELECT COUNT(*) FROM books")
    total_books = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM book_users")
    total_relationships = cursor.fetchone()[0]

    return total_books, total_relationships


# ============================================================================
# PHASE 4: Calculate and print statistics
# ============================================================================

def print_statistics(conn):
    """Print required statistics using SQLite database."""
    cursor = conn.cursor()

    # a) Percentage of books with more than 5 users
    cursor.execute("SELECT COUNT(*) FROM books")
    total_books = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM (
            SELECT book_id FROM book_users GROUP BY book_id HAVING COUNT(*) > 5
        )
    """)
    books_with_multiple_users = cursor.fetchone()[0]

    if total_books > 0:
        percentage = (books_with_multiple_users / total_books) * 100
    else:
        percentage = 0

    # b) Average number of books per user
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM book_users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM book_users")
    total_book_entries = cursor.fetchone()[0]

    if total_users > 0:
        avg_books = total_book_entries / total_users
    else:
        avg_books = 0

    print(f"\n{'='*60}")
    print(f"STATISTICS")
    print(f"{'='*60}")
    print(f"a) Books with >5 users: {percentage:.1f}% ({books_with_multiple_users}/{total_books})")
    print(f"b) Average books per user: {avg_books:.1f}")
    print(f"c) Number of users processed: {total_users}")
    print(f"d) Number of books: {total_books}")
    print(f"{'='*60}\n")


# ============================================================================
# MAIN COORDINATOR
# ============================================================================

def main():
    """Main coordinator for all 4 phases."""

    if API_TOKEN == "YOUR_API_TOKEN_HERE":
        print("ERROR: Set HARDCOVER_API_TOKEN in .env file!")
        sys.exit(1)

    # Load progress
    progress = load_progress()
    batch_num = progress["batches_processed"] + 1
    cursor = progress.get("cursor_created_before")
    seen_ids = progress.get("seen_user_ids_set", set())

    print(f"\n{'='*60}")
    print(f"PROCESSING BATCH #{batch_num}")
    print(f"{'='*60}")
    print(f"Batches already processed: {progress['batches_processed']}")
    print(f"Total unique users so far: {len(seen_ids)}")
    if cursor:
        print(f"Cursor: fetching users created before {cursor}")
    else:
        print(f"Cursor: None (fetching newest users)")
    print(f"{'='*60}\n")

    # ========================================================================
    # PHASE 1: Download 25 users (with duplicate detection)
    # ========================================================================
    print(f"PHASE 1: Downloading users (cursor-based)...")
    try:
        fetched_users = fetch_users(cursor)
        if not fetched_users:
            print("No more users available!")
            sys.exit(0)

        # Filter out duplicates
        new_users = [u for u in fetched_users if u['id'] not in seen_ids]
        duplicates_found = len(fetched_users) - len(new_users)

        if duplicates_found > 0:
            print(f"  Fetched {len(fetched_users)} users, {duplicates_found} were duplicates")

        if not new_users:
            print("All fetched users are duplicates! Updating cursor and retrying...")
            # Update cursor to oldest fetched user to skip past duplicates
            oldest = min(fetched_users, key=lambda u: u.get('created_at', ''))
            progress["cursor_created_before"] = oldest.get('created_at')
            save_progress(progress)
            print(f"Cursor updated to: {progress['cursor_created_before']}")
            print("Run the script again to continue.")
            sys.exit(0)

        # Filter by book count - only keep users with >= MIN_BOOKS_FILTER books
        def get_book_count(user):
            agg = user.get('user_books_aggregate', {})
            return agg.get('aggregate', {}).get('count', 0)

        users_with_enough_books = [u for u in new_users if get_book_count(u) >= MIN_BOOKS_FILTER]
        skipped_low_books = len(new_users) - len(users_with_enough_books)

        if skipped_low_books > 0:
            print(f"  Skipped {skipped_low_books} users with < {MIN_BOOKS_FILTER} books")

        print(f"✓ Downloaded {len(new_users)} new users ({len(users_with_enough_books)} with {MIN_BOOKS_FILTER}+ books)\n")
    except Exception as e:
        print(f"✗ Failed: {e}")
        sys.exit(1)

    # ========================================================================
    # PHASE 2: Get books for each user (only for users with enough books)
    # ========================================================================
    new_user_books = []

    if not users_with_enough_books:
        print(f"PHASE 2: No users with {MIN_BOOKS_FILTER}+ books to process\n")
    else:
        print(f"PHASE 2: Fetching books for {len(users_with_enough_books)} users...")

        for idx, user in enumerate(users_with_enough_books, 1):
            print(f"  [{idx}/{len(users_with_enough_books)}] {user['name']} (@{user['username']})...", end=" ")

            books = fetch_user_books(user['id'])
            counts = {
                "read": len(books.get("read", [])),
                "currently_reading": len(books.get("currently_reading", [])),
                "want_to_read": len(books.get("want_to_read", [])),
                "other": len(books.get("other", []))
            }
            total = sum(counts.values())

            print(f"{total} books ({counts['read']} read)")

            # Only save users with >= MIN_BOOKS_FILTER READ books
            if counts["read"] >= MIN_BOOKS_FILTER:
                new_user_books.append({
                    "user": {
                        "id": user["id"],
                        "name": user["name"],
                        "username": user["username"]
                    },
                    "books": books,
                    "counts": counts
                })
            else:
                print(f"    ^ Skipped (only {counts['read']} read)")

            if idx < len(users_with_enough_books):
                time.sleep(REQUEST_DELAY)

        print(f"✓ Fetched books for {len(users_with_enough_books)} users, saved {len(new_user_books)} with {MIN_BOOKS_FILTER}+ read\n")

    # Load and update users.json (only users with 20+ READ books)
    saved_user_ids = {u["user"]["id"] for u in new_user_books}
    users_to_save = [u for u in users_with_enough_books if u["id"] in saved_user_ids]
    users_data = load_json_file(USERS_FILE, {"metadata": {}, "users": []})
    users_data["users"].extend(users_to_save)
    users_data["metadata"]["count"] = len(users_data["users"])
    save_json_file(USERS_FILE, users_data)

    # ========================================================================
    # PHASE 3: Update SQLite database (incremental)
    # ========================================================================
    print(f"PHASE 3: Updating books->users SQLite database (incremental)...")

    # Open database connection
    conn = get_db_connection()

    # Update with ONLY the new users
    total_books, total_relationships = update_books_users_db(conn, new_user_books)

    print(f"✓ Updated books_users.db with {total_books} books\n")

    # ========================================================================
    # PHASE 4: Print statistics
    # ========================================================================
    print(f"PHASE 4: Calculating statistics...")
    print_statistics(conn)

    # Close database connection
    conn.close()

    # ========================================================================
    # UPDATE PROGRESS - Only after all 4 phases complete
    # ========================================================================
    # Add new user IDs to seen set
    new_user_ids = [u['id'] for u in new_users]
    progress["seen_user_ids"].extend(new_user_ids)
    progress["seen_user_ids_set"].update(new_user_ids)

    # Update cursor to oldest user in this batch (for next batch)
    oldest_user = min(new_users, key=lambda u: u.get('created_at', ''))
    progress["cursor_created_before"] = oldest_user.get('created_at')

    progress["batches_processed"] = batch_num
    progress["total_users"] = len(progress["seen_user_ids"])
    progress["total_books"] = total_books
    save_progress(progress)

    print(f"✓ Progress updated: Batch #{batch_num} complete")
    print(f"  - Scanned {len(new_user_ids)} users, {len(users_with_enough_books)} had {MIN_BOOKS_FILTER}+ total books")
    print(f"  - Saved {len(new_user_books)} users with {MIN_BOOKS_FILTER}+ READ books")
    print(f"  - Total users scanned: {progress['total_users']}")
    print(f"  - Total active users saved: {len(users_data['users'])}")
    print(f"  - Next cursor: {progress['cursor_created_before']}")
    print(f"\nRun this script again to process the next batch.\n")


if __name__ == "__main__":
    main()
