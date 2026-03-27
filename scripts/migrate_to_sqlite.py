#!/usr/bin/env python3
"""
Migrate books_users.json to SQLite database.
"""

import json
import sqlite3
import os
from datetime import datetime, UTC

JSON_FILE = os.path.expanduser("~/data/hardcover/books_users.json")
DB_FILE = os.path.expanduser("~/data/hardcover/books_users.db")


def create_schema(conn):
    """Create the database schema."""
    cursor = conn.cursor()

    # Books table - stores book metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            slug TEXT,
            image_url TEXT,
            cached_contributors TEXT
        )
    """)

    # Book-user relationships
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

    # Index for fast user lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_book_users_user
        ON book_users(user_id)
    """)

    # Index for fast book lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_book_users_book
        ON book_users(book_id)
    """)

    # Metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()


def migrate_data(conn, json_file):
    """Migrate data from JSON to SQLite."""
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    books_list = data.get("books", [])
    metadata = data.get("metadata", {})

    print(f"Found {len(books_list)} books to migrate")

    cursor = conn.cursor()

    # Insert metadata
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("migrated_at", datetime.now(UTC).isoformat())
    )
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("total_books", str(metadata.get("total_books", len(books_list))))
    )
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("total_users", str(metadata.get("total_users", 0)))
    )

    # Process books in batches for better performance
    batch_size = 1000
    total_users_inserted = 0

    for i, book_entry in enumerate(books_list):
        book = book_entry.get("book", {})
        users = book_entry.get("users", [])

        book_id = book.get("id")
        if not book_id:
            continue

        # Insert book
        image = book.get("image")
        image_url = image.get("url") if image else None

        cursor.execute("""
            INSERT OR REPLACE INTO books (id, title, slug, image_url, cached_contributors)
            VALUES (?, ?, ?, ?, ?)
        """, (
            book_id,
            book.get("title"),
            book.get("slug"),
            image_url,
            json.dumps(book.get("cached_contributors")) if book.get("cached_contributors") else None
        ))

        # Insert book-user relationships
        for user in users:
            cursor.execute("""
                INSERT OR REPLACE INTO book_users
                (book_id, user_id, username, name, status, status_id, rating, review_raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                book_id,
                user.get("user_id"),
                user.get("username"),
                user.get("name"),
                user.get("status"),
                user.get("status_id"),
                user.get("rating"),
                user.get("review_raw")
            ))
            total_users_inserted += 1

        # Progress update
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(books_list)} books...")
            conn.commit()

    conn.commit()
    print(f"Migration complete!")
    print(f"  - Books: {len(books_list)}")
    print(f"  - Book-user relationships: {total_users_inserted}")


def verify_migration(conn):
    """Verify the migration was successful."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM books")
    book_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM book_users")
    relationship_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM book_users")
    unique_users = cursor.fetchone()[0]

    print(f"\nDatabase verification:")
    print(f"  - Books: {book_count}")
    print(f"  - Book-user relationships: {relationship_count}")
    print(f"  - Unique users: {unique_users}")


def main():
    if os.path.exists(DB_FILE):
        print(f"Database already exists: {DB_FILE}")
        response = input("Delete and recreate? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
        os.remove(DB_FILE)

    print(f"Creating database: {DB_FILE}")
    conn = sqlite3.connect(DB_FILE)

    try:
        create_schema(conn)
        migrate_data(conn, JSON_FILE)
        verify_migration(conn)
    finally:
        conn.close()

    # Show file sizes
    json_size = os.path.getsize(JSON_FILE) / (1024 * 1024 * 1024)
    db_size = os.path.getsize(DB_FILE) / (1024 * 1024 * 1024)
    print(f"\nFile sizes:")
    print(f"  - JSON: {json_size:.2f} GB")
    print(f"  - SQLite: {db_size:.2f} GB")


if __name__ == "__main__":
    main()
