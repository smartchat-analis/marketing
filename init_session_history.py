import sqlite3
import os

DB_PATH = "output/analysis.db"


def init_session_history_table(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            chat TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id
        ON session_history(session_id);
    """)

    conn.commit()


def migrate_from_chat_analysis(conn):
    cur = conn.cursor()

    # Ambil semua data lama
    cur.execute("""
        SELECT session_id, timestamp, user_message, response
        FROM chat_analysis
        ORDER BY id ASC
    """)

    rows = cur.fetchall()

    inserted_count = 0

    for session_id, timestamp, user_message, response in rows:

        # Insert user message
        if user_message:
            cur.execute("""
                INSERT INTO session_history (session_id, role, chat, created_at)
                VALUES (?, 'user', ?, ?)
            """, (session_id, user_message, timestamp))
            inserted_count += 1

        # Insert assistant response
        if response:
            cur.execute("""
                INSERT INTO session_history (session_id, role, chat, created_at)
                VALUES (?, 'assistant', ?, ?)
            """, (session_id, response, timestamp))
            inserted_count += 1

    conn.commit()
    print(f"Migrated {inserted_count} messages into session_history.")


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    print("Initializing session_history table...")
    init_session_history_table(conn)

    print("Migrating data from chat_analysis...")
    migrate_from_chat_analysis(conn)

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()