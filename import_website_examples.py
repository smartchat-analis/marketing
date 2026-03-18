import csv
import os
import sqlite3

DB_PATH = "output/website_examples.db"
CSV_PATH = "output/website_examples.csv"

def get_db(path):
    return sqlite3.connect(path, check_same_thread=False)

def init_db():
    os.makedirs("output", exist_ok=True)
    db = get_db(DB_PATH)
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS website_examples (
            link TEXT NOT NULL,
            company TEXT NOT NULL,
            jenis_usaha TEXT NOT NULL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_we_company ON website_examples(company)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_we_jenis ON website_examples(jenis_usaha)")
    db.commit()
    db.close()

def import_csv():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    db = get_db(DB_PATH)
    cur = db.cursor()
    cur.execute("DELETE FROM website_examples")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = 0
        for row in reader:
            link = (row.get("link") or "").strip()
            company = (row.get("company") or "").strip().upper()
            jenis_usaha = (row.get("jenis_usaha") or "").strip()
            if not link or not company or not jenis_usaha:
                continue
            cur.execute(
                "INSERT INTO website_examples (link, company, jenis_usaha) VALUES (?, ?, ?)",
                (link, company, jenis_usaha)
            )
            rows += 1

    db.commit()
    db.close()
    return rows

if __name__ == "__main__":
    init_db()
    count = import_csv()
    print(f"[OK] Imported {count} rows into {DB_PATH}")
