import json
import sqlite3
import os

OUTPUT_DIR = "output"

def connect(db_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return sqlite3.connect(os.path.join(OUTPUT_DIR, db_name))

# =========================
# 1. EMBEDDINGS
# =========================
def convert_embeddings():
    with open(f"{OUTPUT_DIR}/embeddings1.json", encoding="utf-8") as f:
        data = json.load(f)

    conn = connect("embeddings.db")
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS embeddings")
    cur.execute("""
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding TEXT
        )
    """)

    for text, emb in data.get("text_embeddings", {}).items():
        cur.execute(
            "INSERT INTO embeddings (text, embedding) VALUES (?, ?)",
            (text, json.dumps(emb))
        )

    conn.commit()
    conn.close()
    print("File output embeddings.db telah dibuat.")

# =========================
# 2. GLOBAL FLOW
# =========================
def convert_global_flow():
    with open(f"{OUTPUT_DIR}/global_flow1.json", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})

    conn = connect("global_flow.db")
    cur = conn.cursor()

    # DROP
    cur.executescript("""
        DROP TABLE IF EXISTS flow_nodes;
        DROP TABLE IF EXISTS flow_texts;
        DROP TABLE IF EXISTS flow_answers;
    """)

    # CREATE
    cur.execute("""
        CREATE TABLE flow_nodes (
            node_id TEXT PRIMARY KEY,
            intent TEXT,
            category TEXT,
            role TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE flow_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            chat TEXT,
            priority INTEGER,
            FOREIGN KEY(node_id) REFERENCES flow_nodes(node_id)
        )
    """)

    cur.execute("""
        CREATE TABLE flow_answers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_node_id TEXT,
            intent TEXT,
            to_node_id TEXT,
            FOREIGN KEY(from_node_id) REFERENCES flow_nodes(node_id)
        )
    """)

    # INSERT
    for node_id, node in nodes.items():
        cur.execute(
            "INSERT INTO flow_nodes VALUES (?, ?, ?, ?)",
            (node_id, node["intent"], node["category"], node["role"])
        )

        # texts
        for t in node.get("texts", []):
            cur.execute("""
                INSERT INTO flow_texts (node_id, chat, priority)
                VALUES (?, ?, ?)
            """, (node_id, t.get("chat"), t.get("priority", 0)))

        # answers (EDGE)
        for intent, targets in node.get("answers", {}).items():
            for trg in targets:
                cur.execute("""
                    INSERT INTO flow_answers
                    (from_node_id, intent, to_node_id)
                    VALUES (?, ?, ?)
                """, (node_id, intent, trg.get("to")))

    conn.commit()
    conn.close()
    print("File output global_flow.db telah dibuat.")

# =========================
# 3. CONVERSATION IDS
# =========================
def convert_conv_ids():
    with open(f"{OUTPUT_DIR}/processed_conv_ids.json") as f:
        ids = json.load(f)

    conn = connect("processed_conv_ids.db")
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS processed_conv_ids")
    cur.execute("""
        CREATE TABLE processed_conv_ids (
            conv_id INTEGER PRIMARY KEY
        )
    """)

    for cid in ids:
        cur.execute(
            "INSERT OR IGNORE INTO processed_conv_ids VALUES (?)",
            (cid,)
        )

    conn.commit()
    conn.close()
    print("File output processed_conv_ids.db telah dibuat.")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    convert_embeddings()
    convert_global_flow()
    convert_conv_ids()
    print("\n SEMUA JSON → SQLITE (RELATIONAL) SELESAI DIBUAT")
