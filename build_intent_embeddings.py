import os
import json
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY tidak ditemukan di environment variable")

def build_intent_embeddings_once(
    flow_db_path="output/global_flow.db",
    intent_embedding_db_path="output/intent_embeddings.db",
    embedding_model="text-embedding-3-large",
    batch_size=100
):
    client = OpenAI(api_key=openai_api_key)

    # ==============================
    # LOAD FLOW NODES
    # ==============================
    if not os.path.exists(flow_db_path):
        raise FileNotFoundError(f"{flow_db_path} tidak ditemukan.")

    conn = sqlite3.connect(flow_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT node_id, intent FROM flow_nodes")
    flow_nodes = cursor.fetchall()
    conn.close()

    print(f"Total nodes ditemukan: {len(flow_nodes)}")

    # ==============================
    # PREPARE INTENTS
    # ==============================
    node_ids = []
    intents = []

    for node_id, intent in flow_nodes:
        if not intent:
            continue

        node_ids.append(node_id)
        intents.append(intent)

    if not intents:
        print("Tidak ada intent untuk di-embed.")
        return

    print(f"Total intent yang akan di-embed: {len(intents)}")

    # ==============================
    # CREATE / REPLACE DB
    # ==============================
    os.makedirs(os.path.dirname(intent_embedding_db_path), exist_ok=True)

    conn = sqlite3.connect(intent_embedding_db_path)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS intent_embeddings")

    cursor.execute("""
    CREATE TABLE intent_embeddings (
        node_id TEXT PRIMARY KEY,
        intent TEXT,
        embedding TEXT
    )
    """)

    conn.commit()

    # ==============================
    # BATCH EMBEDDING
    # ==============================
    for i in range(0, len(intents), batch_size):
        batch_intents = intents[i:i + batch_size]
        batch_node_ids = node_ids[i:i + batch_size]

        response = client.embeddings.create(
            model=embedding_model,
            input=batch_intents
        )

        for node_id, intent_text, emb_data in zip(batch_node_ids, batch_intents, response.data):
            cursor.execute(
                "INSERT INTO intent_embeddings (node_id, intent, embedding) VALUES (?, ?, ?)",
                (node_id, intent_text, json.dumps(emb_data.embedding))
            )

        conn.commit()
        print(f"Batch {i // batch_size + 1} selesai")

    conn.close()

    print("Selesai. Semua intent embeddings sudah dibuat.")

if __name__ == "__main__":
    build_intent_embeddings_once()