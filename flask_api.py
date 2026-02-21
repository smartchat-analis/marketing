from clean_data import clean_data_df
from build_bubble import build_bubble_df_parallel
from mark_payment import mark_payment_df_parallel
from finalize_conversation import finalize_conversation_df_parallel
from label_and_build_global_flow import label_and_build_global_flow_parallel

from flask import Flask, request, jsonify
import pandas as pd
import requests
from openai import OpenAI
from collections import OrderedDict
import traceback
import os
import json
import threading
import sqlite3

client = OpenAI(
    api_key="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"
)
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Flask API Running OK — use POST /run-pipeline to process data"
    })

SKIP_CONVERSATION_IDS = {
    4, 5, 6, 8, 13, 14, 165, 390, 786, 1110, 3074, 
    3075, 3098, 3100, 3101, 3102, 3103, 3104, 3105, 
    3106, 3107, 3108, 3110, 3145, 3146, 3147, 3148, 
    3149, 3150, 3151, 3729, 3737, 3738, 3739, 3740, 
    3741, 4559, 5382, 5383, 5384, 5385, 5386, 5387, 
    5388, 5391, 5393, 5394, 5395, 5396, 5398, 5399, 
    5400, 5407, 5412, 5413, 5414, 5415, 6370, 8984, 
    8985, 10860, 10862, 11122, 11299, 11333, 11334, 
    11911, 11912, 11913, 11914, 11915, 11917, 11918, 
    11920, 11921, 11922, 11923, 11925, 11926, 11927, 
    11928, 11929, 11930, 11931, 11933, 11934, 11935, 
    11936, 12049, 12479, 12983, 13223, 13226, 13233 
}

ALLOWED_USER_IDS = {35, 46, 47, 48}

def load_from_smartchat(conversation_ids):
    url = "https://smartchat2.edakarya.com/api/get-chats-from-id"
    params = [("conversation_ids[]", cid) for cid in conversation_ids]
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise ValueError(resp.text)
    return resp.json()

def get_db(path):
    return sqlite3.connect(path, check_same_thread=False)

def load_processed_ids():
    os.makedirs("output", exist_ok=True)
    db = get_db("output/processed_conv_ids.db")
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS processed_conv_ids (conv_id INTEGER PRIMARY KEY)")
    cur.execute("SELECT conv_id FROM processed_conv_ids")
    ids = {row[0] for row in cur.fetchall()}
    db.close()
    print(f"[LOG] Loaded processed IDs: {len(ids)}")
    return ids

def save_processed_ids(processed_ids):
    os.makedirs("output", exist_ok=True)
    db = get_db("output/processed_conv_ids.db")
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS processed_conv_ids (conv_id INTEGER PRIMARY KEY)")
    for cid in processed_ids:
        cur.execute(
            "INSERT OR IGNORE INTO processed_conv_ids (conv_id) VALUES (?)",
            (cid,)
        )
    db.commit()
    db.close()
    print(f"[LOG] Saved processed IDs: {len(processed_ids)}")

def deduplicate_texts(texts):
    """Deduplicate texts based on 'chat' content."""
    seen = set()
    unique_texts = []
    for text in texts:
        chat = text.get('chat', '')
        if chat not in seen:
            seen.add(chat)
            unique_texts.append(text)
    return unique_texts

def load_global_flow():
    db = get_db("output/global_flow.db")
    cur = db.cursor()
    # ======================
    # LOAD NODES
    # ======================
    cur.execute("""
        SELECT node_id, intent, category, role
        FROM flow_nodes
    """)
    nodes = {
        row[0]: {
            "intent": row[1],
            "category": row[2],
            "role": row[3],
            "texts": [],
            "answers": {}
        }
        for row in cur.fetchall()
    }

    # ======================
    # LOAD TEXTS
    # ======================
    cur.execute("""
        SELECT node_id, chat, priority
        FROM flow_texts
        ORDER BY priority ASC
    """)
    for node_id, chat, priority in cur.fetchall():
        if node_id in nodes:
            nodes[node_id]["texts"].append({
                "chat": chat,
                "priority": priority
            })

    # ======================
    # LOAD ANSWERS
    # ======================
    cur.execute("""
        SELECT from_node_id, intent, to_node_id
        FROM flow_answers
    """)
    for from_node, intent, to_node in cur.fetchall():
        if from_node in nodes:
            nodes[from_node]["answers"].setdefault(intent, []).append({
                "to": to_node
            })

    db.close()
    return nodes

# ============================
# RUN-PIPELINE ENDPOINT
# ============================
@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    print("[LOG] Starting pipeline...")
    body = request.json
    if not body:
        return jsonify({"error": "Body JSON required"}), 400

    conv_ids = body.get("conversation_ids")
    if not conv_ids:
        return jsonify({"error": "conversation_ids is required"}), 400

    conv_ids = [cid for cid in conv_ids if cid not in SKIP_CONVERSATION_IDS]
    print(f"[LOG] Conv IDs after skip: {conv_ids}")

    processed_ids = load_processed_ids()
    print(f"[LOG] Current processed IDs: {processed_ids}")

    already_processed = [cid for cid in conv_ids if cid in processed_ids]
    if already_processed:
        return jsonify({
            "status": "skipped",
            "message": f"conversation_id {already_processed} sudah pernah diproses"
        }), 200

    unprocessed_ids = [cid for cid in conv_ids if cid not in processed_ids]
    if not unprocessed_ids:
        # Load from SQLite instead of JSON
        flow = load_global_flow()
        if flow:
            return jsonify(flow)
        return jsonify({"error": "No cached result found"}), 404

    print(f"[LOG] Processing {len(unprocessed_ids)} conversations")

    try:
        raw_data = load_from_smartchat(unprocessed_ids)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb.splitlines()[-5:]}), 500

    valid_conv_ids = set()
    all_rows = []

    for conv_id_str, bubbles in raw_data.items():
        if not bubbles:
            continue

        first_bubble = bubbles[0]
        conversation = first_bubble.get("conversation", {})

        owner_id = conversation.get("user_id")
        if owner_id in ALLOWED_USER_IDS:
            conv_id = int(conv_id_str)
            valid_conv_ids.add(conv_id)
            all_rows.extend(bubbles)

    if not valid_conv_ids:
        return jsonify({
            "status": "skipped",
            "message": "Tidak ada conversation milik user_id yang diizinkan",
            "allowed_user_ids": list(ALLOWED_USER_IDS)
        }), 200

    print(f"[LOG] Valid conversations: {sorted(valid_conv_ids)}")
    print(f"[LOG] Total rows after ownership filter: {len(all_rows)}")

    df_raw = pd.DataFrame(all_rows)
    try:
        print("[LOG] Step 2: Cleaning data...")
        grouped_clean = clean_data_df(df_raw)

        print("[LOG] Step 3: Building bubbles...")
        grouped_bubble = build_bubble_df_parallel(grouped_clean, max_workers=8)
        df_bubble = pd.concat(grouped_bubble.values(), ignore_index=True)

        print("[LOG] Step 4: Marking payments...")
        df_marked, payment_results = mark_payment_df_parallel(df_bubble, client, max_workers=3)

        print("[LOG] Step 5: Finalizing conversations...")
        summary_results, df_final = finalize_conversation_df_parallel(df_marked, max_workers=8)

        print("[LOG] Step 6: Labeling and building flow...")
        os.makedirs("output", exist_ok=True)
        new_nodes, df_intent = label_and_build_global_flow_parallel(
            df_final,
            client,
            load_existing=True,
            existing_flow_path="output/global_flow.db",
            existing_embedding_path="output/embeddings.db",
            max_workers=8
        )

        print("[LOG] Step 7: Ordering nodes...")
        ordered_new_nodes = OrderedDict()
        for key in sorted(new_nodes.keys(), key=lambda x: int(x[1:])):
            node = new_nodes[key]
            ordered_node = OrderedDict([
                ('intent', node['intent']),
                ('category', node['category']),
                ('role', node['role']),
                ('texts', deduplicate_texts(node['texts'])),
                ('answers', node['answers'])
            ])
            ordered_new_nodes[key] = ordered_node

        save_processed_ids(valid_conv_ids)

        print("[LOG] Pipeline completed successfully.")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[ERROR] Pipeline failed: {e}\n{tb}")
        return jsonify({"error": f"Pipeline error: {str(e)}", "traceback": tb.splitlines()[-5:]}), 500
    return jsonify(ordered_new_nodes)

# ==========================================================================================================
# ==========================================================================================================

from response_claude import chat_with_session, load_flow_and_embeddings

load_flow_and_embeddings() 

# ==========================
# CHAT ENDPOINT
# ==========================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    session_id = data.get("session_id")
    reset = data.get("reset", False)

    if not message:
        return jsonify({"error": "message required"}), 400
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    result = chat_with_session(user_message=message, session_id=session_id, reset=reset)

    if not result or not result.get("response"):
        return jsonify({"error": "Failed to generate response"}), 500

    return jsonify({
        "response": result["response"],
        "node_id": result.get("node_id"),
        "debug": result.get("debug")
    })

# ======================
# RUN
# ======================
if __name__ == "__main__":
    print("Running Flask API on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)