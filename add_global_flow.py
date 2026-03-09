import sqlite3
import os

ANALYSIS_DB_PATH = "output/analysis.db"
GLOBAL_FLOW_DB_PATH = "output/global_flow.db"


def get_db(path):
    return sqlite3.connect(path)


def add_to_flow_texts():
    if not os.path.exists(ANALYSIS_DB_PATH):
        print("[ERROR] analysis.db not found")
        return

    if not os.path.exists(GLOBAL_FLOW_DB_PATH):
        print("[ERROR] global_flow.db not found")
        return

    analysis_db = get_db(ANALYSIS_DB_PATH)
    global_db = get_db(GLOBAL_FLOW_DB_PATH)

    analysis_cur = analysis_db.cursor()
    global_cur = global_db.cursor()

    analysis_cur.execute("""
        SELECT 
            id,
            user_message,
            best_user_node_id,
            assistant_node_id,
            optional_llm_output
        FROM chat_analysis
    """)

    rows = analysis_cur.fetchall()
    inserted_count = 0

    print("\n=== START ADDING TO GLOBAL FLOW (ONLY OPTIONAL LLM PAIRS) ===\n")

    for row in rows:
        (
            row_id,
            user_message,
            best_user_node_id,
            assistant_node_id,
            optional_llm_output
        ) = row

        # ======================================================
        # ONLY PROCESS IF OPTIONAL LLM OUTPUT EXISTS
        # ======================================================
        if not (assistant_node_id and optional_llm_output and best_user_node_id and user_message):
            print(f"[SKIPPED] analysis_id={row_id} (optional_llm_output missing)")
            continue

        user_message_clean = user_message.strip()
        llm_output_clean = optional_llm_output.strip()

        if not user_message_clean or not llm_output_clean:
            print(f"[SKIPPED] analysis_id={row_id} (empty text after strip)")
            continue

        # ==========================
        # INSERT USER MESSAGE
        # ==========================
        global_cur.execute("""
            SELECT 1 FROM flow_texts
            WHERE node_id = ? AND chat = ?
        """, (best_user_node_id, user_message_clean))

        if not global_cur.fetchone():
            global_cur.execute("""
                INSERT INTO flow_texts (node_id, chat, priority)
                VALUES (?, ?, 10)
            """, (
                best_user_node_id,
                user_message_clean
            ))

            print(f"[ADDED][USER] analysis_id={row_id}")
            print(f"  Node: {best_user_node_id}")
            print(f"  → {user_message_clean[:120]}")
            print("-" * 60)

            inserted_count += 1
        else:
            print(f"[EXISTS][USER] analysis_id={row_id} Node={best_user_node_id}")

        # ==========================
        # INSERT OPTIONAL LLM OUTPUT
        # ==========================
        global_cur.execute("""
            SELECT 1 FROM flow_texts
            WHERE node_id = ? AND chat = ?
        """, (assistant_node_id, llm_output_clean))

        if not global_cur.fetchone():
            global_cur.execute("""
                INSERT INTO flow_texts (node_id, chat, priority)
                VALUES (?, ?, 10)
            """, (
                assistant_node_id,
                llm_output_clean
            ))

            print(f"[ADDED][ASSISTANT] analysis_id={row_id}")
            print(f"  Node: {assistant_node_id}")
            print(f"  → {llm_output_clean[:120]}")
            print("-" * 60)

            inserted_count += 1
        else:
            print(f"[EXISTS][ASSISTANT] analysis_id={row_id} Node={assistant_node_id}")

    global_db.commit()

    analysis_db.close()
    global_db.close()

    print("\n=== DONE ===")
    print(f"[SUCCESS] Inserted {inserted_count} new flow_texts rows.\n")


if __name__ == "__main__":
    add_to_flow_texts()