import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_one_conversation(item):
    conv_id, df = item
    merged_rows = []
    current_role = None
    current_text = []

    for _, row in df.iterrows():
        chat_text = str(row["chat"]).strip()
        role = row["role"]

        if role != current_role:
            if current_text:
                merged_rows.append({
                    "conversation_id": row["conversation_id"],
                    "role": current_role,
                    "chat": " ".join(current_text).strip()
                })
            current_role = role
            current_text = [chat_text]
        else:
            current_text.append(chat_text)

    if current_text:
        merged_rows.append({
            "conversation_id": df.iloc[-1]["conversation_id"],
            "role": current_role,
            "chat": " ".join(current_text).strip()
        })

    merged_df = pd.DataFrame(merged_rows)
    return int(conv_id), merged_df


def build_bubble_df_parallel(grouped_clean: dict, max_workers=8):
    grouped_bubble = {}
    items = list(grouped_clean.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one_conversation, item)
            for item in items
        ]
        for fut in as_completed(futures):
            conv_id, merged = fut.result()
            grouped_bubble[conv_id] = merged
    return grouped_bubble