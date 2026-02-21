import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

def _finalize_one_conversation(cid, df, MIN_LEN_GLOBAL, MAX_LEN_GLOBAL):
    conv_df = df[df["conversation_id"] == cid].copy()
    receipt_rows = conv_df[conv_df["payment_marker"] == "receipt found"]
    if not receipt_rows.empty:
        first_receipt_idx = receipt_rows.index[0]
        cut_position = conv_df.index.get_loc(first_receipt_idx)
        conv_df = conv_df.iloc[:cut_position + 1]
        status = "closing"
        cut_index = int(first_receipt_idx)
    else:
        status = "failed"
        cut_index = None

    conv_df = conv_df[conv_df["role"].str.lower() != "media"].copy()
    conv_df = conv_df.reset_index(drop=True)

    merged = []
    if not conv_df.empty:
        prev_role = conv_df.loc[0, "role"]
        merged_chat = str(conv_df.loc[0, "chat"])

        for i in range(1, len(conv_df)):
            role_now = conv_df.loc[i, "role"]
            chat_now = str(conv_df.loc[i, "chat"])

            if role_now == prev_role:
                merged_chat += " " + chat_now
            else:
                merged.append({"role": prev_role, "chat": merged_chat})
                prev_role = role_now
                merged_chat = chat_now
        merged.append({"role": prev_role, "chat": merged_chat})

    conv_clean = pd.DataFrame(merged)
    conv_clean["conversation_id"] = str(cid)
    conv_clean["status"] = status

    raw_length = len(conv_clean)
    priority = raw_length if status == "closing" else 0

    result = {
        "status": status,
        "cut_index": cut_index,
        "raw_length": raw_length,
        "priority": priority
    }
    return str(cid), result, conv_clean

def finalize_conversation_df_parallel(df: pd.DataFrame, max_workers=8):
    required_cols = ["conversation_id", "role", "chat"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' wajib ada di dataframe backend. Hilang: {col}")

    if "payment_marker" not in df.columns:
        raise ValueError("Kolom 'payment_marker' belum ada.")

    df = df.copy()
    df["payment_marker"] = (
        df["payment_marker"].fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    conversation_ids = df["conversation_id"].dropna().unique().tolist()

    MAX_LEN_GLOBAL = 80
    MIN_LEN_GLOBAL = 5

    results = {}
    processed_conversations = []

    workers = min(max_workers, len(conversation_ids))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _finalize_one_conversation,
                cid, df, MIN_LEN_GLOBAL, MAX_LEN_GLOBAL
            )
            for cid in conversation_ids
        ]

        for fut in as_completed(futures):
            cid, res, conv_clean = fut.result()
            results[cid] = res
            processed_conversations.append(conv_clean)

    for cid, res in results.items():
        if res["status"] == "closing":
            raw_len = res["raw_length"]
            if raw_len <= MIN_LEN_GLOBAL:
                res["priority"] = 100
            elif raw_len >= MAX_LEN_GLOBAL:
                res["priority"] = 1
            else:
                res["priority"] = max(
                    1,
                    int(100 - ((raw_len - MIN_LEN_GLOBAL) /
                               (MAX_LEN_GLOBAL - MIN_LEN_GLOBAL)) * 99)
                )
        else:
            res["priority"] = 0
        print(f"Conv {cid}: raw_length={res['raw_length']}, min_global={MIN_LEN_GLOBAL}, max_global={MAX_LEN_GLOBAL}, priority={res['priority']}")

    final_df = pd.concat(processed_conversations, ignore_index=True)

    for cid, res in results.items():
        final_df.loc[
            final_df["conversation_id"] == cid,
            "priority"
        ] = res["priority"]

    return results, final_df
