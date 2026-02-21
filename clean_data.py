import pandas as pd
def clean_data_df(df_raw: pd.DataFrame):
    required_cols = ["created_at", "conversation_id", "role", "chat", "nilai"]
    df = df_raw.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["conversation_id"] = pd.to_numeric(df["conversation_id"], errors="coerce")
    df = df[df["role"] != "system"]
    df = df.sort_values(by=["conversation_id", "created_at"]).reset_index(drop=True)
    df = df[~((df["role"] == "media") & (df["nilai"] == 1))]
    allowed_ext = (".jpg", ".jpeg", ".png", ".webp", ".gif")
    df["chat"] = df["chat"].fillna("").astype(str)
    df = df[~(
        (df["role"] == "media") &
        (~df["chat"].str.lower().str.endswith(allowed_ext))
    )]
    df = df[~df["chat"].str.strip().isin(["", "nan", "nan nan"])]
    df = df.reset_index(drop=True)
    grouped = {
        int(cid): group.reset_index(drop=True)
        for cid, group in df.groupby("conversation_id")
    }
    return grouped