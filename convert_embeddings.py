import ijson
import pickle

node_embeddings = {}

with open("output/embeddings1.json", "rb") as f:
    parser = ijson.parse(f)

    current_node = None
    for prefix, event, value in parser:
        # Deteksi key node_embeddings.<node_id>.item
        if prefix.startswith("node_embeddings.") and event == "number":
            parts = prefix.split(".")
            node_id = parts[1]

            if node_id not in node_embeddings:
                node_embeddings[node_id] = []

            node_embeddings[node_id].append(value)

with open("output/node_emb.pkl", "wb") as f:
    pickle.dump(node_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ node_emb.pkl berhasil dibuat (streaming, no MemoryError)")