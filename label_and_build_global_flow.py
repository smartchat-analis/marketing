import json
import os
import re
import pandas as pd
from collections import OrderedDict
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3

# ==================
# FLOW CATEGORIES
# ==================
FLOW_CATEGORIES = [
    "greeting_value_proposition",
    "segmentasi_kualifikasi_prospek",
    "edukasi_produk",
    "penawaran_paket",
    "call_to_action",
    "conversational_closing",
    "follow_up"
]

# ==================
# COSINE SIMILARITY
# ==================
def cosine_similarity(vecA, vecB):
    if not vecA or not vecB:
        return 0.0
    dot = sum(a * b for a, b in zip(vecA, vecB))
    magA = sum(a * a for a in vecA) ** 0.5
    magB = sum(b * b for b in vecB) ** 0.5
    if magA == 0 or magB == 0:
        return 0.0
    return dot / (magA * magB)

# ==================
# LABEL & BUILD FLOW
# ==================
def label_and_build_global_flow_parallel(
    df: pd.DataFrame,
    client,
    embeddings=None,
    text_col="chat",
    model_name="gpt-4.1",
    temperature=0.1,
    save_path="output/global_flow.db",
    embedding_save_path="output/embeddings.db",
    max_nodes=None,
    include_roles=("user", "assistant"),
    verbose=True,
    load_existing=True,
    existing_flow_path="output/global_flow.db",
    existing_embedding_path="output/embeddings.db",
    save_only_new_nodes=False,
    max_workers=8
):
    # ================
    # LABELING INTENT
    # ================
    def call_label(target_text, target_role, context_messages=None):
        """
        target_text      : pesan TERAKHIR (wajib)
        target_role      : role pesan terakhir
        context_messages : list[str] -> maksimal 3 chat sebelumnya (optional)
        """

        context_block = ""
        if context_messages:
            context_block = "\n".join(context_messages)

        prompt = f"""
        Kamu adalah AI classifier profesional yang menentukan INTENT dari percakapan antara klien dan tim marketing dari salah satu perusahaan berikut:

        1. PT EbyB
        2. PT Asa Inovasi Software (Asain)
        3. PT Eksa Digital Agency (EDA)

        INTENT harus:
        - sangat spesifik
        - menggambarkan tujuan utama pesan
        - minimal 3 kata, maksimal 10 kata
        - menyebut nama perusahaan hanya jika:
            a) perusahaan tersebut disebut dalam pesan, atau
            b) konteksnya tidak bisa dipahami tanpa menyebut perusahaan.
        - jika perusahaan tidak relevan atau tidak disebut, tidak perlu dimasukkan ke intent.

        =======================================================================
        DEFINISI INTENT:
        =======================================================================
        Intent = tujuan spesifik dari pesan terakhir, ditentukan dari:
        1) pesan terakhir
        2) membaca maksimal 3 chat sebelumnya untuk konteks
        3) konteks layanan (website, SEO, google ads, sosmed ads, domain, video promosi, dp, follow-up, negosiasi, dll)
        4) apakah pesan menyebut perusahaan tertentu

        Intent TIDAK BOLEH generik seperti:
        - “tanya harga”
        - “promo”
        - “revisi”
        - “info”
        - “lainnya”
        - “pemesanan”

        Intent harus menjelaskan:
        - aksi + tujuan
        - konteks jelas
        - perusahaan hanya bila relevan

        =======================================================================
        CONTOH INTENT (BENAR):
        =======================================================================
        ✓ "menawarkan layanan digital agency Asain dan menunggu pertanyaan klien"
        ✓ "menanyakan progres pembuatan website EDA tahap awal"
        ✓ "mengirimkan protofolio website dari PT EbyB untuk referensi klien"
        ✓ "menjelaskan detail paket SEO premium untuk website buatan luar EDA"
        ✓ "mengonfirmasi pilihan paket galaxy sosmed ads bulanan"
        ✓ "melakukan negosiasi harga sebelum deal"
        ✓ "menanyakan total biaya setelah diskon"

        =======================================================================
        CONTOH INTENT (SALAH):
        =======================================================================
        ✗ "tanya_harga"
        ✗ "promo"
        ✗ "lainnya"
        ✗ "revisi"
        ✗ "info produk"

        =======================================================================
        KONTEKS (hanya untuk referensi, jangan dilabeli):
        =======================================================================
        {context_block if context_block else "(tidak ada konteks)"}

        =======================================================================
        PESAN TERAKHIR (INI YANG DILABELI):
        =======================================================================
        {target_role.upper()}: {target_text}

        =======================================================================
        OUTPUT (JSON ONLY):
        =======================================================================
        {{
        "intent": "...",
        "role": "{target_role}"
        }}
        """

        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {"intent": "lainnya"}
            parsed["role"] = target_role
            return parsed

        except Exception as e:
            return {"intent": "lainnya", "role": target_role, "error": str(e)}

    def label_worker(task):
        idx, target_text, target_role, ctx = task
        result = call_label(
            target_text=target_text,
            target_role=target_role,
            context_messages=ctx
        )
        return idx, result

    embedding_cache = {}
    node_embedding_cache = {}

    # ====================
    # LOAD EXISTING NODES
    # ====================
    existing_nodes = {}
    existing_node_map = {}

    if load_existing:
        # Load FLOW from existing_flow_path (SQLite)
        if os.path.exists(existing_flow_path):
            try:
                conn = sqlite3.connect(existing_flow_path)
                cursor = conn.cursor()
                
                # Load nodes
                cursor.execute("SELECT node_id, intent, category, role FROM flow_nodes")
                nodes_data = cursor.fetchall()
                
                # Load texts
                cursor.execute("SELECT node_id, chat, priority FROM flow_texts")
                texts_data = cursor.fetchall()
                
                # Load answers
                cursor.execute("SELECT from_node_id, intent, to_node_id FROM flow_answers")
                answers_data = cursor.fetchall()
                
                conn.close()
                
                # Build existing_nodes
                texts_dict = {}
                for node_id, chat, priority in texts_data:
                    if node_id not in texts_dict:
                        texts_dict[node_id] = []
                    texts_dict[node_id].append({"chat": chat, "priority": priority})
                
                answers_dict = {}
                for from_node_id, intent, to_node_id in answers_data:
                    if from_node_id not in answers_dict:
                        answers_dict[from_node_id] = OrderedDict()
                    if intent not in answers_dict[from_node_id]:
                        answers_dict[from_node_id][intent] = []
                    answers_dict[from_node_id][intent].append({"to": to_node_id})
                
                for node_id, intent, category, role in nodes_data:
                    existing_nodes[node_id] = {
                        "intent": intent,
                        "category": category,
                        "role": role,
                        "texts": texts_dict.get(node_id, []),
                        "answers": answers_dict.get(node_id, OrderedDict())
                    }
                    key = (intent.lower(), role.lower())
                    if key not in existing_node_map:
                        existing_node_map[key] = node_id
                
                if verbose:
                    print(f"Loaded {len(existing_nodes)} nodes from existing flow.")
            except Exception as e:
                print("Gagal load existing flow:", e)

        # Load embeddings from existing_embedding_path (SQLite)
        if os.path.exists(existing_embedding_path):
            try:
                conn = sqlite3.connect(existing_embedding_path)
                cursor = conn.cursor()
                cursor.execute("SELECT text, embedding FROM embeddings")
                emb_data = cursor.fetchall()
                conn.close()
                
                for text, emb_str in emb_data:
                    if text.startswith("node_"):
                        node_id = text[5:]  # Remove "node_" prefix
                        node_embedding_cache[node_id] = json.loads(emb_str)
                    else:
                        embedding_cache[text] = json.loads(emb_str)
                
                if verbose:
                    print(f"Loaded {len(node_embedding_cache)} node embeddings & {len(embedding_cache)} text embeddings.")
            except Exception as e:
                print("Gagal load existing embeddings:", e)

    text_role_to_intent = {}
    for node in existing_nodes.values():
        for text_item in node["texts"]:
            key = (text_item["chat"], node["role"])
            text_role_to_intent[key] = node["intent"]

    required_cols = ["conversation_id", "role", text_col, "priority"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' belum ada di dataframe!")

    # =============================================
    # BUILD CONTEXT MAP & PARALLEL INTENT LABELING
    # =============================================
    context_map = {}
    for cid, group in df.groupby("conversation_id"):
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            role = group.loc[i, "role"].lower()
            if role not in include_roles:
                continue
            ctx = []
            for j in range(max(0, i - 3), i):
                r = group.loc[j, "role"].upper()
                t = str(group.loc[j, text_col])
                ctx.append(f"{r}: {t}")
            context_map[(cid, i)] = ctx

    label_tasks = []
    for idx in range(len(df)):
        target_role = str(df.loc[idx, "role"]).lower()
        if target_role not in include_roles:
            continue

        target_text = str(df.loc[idx, text_col])
        conv_id = df.loc[idx, "conversation_id"]
        ctx = context_map.get((conv_id, idx), [])

        label_tasks.append((idx, target_text, target_role, ctx))

    label_cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, result in executor.map(label_worker, label_tasks):
            label_cache[idx] = result

    # ================================
    # BATCH EMBEDDING FOR UNIQUE TEXTS
    # ================================
    unique_texts = list(set(str(row[text_col]) for _, row in df.iterrows() if row["role"].lower() in include_roles))
    if unique_texts:
        emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=unique_texts
        )
        for t, e in zip(unique_texts, emb_resp.data):
            embedding_cache[t] = e.embedding

    label_results = []
    FLOW = {"meta": {"version": "1.0"}, "start": None, "nodes": {}}
    FLOW["nodes"].update(existing_nodes)
    node_map = existing_node_map.copy()

    node_counter = (
        max([int(nid[1:]) for nid in existing_nodes]) + 1
        if existing_nodes else 1
    )
    node_counter_original_start = node_counter
    nodes_created_since_last_save = 0

    # ==========
    # SAVE FLOW
    # ==========
    def save_flow():
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            conn = sqlite3.connect(save_path)
            cursor = conn.cursor()

            for node_id, node in FLOW["nodes"].items():
                cursor.execute(
                    "INSERT OR IGNORE INTO flow_nodes (node_id, intent, category, role) VALUES (?, ?, ?, ?)",
                    (node_id, node["intent"], node.get("category"), node["role"])
                )

                for text_item in node["texts"]:
                    cursor.execute(
                        "INSERT OR IGNORE INTO flow_texts (node_id, chat, priority) VALUES (?, ?, ?)",
                        (node_id, text_item["chat"], text_item["priority"])
                    )

                for intent, answers in node["answers"].items():
                    for ans in answers:
                        cursor.execute(
                            "INSERT OR IGNORE INTO flow_answers (from_node_id, intent, to_node_id) VALUES (?, ?, ?)",
                            (node_id, intent, ans["to"])
                        )

            conn.commit()
            conn.close()
            print(f"[autosave] Flow disimpan → {save_path}")

        except Exception as e:
            print("Gagal menyimpan flow:", e)

    # ===============
    # SAVE EMBEDDINGS
    # ===============
    def save_embeddings():
        try:
            os.makedirs(os.path.dirname(embedding_save_path), exist_ok=True)
            conn = sqlite3.connect(embedding_save_path)
            cursor = conn.cursor()
            
            # Save text embeddings
            for text, emb in embedding_cache.items():
                cursor.execute("INSERT OR IGNORE INTO embeddings (text, embedding) VALUES (?, ?)",
                               (text, json.dumps(emb)))
            
            # Save node embeddings
            for node_id, emb in node_embedding_cache.items():
                cursor.execute("INSERT OR IGNORE INTO embeddings (text, embedding) VALUES (?, ?)",
                               (f"node_{node_id}", json.dumps(emb)))
            
            conn.commit()
            conn.close()
            print(f"[autosave] Embeddings disimpan → {embedding_save_path}")
        except Exception as e:
            print("Gagal menyimpan embeddings:", e)

    # Function to compute node embedding (average of text embeddings)
    def compute_node_embedding(node_id, node):
        texts = [t["chat"] for t in node["texts"]]
        vectors = [embedding_cache[t] for t in texts if t in embedding_cache]
        if vectors:
            avg = np.mean(vectors, axis=0).tolist()
            node_embedding_cache[node_id] = avg
            return avg
        return []

    # ==================================================
    # FALLBACK: ISI ANSWER KOSONG DENGAN SIMILARITY
    # ==================================================
    def fill_empty_answers_with_similarity(FLOW, SIM_THRESHOLD=0.80, max_fallback=2):
        for node_id, node in FLOW["nodes"].items():
            if node.get("answers"):
                continue

            parent_role = node["role"]
            expected_role = "assistant" if parent_role == "user" else "user"
            parent_vec = node_embedding_cache.get(node_id)

            if not parent_vec:
                continue

            candidates = []

            for cand_id, cand_node in FLOW["nodes"].items():
                if cand_id == node_id:
                    continue

                if cand_node["role"] != expected_role:
                    continue

                cand_vec = node_embedding_cache.get(cand_id)
                if not cand_vec:
                    continue

                sim = cosine_similarity(parent_vec, cand_vec)
                if sim >= SIM_THRESHOLD:
                    candidates.append((sim, cand_id, cand_node["intent"]))

            candidates.sort(reverse=True)
            for sim, cand_id, intent in candidates[:max_fallback]:
                if intent not in node["answers"]:
                    node["answers"][intent] = []
                node["answers"][intent].append({
                    "to": cand_id,
                    "fallback": True,
                    "score": round(sim, 3)
                })

    # =======================
    # CATEGORY CLASSIFIER
    # =======================
    def call_flow_category(target_text, intent, target_role):
        prompt = f"""
    Kamu adalah AI CLASSIFIER SALES STRATEGY PROFESSIONAL.

    Tugas kamu adalah menentukan TAHAP FLOW MARKETING dari sebuah pesan
    berdasarkan alur penjualan jasa digital (website & digital marketing).

    ================================================================
    PRINSIP UTAMA (WAJIB DIPATUHI):
    ================================================================
    - Category = POSISI PESAN DALAM ALUR PENJUALAN
    - BUKAN jenis produk (website / ads / seo / dll)
    - Semua produk memakai ALUR YANG SAMA
    - Fokus pada TUJUAN KOMUNIKASI pesan
    - Jangan terpancing keyword, pahami konteks sales

    ================================================================
    FLOW CATEGORY (WAJIB PILIH SATU, TIDAK BOLEH MEMBUAT BARU):
    ================================================================
    - greeting_value_proposition
    - segmentasi_kualifikasi_prospek
    - edukasi_produk
    - penawaran_paket
    - call_to_action
    - conversational_closing
    - follow_up

    ================================================================
    PANDUAN LOGIKA KLASIFIKASI
    ================================================================

    1. greeting_value_proposition
    Gunakan jika:
    - Chat baru dimulai
    - Belum ada konteks kebutuhan user
    Tujuan:
    - Menyapa & menarik perhatian awal

    Contoh:
    "Halo kak, selamat datang di PT Eksa Digital Agency 😊
    Kami membantu pembuatan website & digital marketing untuk ribuan bisnis di Indonesia.
    Boleh tahu kak, saat ini sedang butuh apa?"

    ------------------------------------------------
    2. segmentasi_kualifikasi_prospek
    Gunakan jika:
    - Menggali kondisi & kebutuhan user
    - Mengelompokkan prospek

    Termasuk di dalamnya:
    - Pertanyaan bisnis
    - Identifikasi masalah
    - Bangun urgensi (belum punya website)
    - Optimalisasi (sudah punya website tapi belum maksimal)

    Contoh:
    "Usahanya di bidang apa kak?
    Saat ini sudah punya website belum?
    Kalau sudah, biasanya websitenya dipakai untuk apa?"

    ------------------------------------------------
    3. edukasi_produk
    Gunakan jika:
    - Menjelaskan layanan & manfaat
    - Membangun kepercayaan (trust building)

    Termasuk di dalamnya:
    - Penjelasan sistem kerja
    - Benefit produk
    - Testimoni
    - Portofolio
    - Legalitas perusahaan

    Contoh:
    "Website yang kami buat sudah termasuk domain, hosting, desain profesional,
    dan support penuh kak.
    Kami juga sudah menangani ratusan klien dari berbagai bidang usaha."

    ------------------------------------------------
    4. penawaran_paket
    Gunakan jika:
    - Membahas harga & paket
    - Menawarkan pilihan layanan

    Termasuk di dalamnya:
    - Paket BASIC / GOLD / DIAMOND
    - Upselling
    - Cross selling
    - Add-on layanan (SEO, Ads, Maintenance)

    ------------------------------------------------
    5. call_to_action
    Gunakan jika:
    - Mengajak user ke langkah konkret berikutnya

    Contoh:
    "Mau saya bantu cek domain yang tersedia kak?"
    "Kalau cocok, kita bisa mulai hari ini ya."

    ------------------------------------------------
    6. conversational_closing
    Gunakan jika:
    - Mengunci keputusan secara halus
    - Membahas DP / invoice / pembayaran
    - Menjelaskan sistem pembayaran resmi perusahaan

    Contoh:
    "Nanti pembayarannya via rekening perusahaan ya kak,
    setelah itu tim kami langsung mulai pengerjaan."

    ------------------------------------------------
    7. follow_up
    Gunakan jika:
    - User pasif
    - Perlu follow up ramah tanpa memaksa

    Contoh:
    "Halo kak 😊
    Mau follow up ya, apakah ada yang masih ingin ditanyakan
    terkait pembuatan websitenya?"

    ================================================================
    DATA PESAN:
    ================================================================
    Role   : {target_role}
    Intent : {intent}
    Chat   :
    \"\"\"
    {target_text}
    \"\"

    ================================================================
    OUTPUT (JSON SAJA, WAJIB VALID):
    ================================================================
    {{
    "category": "<salah_satu_flow_category>"
    }}
    """

        try:
            resp = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )

            raw = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            category = parsed.get("category")

            if category in FLOW_CATEGORIES:
                return category

            return None

        except Exception:
            return None

    # =======================
    # CATEGORY CACHE (STABLE)
    # =======================
    category_cache = {}
    unique_category_keys = set()
    for idx, result in label_cache.items():
        intent = result.get("intent", "lainnya").lower()
        role = result.get("role", "user")
        unique_category_keys.add((intent, role))

    def category_worker(args):
        intent, role = args
        category = call_flow_category(
            target_text=intent,
            intent=intent,
            target_role=role
        )
        return (intent, role), category

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for key, category in executor.map(category_worker, unique_category_keys):
            category_cache[key] = category

    # =============== START BUILDING FLOW ===============
    for idx in range(len(df)):

        target_text = str(df.loc[idx, text_col])
        target_role = str(df.loc[idx, "role"]).lower()
        priority    = int(df.loc[idx, "priority"]) if not pd.isna(df.loc[idx, "priority"]) else 0
        conv_id     = df.loc[idx, "conversation_id"]

        if target_role not in include_roles:
            continue

        result = label_cache.get(idx, {"intent": "lainnya"})
        intent = result.get("intent", "lainnya").lower()

        label_results.append({
            "conversation_id": conv_id,
            "index": idx,
            "intent": intent,
            "role": target_role,
            "text": target_text,
            "priority": priority,
        })

        key = (intent, target_role)

        # ==================================
        #  NODE BARU ATAU TAMBAH KE EXISTING
        # ==================================
        if key not in node_map:
            node_id = f"N{node_counter}"
            node_counter += 1
            node_map[key] = node_id
            node_category = category_cache.get((intent, target_role))

            FLOW["nodes"][node_id] = {
                "intent": intent,
                "category": node_category,
                "role": target_role,
                "texts": [{"chat": target_text, "priority": priority}],
                "answers": OrderedDict()
            }

            node_embedding_cache[node_id] = embedding_cache.get(target_text, [])
            nodes_created_since_last_save += 1

        else:
            node_id = node_map[key]
            node = FLOW["nodes"][node_id]

            if not node.get("category"):
                node["category"] = category_cache.get((intent, target_role))

            is_duplicate = any(
                t["chat"] == target_text and t.get("priority", 0) == priority
                for t in node["texts"]
            )

            if not is_duplicate:
                node["texts"].append({"chat": target_text, "priority": priority})
                compute_node_embedding(node_id, node)
            else:
                if verbose:
                    print(f"[skip] Duplicate text ditemukan di node {node_id}, tidak ditambahkan.")

        # ==============================================
        #  LOCAL EXPANSION (Similar Children Expansion)
        # ==============================================
        def local_similar_expansion(parent_id, child_id, FLOW, SIM_THRESHOLD=0.85):
            parent = FLOW["nodes"][parent_id]
            child = FLOW["nodes"][child_id]
            child_intent = child["intent"]
            child_role = child["role"]
            child_vec = node_embedding_cache.get(child_id)

            if not child_vec:
                return

            for cand_id, cand_node in FLOW["nodes"].items():
                if cand_id in (parent_id, child_id):
                    continue

                if cand_node["role"] != child_role:
                    continue

                if cand_node["intent"] == child_intent:
                    continue

                vecA = node_embedding_cache.get(cand_id)
                if vecA and cosine_similarity(vecA, child_vec) >= SIM_THRESHOLD:
                    ans_key = cand_node["intent"]
                    if ans_key not in parent["answers"]:
                        parent["answers"][ans_key] = []

                    if not any(e["to"] == cand_id for e in parent["answers"][ans_key]):
                        parent["answers"][ans_key].append({"to": cand_id})

        # =============================================
        #  GLOBAL EXPANSION (Update Texts dan Answers)
        # =============================================
        def global_expansion(new_child_id, FLOW, SIM_THRESHOLD=0.85):
            new_child = FLOW["nodes"][new_child_id]
            new_intent = new_child["intent"]
            new_role = new_child["role"]
            new_vec = node_embedding_cache.get(new_child_id)

            if not new_vec:
                return

            for parent_id, parent in FLOW["nodes"].items():
                if parent_id == new_child_id:
                    continue

                allowed_role = "assistant" if parent["role"] == "user" else "user"
                if new_role != allowed_role:
                    continue

                if new_intent not in parent.get("answers", {}):
                    continue

                anchor_children = parent["answers"][new_intent]
                if not anchor_children:
                    continue

                anchor_id = anchor_children[0]["to"]
                anchor_vec = node_embedding_cache.get(anchor_id)

                if not anchor_vec:
                    continue

                sim = cosine_similarity(anchor_vec, new_vec)
                if sim >= SIM_THRESHOLD:
                    if not any(e["to"] == new_child_id for e in anchor_children):
                        parent["answers"][new_intent].append({"to": new_child_id})

        # ===========================
        # 1) PARENT → CHILD (UTAMA)
        # ===========================
        if len(label_results) > 1:
            prev = label_results[-2]

            if prev["conversation_id"] == conv_id:
                prev_key = (prev["intent"], prev["role"])
                curr_key = (intent, target_role)

                if prev_key in node_map and curr_key in node_map:
                    parent_id = node_map[prev_key]
                    child_id = node_map[curr_key]

                    parent_node = FLOW["nodes"][parent_id]
                    child_node = FLOW["nodes"][child_id]

                    if parent_node["role"] != child_node["role"]:
                        child_intent = child_node["intent"]

                        if child_intent not in parent_node["answers"]:
                            parent_node["answers"][child_intent] = []

                        if not any(e["to"] == child_id for e in parent_node["answers"][child_intent]):
                            parent_node["answers"][child_intent].append({"to": child_id})

                        # ===================
                        # 2) LOCAL EXPANSION
                        # ===================
                        local_similar_expansion(parent_id, child_id, FLOW)

                        # =====================
                        # 3) GLOBAL EXPANSION
                        # =====================
                        global_expansion(child_id, FLOW)

    fill_empty_answers_with_similarity(FLOW)

    # ==============
    # SET START NODE
    # ==============
    df_intent = pd.DataFrame(label_results)
    if not df_intent.empty:
        first = df_intent.iloc[0]
        first_key = (first["intent"], first["role"])
        FLOW["start"] = node_map.get(first_key)

    save_flow()
    save_embeddings()

    new_nodes_only = {
        nid: node for nid, node in FLOW["nodes"].items()
        if int(nid[1:]) >= node_counter_original_start
    }
    return new_nodes_only, df_intent