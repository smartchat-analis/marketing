import os
import re
import json
import math
import threading
import logging
import requests
import ast
from openai import OpenAI
from collections import OrderedDict
from dotenv import load_dotenv
import anthropic
import sqlite3

def get_db(path):
    return sqlite3.connect(path, check_same_thread=False)

# ======================
# LOGGING SETUP
# ======================
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.DEBUG,  
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CHATBOT_ENGINE")

# ======================
# CONFIG
# ======================
FLOW_PATH = "output/global_flow.db"
NODE_EMB_PATH = "output/embeddings.db"
EMBEDDING_CACHE = OrderedDict() 
EMBEDDING_CACHE_MAX_SIZE = 1000
SESSION_CLEANUP_THRESHOLD = 100
SESSION_STORE_MAX_SIZE = 500
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY tidak ditemukan di environment variable")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY tidak ditemukan di environment variable")
client = OpenAI(
    api_key=openai_api_key
)
claude_client = anthropic.Anthropic(
    api_key=anthropic_api_key
)
SESSION_STORE = {}
SESSION_LOCK = threading.Lock()

# ======================
# LOAD DATA
# ======================
FLOW = None
EMB = None
NODES = None
NODE_EMB = None
LOADED = False

def load_flow_and_embeddings():
    global NODES, NODE_EMB, LOADED

    if LOADED:
        return

    if not os.path.exists(FLOW_PATH):
        raise FileNotFoundError("global_flow.db belum ada. Jalankan pipeline dulu.")

    if not os.path.exists(NODE_EMB_PATH):
        raise FileNotFoundError("embeddings.db belum ada.")

    # ======================
    # LOAD FLOW FROM SQLITE
    # ======================
    db = get_db(FLOW_PATH)
    cur = db.cursor()

    # ---- flow_nodes
    cur.execute("""
        SELECT node_id, intent, category, role
        FROM flow_nodes
    """)
    NODES = {
        row[0]: {
            "intent": row[1],
            "category": row[2],
            "role": row[3],
            "texts": [],
            "answers": {}
        }
        for row in cur.fetchall()
    }

    # ---- flow_texts
    cur.execute("""
        SELECT node_id, chat, priority
        FROM flow_texts
        ORDER BY priority ASC
    """)
    for node_id, chat, priority in cur.fetchall():
        if node_id in NODES:
            NODES[node_id]["texts"].append({
                "chat": chat,
                "priority": priority
            })

    # ---- flow_answers
    cur.execute("""
        SELECT from_node_id, intent, to_node_id
        FROM flow_answers
    """)
    for from_node, intent, to_node in cur.fetchall():
        if from_node in NODES:
            NODES[from_node]["answers"].setdefault(intent, []).append({
                "to": to_node
            })

    db.close()

    # ===========================
    # LOAD EMBEDDINGS FROM SQLITE
    # ===========================
    emb_db = get_db(NODE_EMB_PATH)
    emb_cur = emb_db.cursor()

    emb_cur.execute("""
        SELECT text, embedding
        FROM embeddings
    """)

    NODE_EMB = {}
    for text, emb_str in emb_cur.fetchall():
        try:
            if not text:
                continue
            NODE_EMB[text] = ast.literal_eval(emb_str)
        except Exception:
            continue

    emb_db.close()

# ======================
# UTILS
# ======================
def normalize_text(t: str):
    return re.sub(r"\s+", " ", t.lower().strip())

def cosine_similarity(vecA, vecB):
    if not vecA or not vecB:
        return 0.0
    dot = sum(a * b for a, b in zip(vecA, vecB))
    magA = sum(a * a for a in vecA) ** 0.5
    magB = sum(b * b for b in vecB) ** 0.5
    if magA == 0 or magB == 0:
        return 0.0
    return dot / (magA * magB)

def dynamic_threshold(text):
    length = len(text.split())
    if length <= 3:
        return 0.4  
    elif length <= 6:
        return 0.5 
    else:
        return 0.6

def safe_parse_json(text):
    try:
        return json.loads(text)
    except:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return None

def embed_text(text: str):
    logger.debug(f"[EMBED] Request received | text_preview='{text[:50]}'")

    if text in EMBEDDING_CACHE:
        logger.debug("[EMBED CACHE HIT]")
        EMBEDDING_CACHE.move_to_end(text)
        return EMBEDDING_CACHE[text]

    try:
        logger.debug("[EMBED] Generating embedding from OpenAI")

        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )

        embedding = resp.data[0].embedding

        logger.debug(f"[EMBED SUCCESS] vector_length={len(embedding)}")

        EMBEDDING_CACHE[text] = embedding

        if len(EMBEDDING_CACHE) > EMBEDDING_CACHE_MAX_SIZE:
            logger.debug("[EMBED CACHE] Max size reached, removing oldest item")
            EMBEDDING_CACHE.popitem(last=False)

        return embedding

    except Exception:
        logger.exception(f"[EMBED ERROR] Failed for text_preview='{text[:50]}'")
        return []

def build_conversation_context(session_data, max_messages=10):
    history = session_data.get("history", [])
    if not history:
        return []
    return history[-max_messages:]

def get_top_priority_candidates(candidates, top_k=5):
    if not candidates:
        return []
    for c in candidates:
        similarity = c.get("similarity", 0.0)
        priority = c.get("priority", 0)
        c["final_score"] = (
            0.7 * similarity +
            0.3 * priority
        )
    candidates_sorted = sorted(
        candidates,
        key=lambda x: x["final_score"],
        reverse=True
    )
    return candidates_sorted[:top_k]

# ===============================
# ITERATIVE RAG RETRIEVAL ENGINE
# ===============================
def iterative_node_search(
    user_vec,
    user_message,
    user_category,
    prev_node_id,
    assistant_category,
    max_attempts=5
):
    threshold = dynamic_threshold(user_message)

    for attempt in range(max_attempts):
        best, metadata = find_best_user_node(
            user_vec,
            user_message,
            user_category=user_category,
            prev_node_id=prev_node_id,
            assistant_category=assistant_category,
            custom_threshold=threshold
        )

        if best:
            return best, metadata

        # turunkan threshold bertahap
        threshold *= 0.9

    # fallback: ambil similarity tertinggi walau di bawah threshold
    best, metadata = find_best_user_node(
        user_vec,
        user_message,
        user_category=None,
        prev_node_id=None,
        assistant_category=None
    )

    return best, metadata

# ===========================
# LLM 2 LAYER
# ===========================
def llm_validate_and_generate(    
    user_message,
    user_intent,
    knowledge_context,
    context_messages=None
):
    context = ""
    if context_messages:
        context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in context_messages[-9:]
        )

    logger.debug("[LLM1] Preparing prompt for Claude")
    logger.debug(f"[LLM1] user_intent={user_intent}")
    logger.debug(f"[LLM1] knowledge_length={len(knowledge_context) if knowledge_context else 0}")
    logger.debug(f"[LLM1] context_length={len(context)}")

    prompt = f"""
    KAMU adalah Admin Marketing WhatsApp profesional untuk layanan:
    pembuatan website, SEO, Google Ads, sosial media ads, kelola sosial media,
    company profile PDF, pembuatan akun sosial media, pembuatan Google Maps,
    pembuatan email bisnis, dan layanan digital marketing lainnya.

    PERAN UTAMA KAMU:
    Bukan membuat jawaban panjang.
    Tugas kamu adalah:

    1. Memvalidasi hasil dari proses routing & knowledge selection
    2. Mengolah kandidat jawaban terbaik agar lebih natural
    3. Menjawab secukupnya sesuai pertanyaan user
    4. Tetap terlihat seperti admin manusia, bukan bot

    =====================================================
    PRINSIP UTAMA
    =====================================================

    - Fokus pada kandidat jawaban paling relevan (prioritas pertama).
    - Jangan merangkum semua knowledge sekaligus.
    - Jangan menampilkan terlalu banyak informasi dalam satu pesan.
    - Jangan menggabungkan semua kemungkinan jawaban.
    - Jangan terlihat seperti FAQ generator.

    Jika kandidat pertama sudah sesuai dan tidak bertentangan
    dengan kandidat lain:
    → Gunakan jawaban tersebut.
    → Rapikan bahasanya agar lebih natural.
    → Jangan diperpanjang tanpa alasan.

    =====================================================
    BATAS PANJANG JAWABAN
    =====================================================

    - Jawab hanya sesuai yang ditanyakan user.
    - Jangan memberi informasi tambahan yang belum diminta.
    - Hindari paragraf panjang.
    - Hindari list panjang kecuali memang diminta.
    - Jangan kirim banyak link sekaligus kecuali diminta secara spesifik.
    - Default: 2–5 kalimat saja.

    Tujuan:
    Agar tidak terlihat seperti bot atau sales script panjang.

    =====================================================
    JIKA TIDAK ADA NODE TERPILIH
    =====================================================

    Jika setelah proses routing tidak ada knowledge yang relevan,
    atau jawabannya berpotensi tidak akurat:

    Gunakan respon berikut secara persis:

    "Terima kasih atas pertanyaannya😊 Untuk memastikan informasi yang sesuai, izin kami koordinasikan terlebih dahulu dengan tim terkait ya. Nanti akan segera kami informasikan kembali🙏"

    Jangan dimodifikasi.
    Jangan ditambahkan kalimat lain.

    =====================================================
    KONTROL KONTEKS
    =====================================================

    Prioritas pemahaman:
    1) SUMMARY CONTEXT
    2) USER MESSAGE terbaru
    3) KNOWLEDGE CONTEXT

    Jika knowledge terlalu banyak:
    → Ambil yang paling relevan dengan pertanyaan terakhir saja.

    Jika knowledge tidak relevan:
    → Abaikan.

    =====================================================
    ANTI-OVEREXPLAIN
    =====================================================

    Dilarang:

    - Mengirim 5–10 contoh link sekaligus tanpa diminta
    - Memberikan semua detail paket dalam satu pesan
    - Menjelaskan seluruh layanan jika user hanya tanya satu hal
    - Memberikan promosi panjang tanpa diminta

    =====================================================
    GAYA KOMUNIKASI
    =====================================================

    - Profesional
    - Natural
    - Tidak agresif
    - Tidak berlebihan
    - Tidak menggunakan tanda seru (!)
    - Maksimal 2 emoticon ringan
    - Tidak menggunakan markdown seperti **bold**

    =====================================================
    VALIDASI SEBELUM OUTPUT
    =====================================================

    Pastikan:
    1. Jawaban tidak terlalu panjang
    2. Tidak keluar dari pertanyaan user
    3. Tidak merangkum seluruh knowledge
    4. Tidak terlihat seperti template panjang
    5. Tidak ada tanda seru (!)

    =====================================================
    INPUT
    =====================================================

    SUMMARY CONTEXT:
    {context}

    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    KNOWLEDGE CONTEXT:
    {knowledge_context if knowledge_context else "(KOSONG)"}

    =====================================================
    OUTPUT (JSON ONLY)
    =====================================================

    {{
    "response": "jawaban final yang singkat, natural, dan tervalidasi"
    }}
    """

    try:
        logger.debug("[LLM1] Calling Claude validate+generate")

        resp = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        raw = resp.content[0].text.strip()
        logger.debug(f"[LLM1 RAW OUTPUT] {raw[:300]}")

        parsed = safe_parse_json(raw)
        if not parsed or "response" not in parsed:
            logger.warning("[LLM1] Invalid JSON output, using raw response")
            logger.debug(f"[LLM1 RAW FALLBACK] {raw}")
            return {
                "response": raw,
                "prompt": prompt
            }
        logger.debug("[LLM1] JSON parsed successfully")
        return {
            "response": parsed.get("response", ""),
            "prompt": prompt
        }
    except Exception:
        logger.exception("[LLM1 ERROR] Claude validate+generate failed")
        raise

def sanitize_llm_response(
    user_message: str,
    user_intent: str,
    context: str,
    final_response_llm1: str
):
    logger.debug("[LLM2] Sanitizing response")
    logger.debug(f"[LLM2] user_intent={user_intent}")
    logger.debug(f"[LLM2] LLM1_response_preview='{final_response_llm1[:200]}'")

    prompt = f"""
    Kamu adalah AI FILTER profesional.

    Tugasmu:
    - MENGANALISIS response dari LLM layer 1
    - MENDETEKSI unsur sensitif
    - MENGGANTI dengan placeholder
    - TIDAK membuat jawaban atau kalimat baru
    - TIDAK mengubah struktur kalimat
    - HANYA mengganti bagian sensitif dengan placeholder yang sesuai

    ================================================
    UNSUR SENSITIF (WAJIB DIGANTI)
    ================================================
    Aturan umum:
    - Penggantian HANYA dilakukan jika teks secara eksplisit menyebut nama spesifik perusahaan internal berikut.
    - Jangan mengganti kata umum seperti: "perusahaan", "company", "usaha", atau perusahaan milik klien.
    - Jangan mengganti jika konteksnya sedang menanyakan bidang usaha klien.
    - Jangan melakukan penggantian berdasarkan asumsi.

    1. Nama perusahaan internal

    HANYA jika teks secara eksplisit dan persis menyebut salah satu nama berikut:
    - Asain
    - EDA
    - EbyB
    - PT. Asa Inovasi Software
    - PT. Eksa Digital Agency
    - PT. EBYB Global Marketplace

    Maka WAJIB diganti menjadi:
    {"{{$company}}"}

    Jika hanya menyebut kata umum seperti:
    - perusahaan
    - company
    - bisnis
    - usaha
    - perusahaan kakak
    MAKA JANGAN DIGANTI.

    -----------------------------------------------------------------------------

    2. Alamat perusahaan internal

    HANYA jika menyebut alamat resmi perusahaan internal (Asain, EDA, atau EbyB).

    Bukan alamat milik klien.
    Bukan alamat umum.
    Bukan contoh alamat.

    Jika menyebut alamat internal secara spesifik,
    WAJIB diganti menjadi:
    {"{{$address}}"}

    ------------------------------------------------------------------------------

    3. Nomor rekening pembayaran internal

    HANYA jika menyebut nomor rekening milik perusahaan internal (Asain, EDA, atau EbyB).

    Bukan rekening klien.
    Bukan contoh rekening.
    Bukan sekadar menyebut nama bank.

    Jika menyebut nomor rekening internal secara spesifik,
    WAJIB diganti menjadi:
    {"{{$rekening_pembayaran}}"}

    ================================================
    INPUT
    ================================================
    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    KONTEKS:
    {context}

    FINAL RESPONSE (LLM 1):
    {final_response_llm1}

    ================================================
    OUTPUT (JSON ONLY)
    ================================================
    {{
    "response": "response hanya data sensitif yang diganti",
    "sensitive_found": true/false
    }}
    """
    try:
        logger.debug("[LLM2] Calling GPT sanitize layer")

        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        raw = resp.choices[0].message.content.strip()
        logger.debug(f"[LLM2 RAW OUTPUT] {raw[:300]}")

        # ==========================
        # EXTRACT JSON
        # ==========================
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("[LLM2] No JSON detected, using LLM1 result")
            return {
                "response": final_response_llm1,
                "prompt": prompt
            }

        parsed = safe_parse_json(raw)
        if not parsed:
            return {
                "response": final_response_llm1,
                "prompt": prompt
            }

        response_text = parsed.get("response")
        sensitive_found = parsed.get("sensitive_found", False)

        logger.debug(f"[LLM2] sensitive_found={sensitive_found}")

        # ==========================
        # LOGIC UTAMA
        # ==========================
        if response_text:
            # Kalau ada hasil sanitize → pakai hasil LLM2
            logger.debug("[LLM2] Returning sanitized response")
            return {
                "response": response_text,
                "prompt": prompt
            }

        # Kalau response kosong → fallback
        logger.warning("[LLM2] Response empty, fallback to LLM1")
        return {
            "response": final_response_llm1,
            "prompt": prompt
        }

    except Exception:
        logger.exception("[LLM2 ERROR] Sanitize layer failed")
        return {
            "response": final_response_llm1,
            "prompt": prompt
        }

# =================================
# INTENT AND CATEGORY CLASSIFIER
# =================================
def call_intent_and_category(user_message, role, context_messages):
    context_block = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_messages[-3:]]) if context_messages else "(tidak ada konteks)"

    prompt = f"""
    Kamu adalah AI classifier profesional yang menentukan INTENT dan FLOW CATEGORY dari percakapan antara klien dan tim marketing dari salah satu perusahaan berikut:

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

    =======================================================================
    FLOW CATEGORY:
    =======================================================================
    Tentukan FLOW CATEGORY dari pesan berikut.
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
    - Jika pesan adalah salam pertama tanpa konteks sebelumnya (misalnya, "halo" atau menyebut sumber), klasifikasi sebagai "greeting_value_proposition".  # TAMBAHAN: Aturan eksplisit untuk greeting

    PILIH SATU CATEGORY:
    - greeting_value_proposition
    - segmentasi_kualifikasi_prospek
    - edukasi_produk
    - penawaran_paket
    - call_to_action
    - conversational_closing
    - follow_up

    KONTEKS:
    {context_block}

    PESAN:
    {user_message}

    OUTPUT (JSON ONLY):
    {{
      "intent": "...",
      "category": "..."
    }}
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = resp.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(match.group(0))
        return parsed.get("intent", "lainnya"), parsed.get("category", None)
    except Exception as e:
        logging.error(f"Intent and category classification failed: {e}")
        return "lainnya", None

# ============
# CORE LOGIC
# ============
def find_best_user_node(
    user_vec,
    user_message,
    user_category=None,
    prev_node_id=None,
    assistant_category=None,
    custom_threshold=None
):
    logger.debug("[ROUTING] Start find_best_user_node")

    candidates, flow_candidates, global_candidates = [], [], []
    threshold = custom_threshold if custom_threshold is not None else dynamic_threshold(user_message)
    user_norm = normalize_text(user_message)

    logger.debug(f"[ROUTING] threshold={threshold}")
    logger.debug(f"[ROUTING] user_category={user_category}")
    logger.debug(f"[ROUTING] prev_node_id={prev_node_id}")

    # ======================
    # FLOW FIRST
    # ======================
    if prev_node_id and prev_node_id in NODES:
        logger.debug("[ROUTING] Checking FLOW candidates")

        prev_node = NODES[prev_node_id]

        for intent, edges in prev_node.get("answers", {}).items():
            for e in edges:
                cid = e["to"]

                if cid in NODES and NODES[cid]["role"] == "user":
                    best_sim = 0.0

                    for t in NODES[cid].get("texts", []):
                        emb = NODE_EMB.get(t["chat"])
                        if not emb:
                            continue

                        sim = cosine_similarity(user_vec, emb)
                        best_sim = max(best_sim, sim)

                    if best_sim > 0:
                        flow_candidates.append((cid, best_sim))

                        if best_sim >= threshold:
                            candidates.append((cid, best_sim, "flow"))

        logger.debug(f"[ROUTING] flow_candidates_count={len(flow_candidates)}")

    # ======================
    # GLOBAL FALLBACK
    # ======================
    if not candidates:
        logger.debug("[ROUTING] Checking GLOBAL candidates")

        for nid, node in NODES.items():
            if node["role"] != "user":
                continue

            node_cat = node.get("category")
            if user_category and node_cat != user_category:
                continue

            best_sim = 0.0

            for t in node.get("texts", []):
                emb = NODE_EMB.get(t["chat"])
                if not emb:
                    continue

                sim = cosine_similarity(user_vec, emb)
                best_sim = max(best_sim, sim)

            if best_sim > 0:
                global_candidates.append((nid, best_sim))

                if best_sim >= threshold:
                    candidates.append((nid, best_sim, "global"))

        logger.debug(f"[ROUTING] global_candidates_count={len(global_candidates)}")

    # ======================
    # NO MATCH
    # ======================
    if not candidates:
        logger.debug("[ROUTING] No candidate matched threshold")

        return None, {
            "flow_count": len(flow_candidates),
            "global_count": len(global_candidates)
        }

    # ======================
    # SORT & PICK BEST
    # ======================
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    top_5 = candidates[:5]

    logger.debug(
        f"[ROUTING] best_node={best[0]} "
        f"sim={round(best[1],3)} "
        f"source={best[2]}"
    )
    logger.debug(f"[ROUTING] top_5={[(c[0], round(c[1],3), c[2]) for c in top_5]}")

    return best, {
        "flow_count": len(flow_candidates),
        "global_count": len(global_candidates),
        "best_sim": best[1],
        "best_type": best[2],
        "best_user_node_id": best[0],
        "top_5_user_nodes": [
            {
                "node_id": cid,
                "similarity": round(sim, 3),
                "source": src
            }
            for cid, sim, src in top_5
        ]
    }

def collect_assistant_knowledge_from_user_nodes(top_5_user_nodes):
    knowledge_chunks = []

    for item in top_5_user_nodes:
        user_node_id = item["node_id"]
        node = NODES.get(user_node_id)

        if not node or node["role"] != "user":
            continue

        for _, edges in node.get("answers", {}).items():
            for e in edges:
                aid = e["to"]
                if aid in NODES and NODES[aid]["role"] == "assistant":
                    for t in NODES[aid].get("texts", []):
                        knowledge_chunks.append({
                            "assistant_node_id": aid,
                            "text": t["chat"]
                        })

    return knowledge_chunks

def resolve_assistant_node_from_best_user(best_user_node_id):
    # Fungsi helper untuk resolve assistant node dari best user node
    node = NODES.get(best_user_node_id)
    if not node or node["role"] != "user":
        return None

    # Ambil assistant node pertama dari answers
    for _, edges in node.get("answers", {}).items():
        for e in edges:
            aid = e["to"]
            if aid in NODES and NODES[aid]["role"] == "assistant":
                return aid
    return None

def get_response_from_knowledge(
    best_user_node_id,
    top_5_user_nodes
):
    knowledge = collect_assistant_knowledge_from_user_nodes(
        top_5_user_nodes
    )

    if not knowledge:
        return {
            "assistant_node_id": None,
            "knowledge_context": ""
        }

    knowledge_context = "\n".join(
        f"- {k['text']}"
        for k in knowledge
    )

    return {
        "assistant_node_id": resolve_assistant_node_from_best_user(best_user_node_id),
        "knowledge_context": knowledge_context
    }

def generate_assistant_response(
    user_message,
    user_intent=None,
    user_category=None,
    prev_node_id=None,
    assistant_category=None,
    context_messages=None
):
    user_vec = embed_text(user_message)

    best_user_node, metadata = iterative_node_search(
        user_vec,
        user_message,
        user_category,
        prev_node_id,
        assistant_category
    )

    # ==============================
    # TIDAK ADA USER NODE
    # ==============================
    if not best_user_node:
        detected_intent = user_intent

        llm_result = llm_validate_and_generate(
            user_message=user_message,
            user_intent=detected_intent,
            knowledge_context="",
            context_messages=context_messages
        )

        # ============================================
        # DEBUG: Log LLM1 result
        # ============================================
        logger.debug(f"[LLM1 RESULT] keys={llm_result.keys()}")
        logger.debug(f"[LLM1 RESULT] response_preview={str(llm_result.get('response', ''))[:100]}")

        raw_response = llm_result.get("response", "")

        sanitized = sanitize_llm_response(
            user_message=user_message,
            user_intent=detected_intent,
            context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
            final_response_llm1=raw_response
        )

        final_response = sanitized.get("response", raw_response)

        # ============================================
        # DEBUG: Log LLM2 result
        # ============================================
        logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")

        return {
            "response": final_response,
            "node_id": None,
            "metadata": metadata,
            "llm1_prompt": llm_result.get("prompt"),
            "llm2_prompt": sanitized.get("prompt")
        }

    # ==============================
    # ADA USER NODE
    # ==============================
    user_node_id = best_user_node[0]

    knowledge_data = get_response_from_knowledge(
        best_user_node_id=user_node_id,
        top_5_user_nodes=metadata["top_5_user_nodes"]
    )

    llm_result = llm_validate_and_generate(
        user_message=user_message,
        user_intent=NODES.get(user_node_id, {}).get("intent", ""),
        knowledge_context=knowledge_data["knowledge_context"],
        context_messages=context_messages
    )

    # ============================================
    # DEBUG: Log LLM1 result
    # ============================================
    logger.debug(f"[LLM1 RESULT] keys={llm_result.keys()}")
    logger.debug(f"[LLM1 RESULT] response_preview={str(llm_result.get('response', ''))[:100]}")

    raw_response = llm_result.get("response", "")

    # FINAL QUALITY CONTROL
    sanitized = sanitize_llm_response(
        user_message=user_message,
        user_intent=NODES.get(user_node_id, {}).get("intent", ""),
        context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
        final_response_llm1=raw_response
    )

    final_response = sanitized.get("response", raw_response)
    final_response = final_response.replace("!", "")

    # ============================================
    # DEBUG: Log LLM2 result
    # ============================================logger.debug(f"[LLM2 RESULT] applied_code_product={applied_code}")
    logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")

    return {
        "response": final_response,
        "node_id": knowledge_data["assistant_node_id"],
        "metadata": metadata,
        "llm1_prompt": llm_result.get("prompt"),
        "llm2_prompt": sanitized.get("prompt")
    }

# ======================
# MAIN ENTRY
# ======================
def chat_with_session(user_message, session_id, reset=False):

    logger.debug(f"[SESSION START] session_id={session_id}")
    logger.debug(f"[USER MESSAGE] {user_message}")

    with SESSION_LOCK:
        session_data = SESSION_STORE.get(session_id, {})

        # BUILD CONTEXT
        context_messages = build_conversation_context(
            session_data,
            max_messages=9
        )

        # INTENT & CATEGORY (ONCE)
        user_intent, user_category = call_intent_and_category(
            user_message,
            "user",
            context_messages
        )

        logger.debug(f"[INTENT] {user_intent}")
        logger.debug(f"[CATEGORY] {user_category}")

        if reset:
            prev_node_id = None
            assistant_category = user_category
            session_data = {"history": []}
            SESSION_STORE[session_id] = session_data
        else:
            prev_node_id = session_data.get("prev_node_id")
            assistant_category = session_data.get("assistant_category")

        # SESSION CLEANUP
        if len(SESSION_STORE) > SESSION_CLEANUP_THRESHOLD:
            for key in list(SESSION_STORE.keys())[:10]:
                del SESSION_STORE[key]

        if len(SESSION_STORE) > SESSION_STORE_MAX_SIZE:
            for key in list(SESSION_STORE.keys())[:10]:
                del SESSION_STORE[key]

    # ======================
    # GENERATE RESPONSE
    # ======================
    result = generate_assistant_response(
        user_message,
        user_intent=user_intent,
        user_category=user_category,
        prev_node_id=prev_node_id,
        assistant_category=assistant_category,
        context_messages=context_messages
    )

    final_response = result["response"]
    assistant_node_id = result.get("node_id")
    metadata = result.get("metadata", {})

    # ======================
    # DEBUG INFO
    # ======================
    assistant_node = NODES.get(assistant_node_id, {})
    debug_info = {
        "user_intent": user_intent,
        "user_category": user_category,

        # ROUTING
        "match_type": metadata.get("best_type"),
        "best_similarity": round(metadata.get("best_sim", 0), 3),
        "best_user_node_id": metadata.get("best_user_node_id"),

        # TOP 5 USER NODE AS KNOWLEDGE SOURCE
        "top_5_user_nodes": metadata.get("top_5_user_nodes", []),

        # RESULT
        "selected_user_node": metadata.get("best_user_node_id")
        if metadata.get("top_5_user_nodes") else None,

        "assistant_node_id": assistant_node_id,
        "assistant_intent": assistant_node.get("intent"),
        "assistant_category": assistant_node.get("category") or user_category,

        # COUNTS
        "flow_candidates_count": metadata.get("flow_count"),
        "global_candidates_count": metadata.get("global_count"),

        # CONTEXT SNAPSHOT
        "context_summary": " | ".join(
            f"{m['role']}: {m['content']}"
            for m in context_messages[-9:]
        ) if context_messages else None,
        "llm1_prompt": result.get("llm1_prompt"),
        "llm2_prompt": result.get("llm2_prompt")
    }

    # ======================
    # UPDATE SESSION
    # ======================
    with SESSION_LOCK:
        history = session_data.get("history", [])

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": final_response})

        session_data["history"] = history[-9:]
        session_data["prev_node_id"] = assistant_node_id
        session_data["assistant_category"] = assistant_node.get("category") or user_category

        SESSION_STORE[session_id] = session_data

    logger.debug(f"[FINAL RESPONSE] {final_response}")
    logger.debug(f"[ASSISTANT NODE ID] {assistant_node_id}")
    logger.debug(f"[SESSION END] session_id={session_id}")

    return {
        "response": final_response,
        "node_id": assistant_node_id,
        "debug": debug_info
    }
