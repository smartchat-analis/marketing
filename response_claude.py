import os
import re
import json
import math
import threading
import logging
import requests
import ast
import time
from openai import OpenAI
from collections import OrderedDict
from dotenv import load_dotenv
import anthropic
import sqlite3
load_dotenv()
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
SESSION_STORE_MAX_SIZE = 500
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
    if len(vecA) != len(vecB):
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

def embed_text(text: str, max_retries: int = 3):
    logger.debug(f"[EMBED] Request received | text_preview='{text[:50]}'")

    if not text or not text.strip():
        raise ValueError("embed_text received empty text")

    if text in EMBEDDING_CACHE:
        logger.debug("[EMBED CACHE HIT]")
        EMBEDDING_CACHE.move_to_end(text)
        return EMBEDDING_CACHE[text]

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"[EMBED] Attempt {attempt}")

            resp = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )

            embedding = resp.data[0].embedding

            if not embedding or len(embedding) < 100:
                raise ValueError("Embedding vector invalid or too short")

            EMBEDDING_CACHE[text] = embedding

            if len(EMBEDDING_CACHE) > EMBEDDING_CACHE_MAX_SIZE:
                EMBEDDING_CACHE.popitem(last=False)

            logger.debug(f"[EMBED SUCCESS] vector_length={len(embedding)}")
            return embedding

        except Exception as e:
            last_exception = e
            logger.warning(f"[EMBED ERROR] attempt={attempt} | {e}")
            time.sleep(0.8 * attempt)

    logger.critical("[EMBED FAILED] All retries exhausted")
    raise RuntimeError("Embedding generation failed") from last_exception

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

def get_price_context(category_product_list):
    price_chunks = []

    for cat in category_product_list or []:
        if cat == "WEBSITE":
            from price_website import PRICE_WEBSITE
            price_chunks.append(PRICE_WEBSITE)

        elif cat == "SEO":
            from price_seo import PRICE_SEO
            price_chunks.append(PRICE_SEO)

        elif cat == "GOOGLE_ADS":
            from price_google_ads import PRICE_GOOGLE_ADS
            price_chunks.append(PRICE_GOOGLE_ADS)

        elif cat == "SOSMED_ADS":
            from price_sosmed_ads import PRICE_SOSMED_ADS
            price_chunks.append(PRICE_SOSMED_ADS)

        elif cat == "COMPANY_PROFILE_PDF":
            from price_comprof import PRICE_COMPROF
            price_chunks.append(PRICE_COMPROF)

        elif cat == "SOSIAL_MEDIA_NON_ADS":
            from price_sosmed import PRICE_SOSMED
            price_chunks.append(PRICE_SOSMED)

        elif cat == "LAYANAN_DIGITAL_LAINNYA":
            from price_lainnya import PRICE_LAINNYA
            price_chunks.append(PRICE_LAINNYA)

    return "\n\n".join(price_chunks)

def get_product_knowledge(category_product_list):
    knowledge_chunks = []

    for cat in category_product_list or []:
        if cat == "WEBSITE":
            from knowledge_website import KNOWLEDGE_WEBSITE
            knowledge_chunks.append(KNOWLEDGE_WEBSITE)

        elif cat == "SEO":
            from knowledge_SEO import KNOWLEDGE_SEO
            knowledge_chunks.append(KNOWLEDGE_SEO)

        elif cat == "GOOGLE_ADS":
            from knowledge_google_ads import KNOWLEDGE_GOOGLE_ADS
            knowledge_chunks.append(KNOWLEDGE_GOOGLE_ADS)

        elif cat == "SOSMED_ADS":
            from knowledge_sosmed_ads import KNOWLEDGE_SOSMED_ADS
            knowledge_chunks.append(KNOWLEDGE_SOSMED_ADS)

        elif cat == "COMPANY_PROFILE_PDF":
            from knowledge_comprof import KNOWLEDGE_COMPROF
            knowledge_chunks.append(KNOWLEDGE_COMPROF)

        elif cat == "SOSIAL_MEDIA_NON_ADS":
            from knowledge_sosmed import KNOWLEDGE_SOSMED
            knowledge_chunks.append(KNOWLEDGE_SOSMED)

        elif cat == "LAYANAN_DIGITAL_LAINNYA":
            from knowledge_lainnya import KNOWLEDGE_LAINNYA
            knowledge_chunks.append(KNOWLEDGE_LAINNYA)

    return "\n\n".join(knowledge_chunks)

# ============================
# KNOWLEDGE MISMATCH DETECTOR
# ============================
def is_knowledge_mismatch(category_product, knowledge_context):
    if not category_product or not knowledge_context:
        return False

    category_product = set(category_product)
    knowledge_lower = knowledge_context.lower()

    # Simple heuristic detection
    if "WEBSITE" in category_product and "seo" in knowledge_lower:
        return True

    if "SEO" in category_product and "website" in knowledge_lower:
        return True

    if "GOOGLE_ADS" in category_product and "seo" in knowledge_lower:
        return True

    return False

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

    logger.debug("========== ITERATIVE RAG START ==========")
    logger.debug(f"[ITERATIVE] user_message='{user_message}'")
    logger.debug(f"[ITERATIVE] initial_threshold={round(threshold,3)}")
    logger.debug(f"[ITERATIVE] max_attempts={max_attempts}")

    for attempt in range(1, max_attempts + 1):

        logger.debug(f"----- Attempt {attempt} -----")
        logger.debug(f"[ITERATIVE] threshold={round(threshold,3)}")

        best, metadata = find_best_user_node(
            user_vec,
            user_message,
            user_category=user_category,
            prev_node_id=prev_node_id,
            assistant_category=assistant_category,
            custom_threshold=threshold
        )

        flow_count = metadata.get("flow_count")
        global_count = metadata.get("global_count")

        logger.debug(f"[ITERATIVE] flow_candidates={flow_count}")
        logger.debug(f"[ITERATIVE] global_candidates={global_count}")

        if best:
            logger.debug(
                f"[ITERATIVE SUCCESS] attempt={attempt} | "
                f"node_id={best[0]} | "
                f"similarity={round(best[1],3)} | "
                f"source={best[2]}"
            )
            logger.debug("========== ITERATIVE RAG END ==========")
            return best, metadata

        logger.debug(f"[ITERATIVE] No match on attempt {attempt}")

        # turunkan threshold bertahap
        threshold *= 0.9

    # ===============================
    # FINAL FALLBACK (FORCE BEST MATCH)
    # ===============================
    logger.debug("----- FINAL FALLBACK -----")
    logger.debug("[ITERATIVE] Forcing best similarity without threshold")

    best, metadata = find_best_user_node(
        user_vec,
        user_message,
        user_category=None,
        prev_node_id=None,
        assistant_category=None,
        custom_threshold=0.2
    )

    if best:
        logger.debug(
            f"[ITERATIVE FALLBACK RESULT] "
            f"node_id={best[0]} | "
            f"similarity={round(best[1],3)} | "
            f"source={best[2]}"
        )
    else:
        logger.debug("[ITERATIVE FALLBACK RESULT] No node found at all")

    logger.debug("========== ITERATIVE RAG END ==========")

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
    Kamu adalah Admin Marketing WhatsApp representatif sebuah agency digital.
    Jawab pertanyaan user dengan ringkas, profesional, natural, dan tidak seperti bot.

    TUGAS UTAMA:
    1. Validasi dan olah KNOWLEDGE CONTEXT yang sudah dipilih sistem.
    2. Jawab secukupnya sesuai pertanyaan terakhir user.
    3. Jangan overexplain, jangan seperti FAQ generator.
    4. Tetap terdengar seperti admin manusia.

    =====================================================
    ATURAN MENJAWAB
    =====================================================
    1. Prioritaskan KNOWLEDGE CONTEXT sebagai referensi utama.
    2. Gunakan SUMMARY CONTEXT sebagai riwayat percakapan.
    3. Jika knowledge sebagian tidak relevan (retrieval mismatch), ambil bagian yang relevan saja.
    4. Jika knowledge kosong atau tidak relevan:
    - Boleh gunakan pengetahuan umum tentang digital marketing.
    - TETAPI dilarang keras mengarang informasi krusial perusahaan
        (nama paket, harga spesifik, promo, syarat khusus, kontak, alamat, rekening).

    Jika user menanyakan info krusial yang tidak ada di knowledge:
    Gunakan respon ini:
    "Terima kasih atas pertanyaannya😊 Untuk memastikan informasi yang akurat, izin kami koordinasikan terlebih dahulu dengan tim terkait ya. Nanti akan segera kami informasikan kembali🙏"

    =====================================================
    BATASAN JAWABAN
    =====================================================
    - Default 2–5 kalimat.
    - Jangan kirim list panjang kecuali diminta.
    - Jangan jelaskan semua paket jika hanya ditanya satu.
    - Maksimal 2 emoticon ringan.
    - Tidak boleh menggunakan tanda seru (!).
    - Jangan gunakan markdown seperti **bold**, gunakan format WhatsApp jika perlu (*contoh*).

    =====================================================
    DETEKSI CATEGORY PRODUCT (WAJIB)
    =====================================================
    Tentukan produk yang dibahas berdasarkan prioritas berikut:
    1. USER MESSAGE (eksplisit)
    2. USER INTENT
    3. SUMMARY CONTEXT

    Kategori yang diperbolehkan:
    1. WEBSITE  
    (paket website, buat website, harga website, domain, hosting, fitur website, webmail/email bisnis, payment gateway, toko online, multibahasa, redesign website,
    only design, hapus copyright, beli putus, custom website)

    2. SEO  
    (optimasi website agar muncul di hasil pencarian Google, peningkatan ranking keyword, riset keyword, on-page SEO, technical SEO, optimasi konten,
    laporan performa, biaya & paket SEO)

    3. GOOGLE_ADS  
    (iklan Google, mengiklankan website di Google, pembuatan akun Google Ads, pemasangan tracking konversi Google Ads, penyetingan kampanye Google Ads)

    4. SOSMED_ADS  
    (Instagram Ads, Facebook Ads, TikTok Ads, YouTube Ads, optimasi iklan sosial media, meningkatkan penjualan/leads lewat iklan sosial media)

    5. COMPANY_PROFILE_PDF  
    (pembuatan company profile dalam bentuk dokumen/PDF, bukan website)

    6. SOSIAL_MEDIA_NON_ADS  
    (pembuatan akun IG/FB/YouTube, jasa kelola IG/FB/TikTok/YouTube, tambah followers IG, tambah viewers IG, tambah likes TikTok)

    7. LAYANAN_DIGITAL_LAINNYA  
    (artikel ID/EN, video promosi, desain banner, kartu nama, desain logo, LinkTree, proposal bisnis, promosi status WA, pembuatan Google Bisnis/Google Maps)
    
    Aturan:
    - Jika eksplisit menyebut sebuah produk misalnya website → WAJIB ["WEBSITE"]
    - Jika lebih dari satu → kembalikan dalam ARRAY contoh ["WEBSITE", "SEO"]
    - Jika hanya greeting tanpa konteks → null
    - Dilarang membuat kategori baru
    - Dilarang mengembalikan array kosong []

    =====================================================
    SELF-EVALUATION (WAJIB)
    =====================================================
    Tentukan:
    - "knowledge_relevant": true jika KNOWLEDGE CONTEXT benar-benar relevan dengan pertanyaan terakhir.
    - false jika kosong / mismatch / beda produk.
    - "confidence_score": angka 0.0–1.0 menunjukkan tingkat keyakinan terhadap jawaban final.
    - "force_optional_llm": true jika:
        • Jawaban membutuhkan informasi spesifik yang tidak tertulis jelas di KNOWLEDGE CONTEXT.
        • Pertanyaan menyangkut biaya tambahan, selisih harga, pengecualian paket, domain khusus (.co.id, dll), add-on, atau kondisi khusus.
        • KNOWLEDGE CONTEXT terlalu umum untuk memastikan jawaban 100% akurat.
    - false jika KNOWLEDGE CONTEXT sudah eksplisit dan pasti.

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
    OUTPUT (WAJIB JSON ONLY)
    =====================================================
    {{
    "response": "jawaban final yang ringkas dan natural",
    "category_product": null,
    "knowledge_relevant": true,
    "force_optional_llm": false,
    "confidence_score": 0.0
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
        logger.debug(f"[LLM1 RAW LENGTH] {len(raw)}")

        parsed = safe_parse_json(raw)
        if not parsed or "response" not in parsed:
            logger.warning("[LLM1] Invalid JSON output, using raw response")
            logger.debug(f"[LLM1 RAW FALLBACK] {raw}")
            return {
                "response": raw,
                "category_product": "__PARSING_ERROR__",
                "knowledge_relevant": False,
                "confidence_score": 0.0
            }
        logger.debug("[LLM1] JSON parsed successfully")
        return {
            "response": parsed.get("response", ""),
            "category_product": parsed.get("category_product"),
            "knowledge_relevant": parsed.get("knowledge_relevant", False),
            "force_optional_llm": parsed.get("force_optional_llm", False),
            "confidence_score": parsed.get("confidence_score", 0.0),
            "prompt": prompt,
            "raw_output": raw
        }
    except Exception:
        logger.exception("[LLM1 ERROR] Claude validate+generate failed")
        raise

def llm_optional_product_regenerate(
    user_message,
    user_intent,
    context_messages,
    category_product,
    previous_response
):
    context = ""
    if context_messages:
        context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in context_messages[-9:]
        )

    product_knowledge = get_product_knowledge(category_product)

    prompt = f"""
    KAMU adalah Admin Marketing profesional.

    Previous response TIDAK SESUAI dengan knowledge produk resmi.
    Tugas kamu adalah MEMPERBAIKI jawaban tersebut.

    =====================================================
    PRODUK YANG WAJIB DIFOKUSKAN
    =====================================================
    {category_product}

    =====================================================
    KNOWLEDGE RESMI PRODUK
    =====================================================
    {product_knowledge}

    =====================================================
    PREVIOUS RESPONSE DARI LLM1 (SALAH / TIDAK SESUAI)
    =====================================================
    {previous_response}

    =====================================================
    ATURAN WAJIB
    =====================================================

    1. Abaikan informasi yang tidak ada dalam knowledge resmi.
    2. Buat ulang jawaban agar:
    - Sesuai 100% dengan knowledge resmi
    - Tidak membahas produk lain
    - Tidak menambah informasi di luar knowledge
    3. Jangan mengarang harga, promo, atau detail teknis.
    4. Jawaban singkat dan natural.
    5. Maksimal 5 kalimat.
    6. Tidak menggunakan tanda seru.
    7. Maksimal 2 emoticon ringan.

    =====================================================
    INPUT TAMBAHAN
    =====================================================

    SUMMARY CONTEXT:
    {context}

    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    =====================================================
    OUTPUT (TEXT ONLY)
    =====================================================
    """

    logger.debug(f"[OPTIONAL LLM] prompt_length={len(prompt)}")
    logger.debug(f"[OPTIONAL LLM] category_product={category_product}")
    logger.debug(f"[OPTIONAL LLM] knowledge_length={len(product_knowledge) if product_knowledge else 0}")

    try:
        resp = claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        return {
            "response": raw,
            "prompt": prompt,
            "raw_output": raw
        }
    except Exception:
        logger.exception("[LLM OPTIONAL ERROR]")
        return None
    
def sanitize_llm_response(
    user_message: str,
    user_intent: str,
    context: str,
    raw_response: str,
    category_product=None
):
    logger.debug("[LLM2] Sanitizing response")
    logger.debug(f"[LLM2] user_intent={user_intent}")
    logger.debug(f"[LLM2] LLM1_response_preview='{raw_response[:200]}'")

    if category_product:
        if not isinstance(category_product, list):
            category_product = [category_product]

        price_context = get_price_context(category_product)
    else:
        price_context = "Tidak ada produk terdeteksi"

    logger.debug(f"[LLM2] category_product={category_product}")
    logger.debug(f"[LLM2] price_context_length={len(price_context)}")

    prompt = f"""
    Kamu adalah AI VALIDATOR & SANITIZER profesional.
    Posisi kamu adalah FINAL GATE sebelum jawaban dikirim ke user.

    TUGAS UTAMA:
    1) Validasi dan koreksi harga produk jika tidak sesuai dengan daftar harga resmi.
    2) Sanitasi data sensitif yang mungkin masih muncul pada response dari PREVIOUS LAYER.

    KAMU TIDAK BOLEH:
    Membuat jawaban baru/Menambah kalimat baru/Mengubah struktur kalimat/Mengubah gaya bahasa/Melakukan parafrase.

    KAMU HANYA BOLEH:
    - Mengganti angka harga yang salah.
    - Mengganti data sensitif dengan placeholder yang sesuai.

    ================================================
    ATURAN VALIDASI HARGA PRODUK
    ================================================

    - CATEGORY_PRODUCT menunjukkan produk yang sedang dibahas.
    - PRICE_CONTEXT adalah daftar harga resmi produk tersebut.
    - Gunakan PRICE_CONTEXT sebagai satu-satunya referensi harga yang benar.
    - Analisis teks pada FINAL RESPONSE (PREVIOUS LAYER).

    Jika FINAL RESPONSE menyebut angka harga:
    → Pastikan angka tersebut SAMA dengan harga resmi di PRICE_CONTEXT.
    → Jika berbeda, GANTI hanya angka harga yang salah.
    → Jangan mengubah kalimat lain.

    Jika harga sudah benar:
    → Jangan diubah.

    Jika tidak ada harga disebut:
    → Jangan menambahkan harga.

    Jika CATEGORY_PRODUCT null atau kosong:
    → Jangan lakukan validasi harga sama sekali.

    ================================================
    ATURAN SANITASI DATA SENSITIF
    ================================================

    Penggantian hanya dilakukan jika teks SECARA EKSPLISIT menyebut data internal berikut.
    Jangan mengganti berdasarkan asumsi.
    Jangan mengganti kata umum.

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

    FINAL RESPONSE (PREVIOUS LAYER):
    {raw_response}

    CATEGORY PRODUCT:
    {category_product}

    PRICE_CONTEXT:
    {price_context}

    ================================================
    OUTPUT (JSON ONLY)
    ================================================
    {{
    "response": "response hanya data sensitif yang diganti",
    "sensitive_found": true/false,
    "price_corrected": true/false
    }}
    """
    logger.debug(f"[LLM2] prompt_length={len(prompt)}")
    logger.debug(f"[LLM2] raw_response_length={len(raw_response) if raw_response else 0}")

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        raw = resp.choices[0].message.content.strip()
        logger.debug(f"[LLM2 RAW OUTPUT] {raw[:300]}")

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            logger.warning("[LLM2] No JSON detected, using LLM1 result")
            return {
                "response": raw_response,
                "sensitive_found": False,
                "price_corrected": False
            }

        parsed = safe_parse_json(match.group(0))
        if not parsed:
            logger.warning("[LLM2] JSON parse failed, fallback to LLM1")
            return {
                "response": raw_response,
                "sensitive_found": False,
                "price_corrected": False
            }

        response_text = parsed.get("response")
        sensitive_found = parsed.get("sensitive_found", False)
        price_corrected = parsed.get("price_corrected", False)

        logger.debug(f"[LLM2] sensitive_found={sensitive_found}")
        logger.debug(f"[LLM2] price_corrected={price_corrected}")

        # ==========================
        # LOGIC UTAMA
        # ==========================
        if response_text:
            logger.debug("[LLM2] Returning sanitized response")
            return {
                "response": response_text,
                "sensitive_found": sensitive_found,
                "price_corrected": price_corrected,
                "prompt": prompt,
                "raw_output": raw
            }

        logger.warning("[LLM2] Response empty, fallback to LLM1")
        return {
            "response": raw_response,
            "sensitive_found": False,
            "price_corrected": False
        }

    except Exception:
        logger.exception("[LLM2 ERROR] Sanitize layer failed")
        return {
            "response": raw_response,
            "sensitive_found": False,
            "price_corrected": False
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
    context_messages=None,
    session_category_product=None  
):
    user_vec = embed_text(user_message)
    best_user_node, metadata = iterative_node_search(
        user_vec,
        user_message,
        user_category,       
        prev_node_id,
        assistant_category 
    )

    # =========================================================
    # DETERMINE KNOWLEDGE CONTEXT
    # =========================================================
    if best_user_node:
        user_node_id = best_user_node[0]

        knowledge_data = get_response_from_knowledge(
            best_user_node_id=user_node_id,
            top_5_user_nodes=metadata.get("top_5_user_nodes", [])
        )

        detected_intent = NODES.get(user_node_id, {}).get("intent", "")
        knowledge_context = knowledge_data.get("knowledge_context", "")
        assistant_node_id = knowledge_data.get("assistant_node_id")

    else:
        detected_intent = user_intent
        knowledge_context = ""
        assistant_node_id = None

    # =========================================================
    # CALL LLM1
    # =========================================================
    llm_result = llm_validate_and_generate(
        user_message=user_message,
        user_intent=detected_intent,
        knowledge_context=knowledge_context,
        context_messages=context_messages
    )

    logger.debug(f"[LLM1 RESULT] keys={llm_result.keys()}")
    logger.debug(f"[LLM1 RESULT] response_preview={str(llm_result.get('response',''))[:100]}")

    raw_response = llm_result.get("response", "")
    detected_category_product = llm_result.get("category_product")
    llm1_prompt = llm_result.get("prompt")
    llm1_raw_output = llm_result.get("raw_output")
    optional_llm_raw_output = None

    # =========================================================
    # PRODUCT MEMORY LOGIC
    # =========================================================
    if detected_category_product == "__PARSING_ERROR__":
        # Parsing gagal → pakai memory lama
        final_category_product = session_category_product

    elif detected_category_product is None:
        # Tidak ada produk baru → pertahankan memory lama
        final_category_product = session_category_product

    elif isinstance(detected_category_product, list):
        final_category_product = detected_category_product

    else:
        # unexpected format → fallback aman
        final_category_product = session_category_product

    # =========================================================
    # OPTIONAL CALL LLM PRODUCT-AWARE REGENERATE
    # =========================================================
    used_optional_llm = False
    optional_llm_prompt = None

    knowledge_relevant = llm_result.get("knowledge_relevant")
    confidence_score = float(llm_result.get("confidence_score", 0.0))
    force_optional_llm = llm_result.get("force_optional_llm", False)

    if final_category_product and (
        force_optional_llm
        or knowledge_relevant is False
        or confidence_score < 0.5
    ):

        optional_response = llm_optional_product_regenerate(
            user_message=user_message,
            user_intent=detected_intent,
            context_messages=context_messages,
            category_product=final_category_product,
            previous_response=raw_response
        )

        if optional_response:
            raw_response = optional_response.get("response")
            optional_llm_prompt = optional_response.get("prompt")
            optional_llm_raw_output = optional_response.get("raw_output")
            used_optional_llm = True

    # =========================================================
    # CALL LLM2
    # =========================================================
    raw_response = str(raw_response) if raw_response is not None else ""

    sanitized = sanitize_llm_response(
        user_message=user_message,
        user_intent=detected_intent,
        context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
        raw_response=raw_response,
        category_product=final_category_product
    )

    logger.debug(
        f"[LLM2 FLAGS] sensitive_found={sanitized.get('sensitive_found')} | "
        f"price_corrected={sanitized.get('price_corrected')}"
    )

    final_response = sanitized.get("response", raw_response)
    final_response = final_response.replace("!", "")
    llm2_prompt = sanitized.get("prompt")
    llm2_raw_output = sanitized.get("raw_output")
    

    logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")
    logger.debug(f"[FINAL CATEGORY PRODUCT] {final_category_product}")
    logger.debug(
        f"[PIPELINE SUMMARY] "
        f"LLM1 -> OPTIONAL({used_optional_llm}) -> LLM2 | "
        f"confidence={confidence_score} | "
        f"category={final_category_product}"
    )
    # =========================================================
    # RETURN
    # =========================================================
    return {
        "response": final_response,
        "node_id": assistant_node_id,
        "metadata": metadata,
        "category_product": final_category_product,
        "knowledge_relevant": knowledge_relevant,
        "confidence_score": confidence_score,
        "used_optional_llm": used_optional_llm,
        "llm1_prompt": llm1_prompt,
        "optional_llm_prompt": optional_llm_prompt,
        "llm2_prompt": llm2_prompt,
        "knowledge_context": knowledge_context,
        "sensitive_found": sanitized.get("sensitive_found"),
        "price_corrected": sanitized.get("price_corrected"),
        "llm1_raw_output": llm1_raw_output,
        "optional_llm_raw_output": optional_llm_raw_output,
        "llm2_raw_output": llm2_raw_output
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

        # INTENT & FLOW CATEGORY (ONCE)
        user_intent, user_category = call_intent_and_category(
            user_message,
            "user",
            context_messages
        )

        logger.debug(f"[INTENT] {user_intent}")
        logger.debug(f"[FLOW CATEGORY] {user_category}")

        if reset:
            prev_node_id = None
            assistant_category = user_category
            session_data = {
                "history": [],
                "category_product": []
            }
            SESSION_STORE[session_id] = session_data
        else:
            prev_node_id = session_data.get("prev_node_id")
            assistant_category = session_data.get("assistant_category")

            if "category_product" not in session_data:
                session_data["category_product"] = []

        # SESSION CLEANUP
        if len(SESSION_STORE) > SESSION_STORE_MAX_SIZE:
            logger.debug("[SESSION CLEANUP] Max size exceeded, trimming oldest sessions")

            excess = len(SESSION_STORE) - SESSION_STORE_MAX_SIZE
            for key in list(SESSION_STORE.keys())[:excess]:
                del SESSION_STORE[key]

    # ======================
    # GENERATE RESPONSE
    # ======================
    session_category_product = session_data.get("category_product")

    result = generate_assistant_response(
        user_message=user_message,
        user_intent=user_intent,
        user_category=user_category,
        prev_node_id=prev_node_id,
        assistant_category=assistant_category,
        context_messages=context_messages,
        session_category_product=session_category_product,
    )

    final_response = result.get("response", "")
    assistant_node_id = result.get("node_id")
    metadata = result.get("metadata", {})
    metadata = metadata or {} 
    detected_category_product = result.get("category_product")
    knowledge_relevant = result.get("knowledge_relevant")
    confidence_score = result.get("confidence_score")
    used_optional_llm = result.get("used_optional_llm")
    llm1_prompt = result.get("llm1_prompt")
    optional_llm_prompt = result.get("optional_llm_prompt")
    llm2_prompt = result.get("llm2_prompt")
    llm1_raw_output = result.get("llm1_raw_output")
    optional_llm_raw_output = result.get("optional_llm_raw_output")
    llm2_raw_output = result.get("llm2_raw_output")
    knowledge_context = result.get("knowledge_context")
    sensitive_found = result.get("sensitive_found")
    price_corrected = result.get("price_corrected")
    

    assistant_node = NODES.get(assistant_node_id, {}) if assistant_node_id else {}

    # =================
    # UPDATE SESSION 
    # =================
    with SESSION_LOCK:

        history = session_data.get("history", [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": final_response})

        session_data["history"] = history[-9:]
        session_data["prev_node_id"] = assistant_node_id
        session_data["assistant_category"] = assistant_node.get("category") or user_category

        # =====================================
        # STATE MEMORY CATEGORY PRODUCT
        # =====================================
        last_category_product = session_data.get("category_product", [])

        if detected_category_product == "__PARSING_ERROR__":
            session_data["category_product"] = last_category_product
        elif detected_category_product:
            # Ada produk baru terdeteksi
            session_data["category_product"] = detected_category_product
        else:
            # Tidak ada produk baru → pertahankan memory lama
            session_data["category_product"] = last_category_product

        logger.debug(
            f"[PRODUCT MEMORY] detected={detected_category_product} | "
            f"previous={last_category_product} | "
            f"final={session_data.get('category_product')}"
        )
        SESSION_STORE[session_id] = session_data

    # ======================
    # DEBUG INFO
    # ======================
    debug_info = {
        "user_intent": user_intent,
        "user_category": user_category,

        # ROUTING
        "match_type": metadata.get("best_type"),
        "best_similarity": round(metadata.get("best_sim", 0), 3),
        "best_user_node_id": metadata.get("best_user_node_id"),
        "top_5_user_nodes": metadata.get("top_5_user_nodes", []),

        # RESULT
        "assistant_node_id": assistant_node_id,
        "assistant_intent": assistant_node.get("intent"),
        "assistant_category": assistant_node.get("category") or user_category,

        # CATEGORY PRODUCT
        "detected_category_product": detected_category_product,
        "session_category_product": session_data.get("category_product"),

        # COUNTS
        "flow_candidates_count": metadata.get("flow_count"),
        "global_candidates_count": metadata.get("global_count"),

        # CONTEXT SNAPSHOT
        "context_summary": " | ".join(
            f"{m['role']}: {m['content']}"
            for m in context_messages[-9:]
        ) if context_messages else None,
        "knowledge_context": knowledge_context,

        # LLM SELF EVALUATION
        "knowledge_relevant": knowledge_relevant,
        "confidence_score": confidence_score,
        "used_optional_llm": used_optional_llm,
 
        # LLM PROMPT
        "llm1_prompt": llm1_prompt,
        "optional_llm_prompt": optional_llm_prompt,
        "llm2_prompt": llm2_prompt,

        # LLM OUTPUT
        "llm1_raw_output": llm1_raw_output,
        "optional_llm_raw_output": optional_llm_raw_output,
        "llm2_raw_output": llm2_raw_output,

        #SANITIZE FLAGS
        "sensitive_found": sensitive_found,
        "price_corrected": price_corrected
    }

    logger.debug(f"[FINAL RESPONSE] {final_response}")
    logger.debug(f"[ASSISTANT NODE ID] {assistant_node_id}")
    logger.debug(f"[SESSION CATEGORY PRODUCT] {session_data.get('category_product')}")
    logger.debug(f"[SESSION END] session_id={session_id}")

    return {
        "response": final_response,
        "node_id": assistant_node_id,
        "debug": debug_info
    }
