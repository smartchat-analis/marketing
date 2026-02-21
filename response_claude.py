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
import anthropic
from list_product import CODE_PRODUCT
import sqlite3
from knowledge_website import KNOWLEDGE_WEBSITE
from knowledge_SEO import KNOWLEDGE_SEO
from knowledge_google_ads import KNOWLEDGE_GOOGLE_ADS
from knowledge_sosmed_ads import KNOWLEDGE_SOSMED_ADS
from knowlede_comprof import KNOWLEDGE_COMPROF
from knowledge_sosmed import KNOWLEDGE_SOSMED
from knowledge_layanan_digital import KNOWLEDGE_LAYANAN_DIGITAL

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
client = OpenAI(
    api_key="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"
)
claude_client = anthropic.Anthropic(
    api_key="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"
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

def get_top_priority_candidates(candidates, top_k=3):
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
    pembuatan website, SEO, google ads, sosmed ads, kelola sosmed, company profile pdf, pembuatan akun sosmed, pembuatan google maps, pembuatan email bisnis, dan
    layanan digital marketing lainnya.

    =====================================================
    IDENTITAS ADMIN (WAJIB DIPATUHI)
    =====================================================

    Jika klien MENANYAKAN nama kamu:
    - Jika konteks berasal dari perusahaan EDA atau Asain → jawab nama: Aisyah
    - Jika konteks berasal dari perusahaan EBYB → jawab nama: Alesha
    - Jika asal perusahaan tidak diketahui → perkenalkan diri sebagai admin marketing (tanpa nama spesifik)

    Aturan ketat:
    - Dilarang menyebut nama lain
    - Dilarang mengganti nama di luar aturan
    - Jika tidak ditanya nama → jangan memperkenalkan diri

    =====================================================
    PRIORITAS KONTROL KONTEKS
    =====================================================

    Urutan prioritas memahami konteks:

    1) SUMMARY CONTEXT
    2) USER MESSAGE terbaru
    3) KNOWLEDGE CONTEXT

    Jika KNOWLEDGE tidak relevan dengan SUMMARY atau USER MESSAGE, boleh diabaikan.

    Dilarang:
    - Mengganti topik layanan tanpa diminta
    - Menawarkan layanan lain di luar topik aktif
    - Mengambil knowledge yang tidak relevan
    - Melompat pembahasan

    =====================================================
    KONSISTENSI INFORMASI & ANTI-PENGULANGAN
    =====================================================

    Sebelum mengajukan pertanyaan atau meminta konfirmasi:

    - Periksa apakah informasi tersebut sudah disebutkan atau dikonfirmasi sebelumnya di SUMMARY CONTEXT atau riwayat percakapan.
    - Jika sudah pernah disebutkan atau dikonfirmasi oleh klien, DILARANG menanyakannya kembali.
    - Jangan meminta klien mengulang informasi yang sudah jelas.
    - Jangan mengulang pertanyaan yang sama dengan redaksi berbeda.
    - Jangan berpura-pura tidak tahu jika informasi sudah tersedia di konteks.

    Jika klien sudah:
    - Memilih paket
    - Menentukan layanan
    - Menyebut budget
    - Memberikan detail kebutuhan

    Maka gunakan informasi tersebut sebagai fakta yang sudah valid.
    Jangan konfirmasi ulang kecuali ada perubahan atau ketidakjelasan.

    Tujuan:
    Jawaban harus menunjukkan bahwa admin fokus, memperhatikan detail, dan memahami percakapan secara konsisten.

    =====================================================
    SISTEM MODE OTOMATIS (AKTIF JIKA RELEVAN)
    =====================================================

    Aktifkan mode sesuai topik berikut:

    • WEBSITE  
    (paket website, domain, hosting, fitur, webmail/email bisnis,
    payment gateway, toko online, multibahasa, redesign,
    only design, hapus copyright, beli putus, custom website)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_WEBSITE  
    → KNOWLEDGE_WEBSITE adalah sumber utama fitur, harga, aturan  
    → Jika konflik, ikuti KNOWLEDGE_WEBSITE  
    → Jangan jelaskan seluruh paket jika tidak diminta  

    • SEO  
    (optimasi website, ranking Google, keyword, SEO web dalam/luar, biaya SEO)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_SEO  
    → KNOWLEDGE_SEO adalah sumber utama sistem kerja, harga, kontrak  

    • GOOGLE ADS  
    (iklan Google, CPC, saldo harian, paket mingguan/bulanan, tracking konversi, setup akun & campaign)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_GOOGLE_ADS  
    → KNOWLEDGE_GOOGLE_ADS adalah sumber utama paket & aturan  

    • SOSMED ADS  
    (Instagram Ads, Facebook Ads, TikTok Ads, YouTube Ads, saldo harian, paket iklan, optimasi iklan sosial media)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_SOSMED_ADS  
    → Ikuti alur: Segmentasi → Edukasi → Ringkasan → Detail jika diminta  
    → Tidak bisa DP  

    • COMPANY PROFILE PDF  
    (pembuatan company profile dalam bentuk dokumen/PDF,harga paket, revisi, bonus logo/kartu nama)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_COMPROF  
    → Pastikan benar PDF, bukan website  
    → Jika website → alihkan ke MODE WEBSITE  

    • SOSIAL MEDIA NON-ADS  
    (pembuatan akun IG/FB/YouTube, halaman bisnis, jasa kelola IG/FB/TikTok/YouTube, tambah followers IG, viewers IG, TikTok likes/views)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_SOSMED  
    → Jika ternyata iklan → alihkan ke MODE SOSMED ADS  

    • LAYANAN DIGITAL LAINNYA  
    (artikel ID/EN, video promosi, desain banner, desain kartu nama, desain logo, LinkTree, proposal bisnis, promosi status WA, Google Maps)
    → Gunakan KNOWLEDGE_CONTEXT + KNOWLEDGE_LAYANAN_DIGITAL  
    → Pastikan bukan website atau layanan iklan  

    Jika terjadi perbedaan informasi → ikuti knowledge modul yang aktif.  
    Jika tidak ada di knowledge → gunakan fallback koordinasi tim.
    
    =====================================================
    GLOBAL RULE – CODE_PRODUCT
    =====================================================

    - CODE_PRODUCT hanya digunakan jika di knowledge secara eksplisit terdapat instruksi: (pakai CODE_PRODUCT "NAMA_CODE")
    - Jika tidak ada instruksi tersebut di knowledge, maka jangan gunakan placeholder.
    - Jangan mengarang atau mengubah nama code.
    - Jika tidak ada harga utama → jangan tampilkan CODE_PRODUCT.
    - Jika beberapa paket → setiap harga utama paket wajib memiliki CODE_PRODUCT masing-masing.
    - CODE_PRODUCT tidak boleh dijelaskan ke user.

    =====================================================
    DETEKSI PESAN RENDAH INFORMASI
    =====================================================

    Jika USER MESSAGE hanya berupa:
    - typo
    - koreksi singkat
    - emoji
    - respon pendek tanpa konteks bisnis

    Maka:
    - Jangan gunakan knowledge
    - Jangan bahas layanan
    - Balas natural sesuai konteks sebelumnya
    - Maksimal 2 kalimat

    =====================================================
    MODE KERJA UTAMA
    =====================================================

    A. Jika KNOWLEDGE tersedia dan relevan:
    - Gunakan hanya informasi dari KNOWLEDGE
    - Ambil semua poin relevan
    - Jangan menambah fakta
    - Jangan mengarang
    - Jangan keluar dari topik aktif
    - Boleh 1 pertanyaan klarifikasi jika memang diperlukan

    B. Jika KNOWLEDGE kosong:
    - Jangan menjawab detail layanan
    - Ajukan 1 pertanyaan klarifikasi

    Jika user menanyakan layanan/detail yang tidak tersedia:
    Gunakan fallback koordinasi tim.

    =====================================================
    ATURAN FORMAT KETAT
    =====================================================

    - Dilarang menggunakan tanda seru (!)
    - Jika muncul tanda seru, hapus sebelum output
    - Maksimal 2 emoticon ringan
    - Jangan huruf kapital berlebihan
    - Jangan promosi agresif
    - Jangan gunakan format markdown seperti **bold**
    - Jika ingin menebalkan teks, gunakan format WhatsApp:
        contoh: *Paket Platinum*

    =====================================================
    VALIDASI SEBELUM OUTPUT
    =====================================================

    Sebelum mengeluarkan jawaban, pastikan:
    1. Tidak ada tanda seru (!)
    2. Tidak keluar dari topik aktif
    3. Tidak mengulang pertanyaan yang sudah dikonfirmasi
    4. Tidak menambah asumsi di luar knowledge
    5. Jika MODE tertentu aktif → informasi mengikuti knowledge modul tersebut
    6. Jika pesan rendah informasi → tidak membahas layanan

    =================================
    VALIDASI HARGA & CODE_PRODUCT
    =================================

    PRINSIP DASAR:
    CODE_PRODUCT HANYA digunakan untuk harga yang secara eksplisit
    di knowledge memiliki instruksi: (pakai CODE_PRODUCT "NAMA_CODE")

    Jika di knowledge tidak ada instruksi tersebut,
    maka nominal TIDAK BOLEH diganti placeholder.

    -------------------------------------------------

    ATURAN PENGGANTIAN HARGA UTAMA:
    Jika suatu harga di knowledge memiliki instruksi (pakai CODE_PRODUCT "NAMA_CODE") maka:
    - Nominal harga WAJIB DIGANTI SEPENUHNYA dengan: {"{{$price:NAMA_CODE}}"}
    - Angka/nominal asli tidak boleh ditampilkan
    - Tidak boleh menampilkan angka + placeholder berdampingan
    - Placeholder menjadi satu-satunya representasi harga tersebut

    -------------------------------------------------

    ATURAN BREAKDOWN INTERNAL:
    Jika jawaban menampilkan nominal yang merupakan bagian dari breakdown internal, seperti:
    - Saldo iklan
    - Biaya per hari
    - Pajak
    - Fee
    - Registrasi
    - Akun VVIP
    - Biaya komponen lainnya
    DAN di knowledge TIDAK ada instruksi (pakai CODE_PRODUCT "NAMA_CODE") untuk nominal tersebut:
    Maka:
    - Jangan gunakan placeholder
    - Jangan mengganti angka
    - Tampilkan nominal apa adanya
    - Jangan membuat CODE_PRODUCT baru

    -------------------------------------------------

    JIKA MENAMPILKAN LEBIH DARI SATU PAKET:
    - Setiap harga utama paket wajib menggunakan CODE_PRODUCT masing-masing sesuai knowledge
    - Jangan menggunakan satu CODE_PRODUCT untuk seluruh angka dalam satu paket

    -------------------------------------------------

    JIKA TIDAK ADA INSTRUKSI CODE_PRODUCT DI KNOWLEDGE:
    - Jangan menyisipkan placeholder apa pun

    Sebelum mengeluarkan jawaban final, lakukan pengecekan ulang terhadap seluruh aturan di atas.
    Jika ditemukan pelanggaran, koreksi terlebih dahulu, baru tampilkan output final.

    =====================================================
    INPUT
    =====================================================

    SUMMARY CONTEXT:
    {context}

    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    DAFTAR CODE_PRODUCT VALID:
    {CODE_PRODUCT}

    =====================================================
    KNOWLEDGE CONTEXT (JIKA ADA)
    =====================================================

    {knowledge_context if knowledge_context else "(KOSONG)"}

    =====================================================
    KNOWLEDGE MODULES
    =====================================================

    --- KNOWLEDGE_WEBSITE ---
    {KNOWLEDGE_WEBSITE}

    --- KNOWLEDGE_SEO ---
    {KNOWLEDGE_SEO}

    --- KNOWLEDGE_GOOGLE_ADS ---
    {KNOWLEDGE_GOOGLE_ADS}

    --- KNOWLEDGE_SOSMED_ADS ---
    {KNOWLEDGE_SOSMED_ADS}

    --- KNOWLEDGE_SOSMED ---
    {KNOWLEDGE_SOSMED}

    --- KNOWLEDGE_COMPROF ---
    {KNOWLEDGE_COMPROF}

    --- KNOWLEDGE_LAYANAN_DIGITAL ---
    {KNOWLEDGE_LAYANAN_DIGITAL}

    =====================================================
    OUTPUT (JSON ONLY)
    =====================================================
    {{
    "response": "response lengkap dengan placeholder harga jika ada",
    "has_price": true/false
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
        
        # Parse JSON response
        parsed = safe_parse_json(raw)

        if not parsed or "response" not in parsed:
            logger.warning("[LLM1] Invalid JSON output, using raw response")
            logger.debug(f"[LLM1 RAW FALLBACK] {raw}")
            return {
                "response": raw,
                "has_price": False
            }

        # Check if response contains price placeholders
        has_price = "{$price:" in parsed.get("response", "") or "{{$price:" in parsed.get("response", "")
        
        logger.debug("[LLM1] JSON parsed successfully")
        logger.debug(f"[LLM1] has_price={has_price}")

        return {
            "response": parsed.get("response", ""),
            "has_price": has_price
        }

    except Exception:
        logger.exception("[LLM1 ERROR] Claude validate+generate failed")
        raise

def sanitize_llm_response(
    user_message: str,
    user_intent: str,
    context: str,
    final_response_llm1: str,
    detected_code_product: str | None
):
    logger.debug("[LLM2] Sanitizing response")
    logger.debug(f"[LLM2] user_intent={user_intent}")
    logger.debug(f"[LLM2] detected_code_product={detected_code_product}")
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
    - MENGGUNAKAN code_product yang SUDAH terdeteksi oleh LLM 1 untuk diaplikasikan ke placeholder {{$price:CODE_PRODUCT}}
    - TIDAK perlu mendeteksi code_product lagi

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

    -----------------------------------------------------------------------------

    4. Harga produk / layanan

    JANGAN mengganti harga jika di dalam teks sudah terdapat placeholder {{$price:...}}.

    Hanya lakukan penggantian jika:
    - Masih ada angka harga asli
    - Dan angka tersebut adalah harga utama paket
    - Dan detected_code_product tidak null

    Jangan mengganti angka yang merupakan bagian breakdown seperti:
    - pajak
    - fee
    - registrasi
    - saldo
    - biaya per hari
    - komponen lain

    ================================================
    CATATAN PENTING
    ================================================

    - code_product SUDAH dideteksi oleh LLM 1: {detected_code_product if detected_code_product else "TIDAK ADA"}
    - JIKA ADA code_product → gunakan placeholder {"{{$price:{detected_code_product}}}"}
    - JIKA TIDAK ADA code_product → cukup sanitize sensitive elements lain

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
    "response": "response dengan placeholder harga tetap utuh, hanya data sensitif diganti",
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

        raw = resp.choices[0].message.content
        logger.debug(f"[LLM2 RAW OUTPUT] {raw[:300]}")

        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            logger.warning("[LLM2] No JSON detected in response")
            raise ValueError("No JSON found")

        parsed = json.loads(match.group(0))

        logger.debug(
            f"[LLM2 RESULT] applied_code_product={parsed.get('applied_code_product')}"
        )

        # ================================
        # AMBIL RESPONSE TEXT
        # ================================
        response_text = parsed.get("response")

        # ================================
        # SAFETY CHECK – JANGAN RUSAK PLACEHOLDER
        # ================================

        # Jika placeholder sudah ada dari LLM1, jangan ubah apapun
        if "{{$price:" in final_response_llm1:
            logger.debug("[LLM2] Placeholder already exists from LLM1, skipping price modification")

            return {
                "response": final_response_llm1,
                "applied_code_product": detected_code_product
            }

        # Jika LLM2 menghasilkan response dengan placeholder (safety net)
        if response_text and "{{$price:" in response_text:
            logger.debug("[LLM2] Price placeholder applied by LLM2 safety net")

            return {
                "response": response_text,
                "applied_code_product": detected_code_product
            }

        # ==========================
        # NO SENSITIVE ELEMENT FOUND
        # ==========================
        if response_text is None:
            logger.debug("[LLM2] No sensitive element found")

            return {
                "response": final_response_llm1,
                "applied_code_product": detected_code_product
            }

        # Default: return LLM2 response dengan applied_code_product dari LLM1
        return {
            "response": response_text,
            "applied_code_product": detected_code_product
        }

    except Exception:
        logger.exception("[LLM2 ERROR] Sanitize layer failed")

        return {
            "response": final_response_llm1,
            "applied_code_product": detected_code_product
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
    assistant_category=None
):
    logger.debug("[ROUTING] Start find_best_user_node")

    candidates, flow_candidates, global_candidates = [], [], []
    threshold = dynamic_threshold(user_message)
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
    top_3 = candidates[:3]

    logger.debug(
        f"[ROUTING] best_node={best[0]} "
        f"sim={round(best[1],3)} "
        f"source={best[2]}"
    )
    logger.debug(f"[ROUTING] top_3={[(c[0], round(c[1],3), c[2]) for c in top_3]}")

    return best, {
        "flow_count": len(flow_candidates),
        "global_count": len(global_candidates),
        "best_sim": best[1],
        "best_type": best[2],
        "best_user_node_id": best[0],
        "top_3_user_nodes": [
            {
                "node_id": cid,
                "similarity": round(sim, 3),
                "source": src
            }
            for cid, sim, src in top_3
        ]
    }

def collect_assistant_knowledge_from_user_nodes(top_3_user_nodes):
    knowledge_chunks = []

    for item in top_3_user_nodes:
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
    top_3_user_nodes
):
    knowledge = collect_assistant_knowledge_from_user_nodes(
        top_3_user_nodes
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

    force_flow = (
        prev_node_id is not None
        and len(user_message.split()) <= 5
    )

    best_user_node, metadata = find_best_user_node(
        user_vec,
        user_message,
        user_category=user_category,
        prev_node_id=prev_node_id if force_flow else None,
        assistant_category=assistant_category
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
        logger.debug(f"[LLM1 RESULT] detected_code_product={llm_result.get('detected_code_product')}")

        # ============================================
        # FIX: Gunakan key yang benar
        # ============================================
        raw_response = llm_result.get("response", "")
        detected_code = llm_result.get("detected_code_product")

        sanitized = sanitize_llm_response(
            user_message=user_message,
            user_intent=detected_intent,
            context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
            final_response_llm1=raw_response,
            detected_code_product=detected_code  # FIX: parameter名称
        )

        final_response = sanitized.get("response", raw_response)
        applied_code = sanitized.get("applied_code_product")

        # ============================================
        # DEBUG: Log LLM2 result
        # ============================================
        logger.debug(f"[LLM2 RESULT] applied_code_product={applied_code}")
        logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")

        # Validasi
        if "{{price:" in final_response and not applied_code:
            raise ValueError("Price placeholder detected without product code")

        return {
            "response": final_response,
            "node_id": None,
            "metadata": metadata
        }

    # ==============================
    # ADA USER NODE
    # ==============================
    user_node_id = best_user_node[0]

    knowledge_data = get_response_from_knowledge(
        best_user_node_id=user_node_id,
        top_3_user_nodes=metadata["top_3_user_nodes"]
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
    logger.debug(f"[LLM1 RESULT] detected_code_product={llm_result.get('detected_code_product')}")

    # ============================================
    # FIX: Gunakan key yang benar
    # ============================================
    raw_response = llm_result.get("response", "")
    detected_code = llm_result.get("detected_code_product")

    # FINAL QUALITY CONTROL
    sanitized = sanitize_llm_response(
        user_message=user_message,
        user_intent=NODES.get(user_node_id, {}).get("intent", ""),
        context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
        final_response_llm1=raw_response,
        detected_code_product=detected_code  # FIX: parameter名称
    )

    final_response = sanitized.get("response", raw_response)
    final_response = final_response.replace("!", "")
    applied_code = sanitized.get("applied_code_product")

    # ============================================
    # DEBUG: Log LLM2 result
    # ============================================
    logger.debug(f"[LLM2 RESULT] applied_code_product={applied_code}")
    logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")

    # Validasi
    if "{{price:" in final_response and not applied_code:
        raise ValueError("Price placeholder detected without product code")

    return {
        "response": final_response,
        "node_id": knowledge_data["assistant_node_id"],
        "metadata": metadata
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

        # TOP 3 USER NODE AS KNOWLEDGE SOURCE
        "top_3_user_nodes": metadata.get("top_3_user_nodes", []),

        # RESULT
        "selected_user_node": metadata.get("best_user_node_id")
        if metadata.get("top_3_user_nodes") else None,

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
        ) if context_messages else None
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
