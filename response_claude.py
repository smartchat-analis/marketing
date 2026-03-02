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

def trim_text_by_char(text, max_chars=15000):
    if not text:
        return text
    return text[:max_chars]

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
        custom_threshold=0.45
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
            for m in context_messages[-10:]
        )

    logger.debug("[LLM1] Preparing prompt for Claude")
    logger.debug(f"[LLM1] user_intent={user_intent}")
    logger.debug(f"[LLM1] knowledge_length={len(knowledge_context) if knowledge_context else 0}")
    logger.debug(f"[LLM1] context_length={len(context)}")

    prompt = f"""
    You are a WhatsApp Marketing Admin representing a digital agency.

    Your replies must sound natural, human, warm, and professional.
    Do NOT sound like a bot, FAQ generator, or AI assistant.

    Always respond in the same language as the user.
    If the user writes in Indonesian, respond in Indonesian.
    If the user writes in English, respond in English.

    =====================================================
    CONTEXT PRIORITY (ABSOLUTE ORDER)
    =====================================================

    You must process information in this exact order:

    1) USER MESSAGE (highest authority)
    2) SUMMARY CONTEXT (conversation continuity only)
    3) KNOWLEDGE CONTEXT (lowest authority, must be validated)

    USER MESSAGE always overrides everything.
    KNOWLEDGE CONTEXT is NEVER allowed to override USER MESSAGE.

    =====================================================
    ADMIN IDENTITY RULE (STRICT)
    =====================================================

    If the user ASKS your name:
    - If company context is EDA or Asain → introduce yourself as Aisyah.
    - If company context is EBYB → introduce yourself as Alesha.
    - If company origin is unknown → introduce yourself only as admin marketing {"{{$company}}"} (no personal name).

    If the user does NOT ask your name:
    - Do NOT introduce yourself.
    - Never mention any name outside these rules.

    =====================================================
    USER MESSAGE PRIORITY OVERRIDE (CRITICAL)
    =====================================================

    If the USER MESSAGE:
    - Does NOT explicitly mention a product category,
    - Does NOT mention a specific package,
    - Is a greeting or general inquiry,

    Then:

    - category_product = null
    - knowledge_relevant = false
    - force_optional_llm = false
    - IGNORE KNOWLEDGE CONTEXT completely.

    In this case:
    Respond naturally and ask clarification.
    Never let KNOWLEDGE CONTEXT decide the product
    if the USER MESSAGE does not explicitly mention one.

    =====================================================
    STRICT KNOWLEDGE VALIDATION (CRITICAL)
    =====================================================

    KNOWLEDGE CONTEXT is raw RAG output.
    It may contain low-similarity, mixed-package, or partially related nodes.
    Do NOT assume it is fully relevant.
    If KNOWLEDGE CONTEXT is EMPTY → treat as NOT relevant.

    For ANY service-related question (package, pricing, feature, process, timeline, domain, hosting, add-on, technical detail):

    You may answer using KNOWLEDGE CONTEXT ONLY IF ALL conditions are met:

    1. Exact product category matches.
    2. Exact package name mentioned by user exists explicitly.
    3. Exact requested detail exists explicitly.
    4. No inference, comparison, estimation, or assumption is needed.
    5. No different package name appears in the knowledge.

    Same category ≠ relevant.
    Different package ≠ relevant.

    If ANY condition fails:
        knowledge_relevant = false
        force_optional_llm = true
        Do NOT approximate.
        Do NOT fill gaps.
        Do NOT infer from other packages.

    If there is ANY doubt → treat as mismatch.

    -----------------------------------------------------

    SUMMARY CONTEXT:
    - Used only to understand conversation flow.
    - Helps resolve references like “itu” or “yang tadi”.
    - Cannot introduce new product assumptions.
    - Cannot override USER MESSAGE.

    =====================================================
    RESPONSE STYLE RULES
    =====================================================

    - Natural WhatsApp tone.
    - Not stiff.
    - Not brochure-style.
    - Not overly formal.
    - Not aggressive selling.
    - Do not re-ask clearly answered information.
    - Maximum 2 light emojis.
    - Avoid long bullet lists.
    - Do NOT explain features one by one unless explicitly requested.

    =====================================================
    PRODUCT CATEGORY DETECTION (MANDATORY LOGIC)
    =====================================================
    Determine category_product using this priority:
    1. USER MESSAGE
    2. USER INTENT
    3. SUMMARY CONTEXT

    Allowed categories:

    1. WEBSITE
    (paket website, buat website, harga website, domain, hosting, fitur website, webmail/email bisnis, payment gateway, toko online, multibahasa, redesign website,
    only design, hapus copyright, beli putus, custom website)

    2. SEO
    (optimasi website agar muncul di Google, ranking Google, riset keyword, on-page SEO, technical SEO, optimasi konten, laporan performa, biaya SEO)

    3. GOOGLE_ADS
    (iklan Google, akun Google Ads, tracking konversi, setting kampanye Google Ads)

    4. SOSMED_ADS
    (Instagram Ads, Facebook Ads, TikTok Ads, YouTube Ads, optimasi iklan sosial media)

    5. COMPANY_PROFILE_PDF
    (pembuatan company profile dalam bentuk dokumen/PDF, bukan website)

    6. SOSIAL_MEDIA_NON_ADS
    (pembuatan akun IG/FB/YouTube, kelola sosmed, tambah followers, viewers, likes)

    7. LAYANAN_DIGITAL_LAINNYA
    (artikel, video promosi, desain banner, kartu nama, logo, LinkTree, proposal bisnis, promosi WA, Google Bisnis/Maps)

    Rules:
    - If explicitly mentioned → must return array like ["WEBSITE"]
    - If multiple → ["WEBSITE", "SEO"]
    - If greeting only → null
    - Forbidden to create new categories
    - Forbidden to return empty array []

    =====================================================
    SELF-EVALUATION (STRICT ANTI-HALLUCINATION LOGIC)
    =====================================================
    You must evaluate logically, not optimistically.

    -----------------------------
    1) knowledge_relevant
    -----------------------------

    Set to TRUE ONLY IF ALL conditions below are met:
    - KNOWLEDGE CONTEXT matches the active product category.
    - The exact package or item asked by the user exists explicitly in KNOWLEDGE CONTEXT.
    - The specific detail requested by the user is explicitly written in KNOWLEDGE CONTEXT.
    - No assumption is required.

    Set to FALSE if ANY of the following occur:
    - KNOWLEDGE CONTEXT is EMPTY.
    - The package name is mentioned but its detailed features are NOT explicitly written.
    - The user asks about specific features but knowledge only contains other packages.
    - Knowledge is generic while the question is specific.
    - You would need to assume, estimate, or guess.
    - There is any product-category inconsistency.
    - You feel even slightly unsure.

    If there is doubt → set FALSE.

    -----------------------------
    2) force_optional_llm
    -----------------------------

    Set to TRUE if ANY of the following occur:
    - knowledge_relevant is FALSE.
    - The user asks about detailed package features but the exact feature list is not explicitly available.
    - The user asks about pricing adjustments, negotiation, or custom DP.
    - The user asks about domain extensions or add-ons not clearly written.
    - There is ambiguity between multiple packages.
    - The knowledge is partial, incomplete, or mixed between packages.
    - confidence_score would reasonably be below 0.80.

    When unsure → set TRUE.

    Set to FALSE ONLY if:
    - The package and its details are explicitly written.
    - No assumption is required.
    - Product context is fully consistent.
    - You are highly certain.

    -----------------------------
    3) confidence_score
    -----------------------------

    Provide a float between 0.0 and 1.0.

    0.90–0.99 → Exact package + exact detail clearly written.
    0.75–0.89 → Mostly clear, very minor uncertainty.
    0.50–0.74 → Partial detail available.
    Below 0.50 → Mismatch or assumption needed.

    If you had to assume any missing detail → score must be below 0.80.

    Never output 1.0.
    Never inflate confidence.

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
    {knowledge_context if knowledge_context else "(EMPTY)"}

    =====================================================
    OUTPUT (JSON ONLY)
    =====================================================
    {{
    "response": "final short and natural WhatsApp reply",
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
            "raw_output": parsed.get("response", raw)
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
            for m in context_messages[-10:]
        )

    product_knowledge = get_product_knowledge(category_product)

    prompt = f"""
    You are a WhatsApp Marketing Admin representing {"{{$company}}"}.  

    The previous response was NOT aligned with official product knowledge.  
    Your task is to correct and rewrite it so it becomes:

    - Fully accurate based ONLY on official knowledge.
    - Natural and human-like.
    - Short, clear, and persuasive.
    - Focused only on the active product.

    Always respond in the same language as the user.
    If the user writes in Indonesian, respond in Indonesian.
    If the user writes in English, respond in English.

    =====================================================
    INFORMATION PRIORITY (STRICT ORDER)
    =====================================================
    1. USER MESSAGE (highest authority)
    2. OFFICIAL PRODUCT KNOWLEDGE
    3. PREVIOUS RESPONSE (reference only for correction)

    If PREVIOUS RESPONSE contains any information
    that is NOT explicitly written in OFFICIAL PRODUCT KNOWLEDGE,
    you MUST remove it completely.

    =====================================================
    ACTIVE PRODUCT (STRICT FOCUS)
    =====================================================
    You MUST focus ONLY on:
    {category_product}

    Switching product or mentioning other services is strictly forbidden.

    If category_product contains multiple items,
    use knowledge strictly within each category.
    Do NOT mix details between categories.

    =====================================================
    OFFICIAL PRODUCT KNOWLEDGE
    =====================================================
    {product_knowledge}

    If OFFICIAL PRODUCT KNOWLEDGE is empty,
    you MUST use the coordination response.

    =====================================================
    PREVIOUS RESPONSE (INCORRECT / NEEDS FIX)
    =====================================================
    {previous_response}

    =====================================================
    ADDITIONAL CONTEXT
    =====================================================
    SUMMARY CONTEXT:
    {context}

    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    =====================================================
    MANDATORY RULES
    =====================================================
    1. Use ONLY information available in OFFICIAL PRODUCT KNOWLEDGE.
    2. Ignore any details not explicitly written in the knowledge.
    3. Do NOT invent pricing, promotions, technical details, add-ons, or policies.
    4. Answer ONLY the user's latest question.
    5. Maximum 3–6 WhatsApp lines (including line breaks).
    6. If the response feels long, rewrite it shorter.
    7. Do NOT overexplain.
    8. Do NOT list features one by one unless explicitly requested.
    9. Be slightly persuasive, not purely informational.
    10. Guide softly to next step when appropriate.
    11. Maximum 2 light emojis.
    12. Do NOT sound like a brochure or catalog.
    13. Do NOT repeat information already clear in context.

    If official knowledge is insufficient to answer with certainty,
    use this coordination response naturally:

    "Terima kasih atas pertanyaannya😊 Untuk memastikan informasi yang akurat, izin kami koordinasikan terlebih dahulu dengan tim terkait ya. Nanti akan segera kami informasikan kembali🙏"

    =====================================================
    OUTPUT
    =====================================================
    Return the final corrected WhatsApp reply as plain text only.
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
    logger.debug(f"[LLM2] Previouse_LLM_response_preview='{raw_response[:200]}'")

    if category_product:
        if not isinstance(category_product, list):
            category_product = [category_product]

        price_context = get_price_context(category_product)
    else:
        price_context = "Tidak ada produk terdeteksi"

    logger.debug(f"[LLM2] category_product={category_product}")
    logger.debug(f"[LLM2] price_context_length={len(price_context)}")

    prompt = f"""
    You are a professional AI VALIDATOR & SANITIZER.
    You are the FINAL GATE before the response is sent to the user.

    MAIN TASK:
    1) Validate and correct product pricing if it does not match the official price list.
    2) Sanitize sensitive data that may still appear in the response from the PREVIOUS LAYER.

    YOU ARE NOT ALLOWED TO:
    Create a new response / Add new sentences / Change sentence structure / Change writing style / Paraphrase.

    YOU ARE ONLY ALLOWED TO:
    - Replace incorrect price numbers.
    - Replace sensitive data with the appropriate placeholder.

    ================================================
    PRODUCT PRICE VALIDATION RULES
    ================================================
    - CATEGORY_PRODUCT indicates the product being discussed.
    - PRICE_CONTEXT is the official price list of that product.
    - Use PRICE_CONTEXT as the only valid pricing reference.
    - Analyze the text inside FINAL RESPONSE (PREVIOUS LAYER).

    If FINAL RESPONSE contains a price number:
    → Ensure the number EXACTLY matches the official price in PRICE_CONTEXT.
    → If different, REPLACE only the incorrect price number.
    → Do not modify any other part of the sentence.

    If the price is already correct:
    → Do not change anything.

    If no price is mentioned:
    → Do not add any price.

    If CATEGORY_PRODUCT is null or empty:
    → Do not perform any price validation.

    ================================================
    SENSITIVE DATA SANITIZATION RULES
    ================================================
    Replacement is only allowed if the text EXPLICITLY mentions the following internal data.
    Do not replace based on assumptions.
    Do not replace general words.

    1. Internal company name

    ONLY if the text explicitly and exactly mentions one of the following:
    - Asain
    - EDA
    - EbyB
    - PT. Asa Inovasi Software
    - PT. Eksa Digital Agency
    - PT. EBYB Global Marketplace

    Then it MUST be replaced with:
    {"{{$company}}"}

    If the text only contains general words such as:
    - perusahaan
    - company
    - bisnis
    - usaha
    - perusahaan kakak
    DO NOT replace them.

    -----------------------------------------------------------------------------

    2. Internal company address

    ONLY if it mentions the official address of internal companies (Asain, EDA, or EbyB).

    Not user address.
    Not general address.
    Not example address.

    If it explicitly mentions internal company address,
    it MUST be replaced with:
    {"{{$address}}"}

    -----------------------------------------------------------------------------

    3. Internal payment bank account number

    ONLY if it explicitly mentions internal company bank account numbers (Asain, EDA, or EbyB).

    Not user bank account.
    Not example account.
    Not merely bank name mention.

    If it explicitly mentions internal bank account number,
    it MUST be replaced with:
    {"{{$rekening_pembayaran}}"}

    ================================================
    INPUT
    ================================================
    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    CONTEXT:
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
    "response": "response with only sensitive data replaced if any",
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
                "raw_output": response_text
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

        knowledge_context = trim_text_by_char(knowledge_context, 15000)

        best_similarity = metadata.get("best_sim", 0)

        if best_similarity < 0.5:
            logger.debug(
                f"[RAG CONFIDENCE GATE] similarity={round(best_similarity,3)} < 0.5 → knowledge ignored"
            )
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
    llm1_output = llm_result.get("raw_output")
    optional_llm_output = None

    # =========================================================
    # PRODUCT MEMORY LOGIC
    # =========================================================

    # Pastikan session_category_product selalu list
    if not isinstance(session_category_product, list):
        session_category_product = (
            [session_category_product] if session_category_product else []
        )

    if detected_category_product == "__PARSING_ERROR__":
        # Parsing gagal → pakai memory lama
        final_category_product = session_category_product

    elif detected_category_product is None:
        # Tidak ada produk baru → pertahankan memory lama
        final_category_product = session_category_product

    elif isinstance(detected_category_product, list):

        # Merge tanpa duplikasi
        merged = session_category_product.copy()

        for cat in detected_category_product:
            if cat not in merged:
                merged.append(cat)

        final_category_product = merged

    else:
        # Unexpected format → fallback aman
        final_category_product = session_category_product

    # =========================================================
    # OPTIONAL CALL LLM PRODUCT-AWARE REGENERATE
    # =========================================================
    used_optional_llm = False
    optional_llm_prompt = None

    knowledge_relevant = llm_result.get("knowledge_relevant")
    confidence_score = float(llm_result.get("confidence_score", 0.0))
    force_optional_llm = llm_result.get("force_optional_llm", False)

    has_active_product_signal = (
        detected_category_product
        and detected_category_product != "__PARSING_ERROR__"
    )

    if (
        has_active_product_signal
        and final_category_product
        and (
            force_optional_llm
            or knowledge_relevant is False
            or confidence_score < 0.80
        )
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
            optional_llm_output = optional_response.get("raw_output")
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
    final_response = final_response.replace("!", ".")
    final_response = re.sub(r"\*{2,}(.*?)\*{2,}", r"*\1*", final_response)
    final_response = re.sub(r'\n+', '\n', final_response)
    final_response = re.sub(r'\.\s+([\U0001F300-\U0001FAFF])', r'\1', final_response)
    llm2_prompt = sanitized.get("prompt")
    llm2_output = sanitized.get("raw_output")

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
        "llm1_output": llm1_output,
        "optional_llm_output": optional_llm_output,
        "llm2_output": llm2_output
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
            max_messages=10
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
    llm1_output = result.get("llm1_output")
    optional_llm_output = result.get("optional_llm_output")
    llm2_output = result.get("llm2_output")
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

    # Gunakan hasil final dari generate_assistant_response
    session_data["category_product"] = result.get("category_product", [])

    logger.debug(
        f"[PRODUCT MEMORY FINALIZED] "
        f"stored={session_data.get('category_product')}"
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
            for m in context_messages[-10:]
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
        "llm1_output": llm1_output,
        "optional_llm_output": optional_llm_output,
        "llm2_output": llm2_output,

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
