import os
import re
import json
import math
import threading
import logging
import requests
import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from collections import OrderedDict
from dotenv import load_dotenv
import anthropic
import sqlite3
from website_examples import maybe_build_examples_response
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
INTENT_EMB_PATH = "output/intent_embeddings.db"
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
SMARTCHAT_LATEST_CHAT_API = "https://smartchat2.edakarya.com/api/get-latest-chat"
SMARTCHAT_DETAIL_API = "https://smartchat2.edakarya.com/api/get-conversation-detail"
SMARTCHAT_TOKEN = "bduahdoawdwd9d9u308rf802f824hf8240h28gh8024g0824hg082h8"
SESSION_STORE = {}
SESSION_LOCK = threading.Lock()

# ======================
# LOAD DATA
# ======================
FLOW = None
EMB = None
NODES = None
LOADED = False
NODE_INTENT_EMB = None

def load_flow_and_embeddings():
    global NODES, NODE_INTENT_EMB, LOADED

    if LOADED:
        return

    if not os.path.exists(FLOW_PATH):
        raise FileNotFoundError("global_flow.db belum ada. Jalankan pipeline dulu.")

    if not os.path.exists(INTENT_EMB_PATH):
        raise FileNotFoundError("intent_embeddings.db belum ada.")

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
    # LOAD INTENT EMBEDDINGS
    # ===========================
    intent_db = get_db(INTENT_EMB_PATH)
    intent_cur = intent_db.cursor()

    intent_cur.execute("""
        SELECT node_id, embedding
        FROM intent_embeddings
    """)

    NODE_INTENT_EMB = {}

    for node_id, emb_str in intent_cur.fetchall():
        try:
            NODE_INTENT_EMB[node_id] = ast.literal_eval(emb_str)
        except Exception:
            continue

    intent_db.close()

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

def normalize_category_product(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    return []

def _normalize_price_number(raw_value):
    digits = re.sub(r"[^\d]", "", str(raw_value or ""))
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None

def _extract_negotiated_price(context_text):
    text = str(context_text or "")
    if not text.strip():
        return None

    patterns = [
        r"(?i)(?:jadi|menjadi|harga(?:nya)?\s+jadi|total(?:nya)?\s+jadi|boleh(?:nya)?|deal(?:nya)?|sepakat(?:nya)?|acc(?:\s+di)?|approved(?:\s+price)?)\s*(?:rp\.?\s*)?([\d\.\,]{3,})",
        r"(?i)(?:dari|harga\s+awal)\s*(?:rp\.?\s*)?[\d\.\,]{3,}\s*(?:jadi|ke|->|menjadi)\s*(?:rp\.?\s*)?([\d\.\,]{3,})",
        r"(?i)(?:diskon|nego(?:siasi)?)\b[\s\S]{0,40}?(?:rp\.?\s*)?([\d\.\,]{3,})"
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if not matches:
            continue
        for match in reversed(matches):
            price_value = _normalize_price_number(match)
            if price_value:
                return price_value
    return None

def _format_price_like_reference(price_value, reference_text):
    if not price_value:
        return None
    ref = str(reference_text or "")
    if "Rp" in ref or "rp" in ref:
        formatted = f"{price_value:,}".replace(",", ".")
        return f"Rp{formatted}"
    if "." in ref:
        return f"{price_value:,}".replace(",", ".")
    if "," in ref:
        return f"{price_value:,}".replace(",", ",")
    return str(price_value)

def _preserve_negotiated_price(response_text, negotiated_price):
    if not response_text or not negotiated_price:
        return response_text

    price_matches = list(re.finditer(r"(?i)(?:rp\.?\s*)?\d[\d\.\,]*", response_text))
    if not price_matches:
        return response_text

    normalized_matches = []
    for match in price_matches:
        value = _normalize_price_number(match.group(0))
        if value:
            normalized_matches.append((match, value))

    if not normalized_matches:
        return response_text

    if any(value == negotiated_price for _, value in normalized_matches):
        return response_text

    match_to_replace, _ = normalized_matches[-1]
    replacement = _format_price_like_reference(negotiated_price, match_to_replace.group(0))
    if not replacement:
        return response_text

    return (
        response_text[:match_to_replace.start()]
        + replacement
        + response_text[match_to_replace.end():]
    )

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
    threshold = 0.45

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
    context_messages=None,
    company=None
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

    =====================================================
    PAYMENT DETAILS SAFETY (CRITICAL)
    =====================================================
    If the user asks about payment or bank transfer details:
    - You MUST use ONLY the exact "Nomor rekening" and "Atas nama" text that exists in KNOWLEDGE CONTEXT.
    - You MUST NOT invent, guess, or reformat any bank account number.
    - You MUST NOT replace "Atas nama" with company context or placeholders.

    COMPANY MISMATCH RULE:
    - If the user clearly indicates they are an EDA client, do NOT provide Asain-only accounts.
    - If the user clearly indicates they are an Asain client, do NOT provide EDA-only accounts.
    - If there is any mismatch or doubt, always prioritize the 4 main EBYB accounts.
    - Alternative accounts are ONLY allowed when:
      1) The user is an EDA or Asain client, AND
      2) The user expresses doubt/concern about the account name.
    - If the user does NOT express doubt/concern, DO NOT include any alternative accounts at all.
    - If alternative accounts are allowed:
      - Include ONLY the one that matches the client company (Asain or EDA).
      - Never include both Asain and EDA accounts together.

    FALLBACK IF KNOWLEDGE CONTEXT HAS NO BANK ACCOUNT DETAILS:
    Use ONLY the following main payment content:

    Pembayaran dapat dilakukan ke salah satu rekening berikut:

    BANK MEGA  
    Nomor rekening: 01-351-00-16-00004-3  
    Atas nama: PT EBYB GLOBAL MARKETPLACE  

    BCA  
    Nomor rekening: 878-0532239  
    Atas nama: EBYB GLOBAL MARKETPLACE  

    MANDIRI  
    Nomor rekening: 118-00-1500440-0  
    Atas nama: PT EBYB GLOBAL MARKETPLACE  

    BRI  
    Nomor rekening: 050201000623569  
    Atas nama: EBYB GLOBAL MARKETPLACE  

    Jika membutuhkan invoice, dapat kami buatkan.
    Setelah melakukan pembayaran, mohon kirimkan bukti transfer agar bisa segera kami proses.

    If (and ONLY if) the user explicitly expresses doubt/concern about the account name,
    you may append ONE of the following alternative blocks that matches the client company:

    Asain client:
    BCA
    Nomor rekening: 03 7958 3999
    Atas nama: PT. ASA INOVASI SOFTWARE

    EDA client:
    BCA
    Nomor rekening: 099999 555 3
    Atas nama: PT EKSA DIGITAL AGENCY

    Always keep the 4 main accounts as the primary payment option.

    ===============================
    SUMMARY CONTEXT:
    ===============================
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
    - If the user asks for contoh/portfolio/reference website, set wants_examples = true.
    - If the user does not ask for website examples, set wants_examples = false.

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
    COMPANY CONTEXT:
    {company if company else "(UNKNOWN)"}

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
    "wants_examples": false,
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
            "wants_examples": parsed.get("wants_examples", False),
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
    previous_response,
    company=None
):
    context = ""
    if context_messages:
        context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in context_messages[-10:]
        )

    product_knowledge = get_product_knowledge(category_product)

    prompt = f"""
    You are a WhatsApp Marketing Admin representing {company if company else "(UNKNOWN COMPANY)"}.

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
    COMPANY CONTEXT:
    {company if company else "(UNKNOWN)"}

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
    category_product=None,
    company=None,
    negotiated_price=None
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
    logger.debug(f"[LLM2] negotiated_price={negotiated_price}")

    negotiated_price_text = negotiated_price if negotiated_price is not None else "(NONE)"

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
    
    EXCEPTION FOR NEGOTIATED PRICES:
    If the CONTEXT explicitly indicates a negotiated/approved price,
    you MUST allow that exact price to remain in FINAL RESPONSE even if it differs from PRICE_CONTEXT.

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

    EXCEPTION: If the company name appears within payment details
    (lines containing "Nomor rekening", "Atas nama", or "Rekening"),
    DO NOT replace it. Keep the original text exactly as written.

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
    COMPANY CONTEXT:
    {company if company else "(UNKNOWN)"}

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

    NEGOTIATED_PRICE:
    {negotiated_price_text}

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

        if response_text and negotiated_price is not None:
            preserved_response = _preserve_negotiated_price(response_text, negotiated_price)
            if preserved_response != response_text:
                response_text = preserved_response
                price_corrected = True

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
        raw = (resp.choices[0].message.content or "").strip()
        parsed = safe_parse_json(raw)
        if not parsed:
            match = re.search(r"\{[\s\S]*\}", raw)
            parsed = safe_parse_json(match.group(0)) if match else None

        if not isinstance(parsed, dict):
            logger.warning("[INTENT CLASSIFIER] invalid JSON output, fallback default")
            return "lainnya", None

        intent = parsed.get("intent", "lainnya")
        category = parsed.get("category", None)
        return intent, category
    except Exception as e:
        logging.error(f"Intent and category classification failed: {e}")
        return "lainnya", None

def all_intent_and_category(user_message, role, context_messages=None):
    return call_intent_and_category(user_message, role, context_messages or [])

def get_assistant_candidates_from_user_node(user_node_id):
    node = NODES.get(user_node_id, {})
    if not node or node.get("role") != "user":
        return []

    candidates = []
    seen = set()
    for _, edges in node.get("answers", {}).items():
        for e in edges:
            assistant_node_id = e.get("to")
            if not assistant_node_id or assistant_node_id in seen:
                continue
            assistant_node = NODES.get(assistant_node_id, {})
            if assistant_node.get("role") != "assistant":
                continue
            seen.add(assistant_node_id)
            candidates.append({
                "assistant_node_id": assistant_node_id,
                "assistant_intent": assistant_node.get("intent"),
                "assistant_category": assistant_node.get("category"),
                "texts": sorted(
                    assistant_node.get("texts", []),
                    key=lambda t: t.get("priority", 0),
                    reverse=True
                )
            })
    return candidates

def score_assistant_candidate_with_llm(user_message, user_intent, candidate, context_messages=None):
    assistant_node_id = candidate.get("assistant_node_id")
    assistant_intent = candidate.get("assistant_intent", "")
    assistant_category = candidate.get("assistant_category", "")
    assistant_texts = [
        t.get("chat")
        for t in candidate.get("texts", [])[:5]
        if t.get("chat")
    ]
    context_block = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in (context_messages or [])[-10:]
    ) or "(tidak ada konteks)"
    assistant_text_block = "\n".join(f"- {txt}" for txt in assistant_texts) or "(kosong)"

    prompt = f"""
    Kamu adalah evaluator routing assistant node.

    Tugas: beri skor kecocokan kandidat assistant node terhadap kebutuhan user saat ini.

    USER MESSAGE:
    {user_message}

    USER INTENT:
    {user_intent}

    CONTEXT MESSAGES:
    {context_block}

    ASSISTANT NODE ID:
    {assistant_node_id}

    ASSISTANT INTENT:
    {assistant_intent}

    ASSISTANT CATEGORY:
    {assistant_category}

    ASSISTANT TEXTS:
    {assistant_text_block}

    KRITERIA PENILAIAN:
    1) Kesesuaian dengan USER MESSAGE (40%)
    2) Kesesuaian dengan USER INTENT (30%)
    3) Kesesuaian dengan CONTEXT MESSAGES (20%)
    4) Kualitas isi ASSISTANT TEXTS (10%)

    ATURAN PENALTI:
    - Jika kandidat membahas produk/layanan berbeda dari kebutuhan user -> maksimal skor 0.35.
    - Jika kandidat hanya salam/closing tanpa menjawab kebutuhan user -> maksimal skor 0.25.
    - Jika kandidat bertentangan jelas dengan konteks terbaru -> maksimal skor 0.20.

    OUTPUT JSON SAJA:
    {{
      "score": 0.0,
      "reason": "alasan singkat 1 kalimat"
    }}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = safe_parse_json(raw)
        if not parsed:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = safe_parse_json(match.group(0)) if match else None
        score = float((parsed or {}).get("score", 0.0))
        score = max(0.0, min(score, 1.0))
        reason = (parsed or {}).get("reason", "")
        return {
            **candidate,
            "score": score,
            "score_reason": reason
        }
    except Exception as e:
        logger.warning(f"[ASSISTANT RANKER] node={assistant_node_id} failed: {e}")
        return {
            **candidate,
            "score": 0.0,
            "score_reason": "ranker_failed"
        }

def select_best_assistant_node_parallel(user_message, user_intent, candidates, context_messages=None):
    if not candidates:
        return None, []

    if len(candidates) == 1:
        only = dict(candidates[0])
        only["score"] = 1.0
        only["score_reason"] = "single_candidate"
        return only, [only]

    max_workers = min(5, len(candidates))
    scored = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                score_assistant_candidate_with_llm,
                user_message,
                user_intent,
                candidate,
                context_messages
            )
            for candidate in candidates
        ]
        for future in as_completed(futures):
            scored.append(future.result())

    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored[0], scored

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
    threshold = custom_threshold if custom_threshold is not None else 0.65
    flow_candidates = []
    global_candidates = []

    logger.debug(f"[ROUTING] threshold={threshold}")
    logger.debug(f"[ROUTING] prev_node_id={prev_node_id}")

    if prev_node_id and prev_node_id in NODES:
        prev_node = NODES[prev_node_id]
        for _, edges in prev_node.get("answers", {}).items():
            for e in edges:
                node_id = e.get("to")
                node = NODES.get(node_id)
                if not node or node.get("role") != "user":
                    continue

                emb = NODE_INTENT_EMB.get(node_id)
                if not emb:
                    continue

                sim = cosine_similarity(user_vec, emb)
                if sim >= threshold:
                    flow_candidates.append({
                        "node_id": node_id,
                        "similarity": sim,
                        "source": "flow"
                    })

    flow_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    if flow_candidates:
        best = flow_candidates[0]
        return (best["node_id"], best["similarity"], best["source"]), {
            "flow_count": len(flow_candidates),
            "global_count": 0,
            "best_sim": best["similarity"],
            "best_type": best["source"],
            "best_user_node_id": best["node_id"],
            "top_5_user_nodes": [
                {
                    "node_id": c["node_id"],
                    "similarity": round(c["similarity"], 3),
                    "source": c["source"]
                }
                for c in flow_candidates[:5]
            ]
        }

    for node_id, node in NODES.items():
        if node.get("role") != "user":
            continue

        emb = NODE_INTENT_EMB.get(node_id)
        if not emb:
            continue

        sim = cosine_similarity(user_vec, emb)
        if sim >= threshold:
            global_candidates.append({
                "node_id": node_id,
                "similarity": sim,
                "source": "global"
            })

    global_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    if global_candidates:
        best = global_candidates[0]
        return (best["node_id"], best["similarity"], best["source"]), {
            "flow_count": 0,
            "global_count": len(global_candidates),
            "best_sim": best["similarity"],
            "best_type": best["source"],
            "best_user_node_id": best["node_id"],
            "top_5_user_nodes": [
                {
                    "node_id": c["node_id"],
                    "similarity": round(c["similarity"], 3),
                    "source": c["source"]
                }
                for c in global_candidates[:5]
            ]
        }

    return None, {
        "flow_count": 0,
        "global_count": 0,
        "best_sim": 0.0,
        "best_type": "no_match",
        "best_user_node_id": None,
        "top_5_user_nodes": []
    }

def collect_assistant_knowledge_from_user_nodes(top_5_user_nodes):
    knowledge_chunks = []
    assistant_candidates = []
    seen_candidates = set()

    for item in top_5_user_nodes:
        user_node_id = item["node_id"]
        node = NODES.get(user_node_id)

        if not node or node["role"] != "user":
            continue

        for _, edges in node.get("answers", {}).items():
            for e in edges:
                aid = e["to"]
                if aid in NODES and NODES[aid]["role"] == "assistant":
                    if aid not in seen_candidates:
                        assistant_candidates.append({
                            "assistant_node_id": aid,
                            "from_user_node_id": user_node_id,
                            "similarity": item.get("similarity"),
                            "source": item.get("source")
                        })
                        seen_candidates.add(aid)
                    for t in NODES[aid].get("texts", []):
                        knowledge_chunks.append({
                            "assistant_node_id": aid,
                            "text": t["chat"]
                        })

    return knowledge_chunks, assistant_candidates

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
    knowledge, assistant_candidates = collect_assistant_knowledge_from_user_nodes(
        top_5_user_nodes
    )
    best_assistant_node_id = resolve_assistant_node_from_best_user(best_user_node_id)

    if not knowledge:
        return {
            "assistant_node_id": None,
            "best_assistant_node_id": best_assistant_node_id,
            "assistant_candidates": assistant_candidates,
            "knowledge_context": ""
        }

    knowledge_context = "\n".join(
        f"- {k['text']}"
        for k in knowledge
    )

    return {
        "assistant_node_id": best_assistant_node_id,
        "best_assistant_node_id": best_assistant_node_id,
        "assistant_candidates": assistant_candidates,
        "knowledge_context": knowledge_context
    }

def generate_assistant_response(
    user_message,
    user_intent=None,
    user_category=None,
    prev_node_id=None,
    assistant_category=None,
    context_messages=None,
    session_category_product=None,
    company=None
):
    if not isinstance(session_category_product, list):
        session_category_product = (
            [session_category_product] if session_category_product else []
        )

    intent_vec = embed_text(user_intent)
    best_user_node, metadata = find_best_user_node(
        intent_vec,
        user_message=user_message,
        user_category=user_category,
        prev_node_id=prev_node_id,
        assistant_category=assistant_category,
        custom_threshold=0.65
    )

    detected_intent = user_intent
    knowledge_context = ""
    assistant_node_id = None
    best_assistant_node_id = None
    assistant_candidates = []
    llm1_prompt = None
    llm1_output = None
    optional_llm_prompt = None
    optional_llm_output = None
    detected_category_product = None
    knowledge_relevant = False
    confidence_score = 0.0
    used_optional_llm = False
    raw_response = ""
    final_category_product = list(session_category_product)
    force_optional_llm = False
    force_optional_for_no_user_node = False
    # =========================================================
    # DETERMINE KNOWLEDGE CONTEXT
    # =========================================================
    if best_user_node:
        user_node_id = best_user_node[0]
        logger.debug(
            f"[ROUTING] matched_user_node={user_node_id} "
            f"sim={round(float(best_user_node[1]),3)} "
            f"source={best_user_node[2]}"
        )
        detected_intent = NODES.get(user_node_id, {}).get("intent", user_intent)
        resolved_from_best_user = resolve_assistant_node_from_best_user(user_node_id)

        assistant_candidates = get_assistant_candidates_from_user_node(user_node_id)
        logger.debug(f"[ASSISTANT CANDIDATES] count={len(assistant_candidates)}")
        best_assistant_candidate = None
        scored_assistants = []
        if len(assistant_candidates) == 1:
            best_assistant_candidate = dict(assistant_candidates[0])
            best_assistant_candidate["score"] = 1.0
            best_assistant_candidate["score_reason"] = "single_candidate"
            scored_assistants = [best_assistant_candidate]
            logger.debug("[ASSISTANT CANDIDATES] single candidate, skip parallel ranker")
        elif len(assistant_candidates) > 1:
            best_assistant_candidate, scored_assistants = select_best_assistant_node_parallel(
                user_message=user_message,
                user_intent=detected_intent,
                candidates=assistant_candidates,
                context_messages=context_messages
            )
        metadata["assistant_candidate_scores"] = [
            {
                "assistant_node_id": c.get("assistant_node_id"),
                "assistant_intent": c.get("assistant_intent"),
                "assistant_category": c.get("assistant_category"),
                "score": round(float(c.get("score", 0.0)), 3),
                "score_reason": c.get("score_reason")
            }
            for c in scored_assistants
        ]
        best_similarity = metadata.get("best_sim", 0)
        if best_assistant_candidate:
            logger.debug(
                f"[RAG CONFIDENCE GATE] similarity={round(best_similarity,3)} < 0.5 → knowledge ignored"
            )
            assistant_node_id = best_assistant_candidate.get("assistant_node_id")
            best_assistant_node_id = assistant_node_id
            logger.debug(
                f"[ASSISTANT SELECTED] node={assistant_node_id} "
                f"score={round(float(best_assistant_candidate.get('score', 0.0)),3)} "
                f"reason={best_assistant_candidate.get('score_reason', '')}"
            )
            best_texts = [
                t.get("chat")
                for t in best_assistant_candidate.get("texts", [])
                if t.get("chat")
            ]
            knowledge_context = "\n".join(f"- {txt}" for txt in best_texts)
            knowledge_context = trim_text_by_char(knowledge_context, 15000)

        elif resolved_from_best_user:
            assistant_node_id = resolved_from_best_user
            best_assistant_node_id = assistant_node_id
            fallback_node = NODES.get(assistant_node_id, {})
            fallback_texts = sorted(
                fallback_node.get("texts", []),
                key=lambda t: t.get("priority", 0),
                reverse=True
            )
            best_texts = [
                t.get("chat")
                for t in fallback_texts
                if t.get("chat")
            ]
            knowledge_context = "\n".join(f"- {txt}" for txt in best_texts)
            knowledge_context = trim_text_by_char(knowledge_context, 15000)
            metadata["assistant_resolved_fallback"] = True
            logger.debug(f"[ASSISTANT FALLBACK RESOLVE] node={assistant_node_id}")
    else:
        logger.debug("[ROUTING] No user node matched threshold 0.65")
        metadata.setdefault("assistant_candidate_scores", [])

    # =========================================================
    # FILTER KNOWLEDGE CONTEXT BY COMPANY (PAYMENT INFO)
    # =========================================================
    if company and knowledge_context:
        bank_marker = re.search(r"(?i)\b(atas nama|nomor rekening)\b", knowledge_context)
        if bank_marker:
            company_tokens = _normalize_company_tokens(company)
            blocks = _split_knowledge_blocks(knowledge_context)
            kept_blocks = []
            for block in blocks:
                if re.search(r"(?i)\b(atas nama|nomor rekening)\b", block):
                    atas_nama_match = re.search(r"(?i)atas nama\s*:\s*(.+)", block)
                    atas_nama_text = atas_nama_match.group(1) if atas_nama_match else block
                    atas_tokens = _normalize_company_tokens(atas_nama_text)
                    is_ebyb = "ebyb" in atas_tokens
                    has_company_overlap = bool(company_tokens.intersection(atas_tokens))
                    if not (has_company_overlap or is_ebyb):
                        continue
                kept_blocks.append(block)
            knowledge_context = "\n- ".join(kept_blocks)
            if knowledge_context:
                knowledge_context = f"- {knowledge_context}"
            else:
                logger.debug("[PAYMENT FILTER] knowledge_context cleared due to company mismatch")

    # =========================================================
    # CALL LLM1
    # =========================================================
    llm_result = llm_validate_and_generate(
        user_message=user_message,
        user_intent=detected_intent,
        knowledge_context=knowledge_context,
        context_messages=context_messages,
        company=company
    )

    logger.debug(f"[LLM1 RESULT] keys={llm_result.keys()}")
    logger.debug(f"[LLM1 RESULT] response_preview={str(llm_result.get('response',''))[:100]}")

    raw_response = llm_result.get("response", "")
    wants_examples = bool(llm_result.get("wants_examples", False))
    detected_category_product = llm_result.get("category_product")
    llm1_prompt = llm_result.get("prompt")
    llm1_output = llm_result.get("raw_output")
    optional_llm_output = None
    examples_replaced = False

    # =========================================================
    # REPLACE WITH DB EXAMPLES (POST-LLM1)
    # =========================================================
    context_summary = " | ".join(
        f"{m['role']}: {m['content']}" for m in (context_messages or [])
    )
    negotiated_price = _extract_negotiated_price(context_summary)
    example_result = maybe_build_examples_response(
        user_message=user_message,
        context_summary=context_summary,
        llm1_output=raw_response,
        company=company,
        wants_examples=wants_examples
    )
    if example_result and example_result.get("response"):
        raw_response = example_result["response"]
        examples_replaced = True

    # =========================================================
    # PRODUCT MEMORY LOGIC
    # =========================================================

    # Pastikan session_category_product selalu list
    if not isinstance(session_category_product, list):
        session_category_product = (
            [session_category_product] if session_category_product else []
        )

    detected_list = normalize_category_product(detected_category_product)
    if detected_category_product == "__PARSING_ERROR__":
        # Parsing gagal → pakai memory lama
        final_category_product = list(session_category_product)

    elif detected_list:
        # Tidak ada produk baru → pertahankan memory lama
        merged = list(session_category_product)

        for cat in detected_list:
            if cat not in merged:
                merged.append(cat)

        final_category_product = merged

    else:
        # Unexpected format → fallback aman
        final_category_product = list(session_category_product)

    # =========================================================
    # OPTIONAL CALL LLM PRODUCT-AWARE REGENERATE
    # =========================================================
    used_optional_llm = False
    optional_llm_prompt = None

    knowledge_relevant = llm_result.get("knowledge_relevant")
    confidence_score = float(llm_result.get("confidence_score", 0.0))
    force_optional_llm = llm_result.get("force_optional_llm", False)

    has_active_product_signal = len(final_category_product) > 0
    if not best_user_node and has_active_product_signal:
        force_optional_for_no_user_node = True

    optional_trigger_reasons = []
    if force_optional_llm:
        optional_trigger_reasons.append("llm1_force_optional")
    if force_optional_for_no_user_node:
        optional_trigger_reasons.append("no_user_node_with_product")
    if knowledge_relevant is False:
        optional_trigger_reasons.append("knowledge_not_relevant")
    if confidence_score < 0.80:
        optional_trigger_reasons.append("low_confidence")
    if not assistant_node_id:
        optional_trigger_reasons.append("assistant_node_missing")

    if (
        has_active_product_signal
        and not examples_replaced
        and (
            force_optional_llm
            or force_optional_for_no_user_node
            or knowledge_relevant is False
            or confidence_score < 0.80
            or not assistant_node_id
        )
    ):

        optional_response = llm_optional_product_regenerate(
            user_message=user_message,
            user_intent=detected_intent,
            context_messages=context_messages,
            category_product=final_category_product,
            previous_response=raw_response,
            company=company
        )
        logger.debug(
            f"[OPTIONAL LLM] triggered reasons={optional_trigger_reasons} "
            f"category={final_category_product}"
        )

        if optional_response:
            raw_response = optional_response.get("response")
            optional_llm_prompt = optional_response.get("prompt")
            optional_llm_output = optional_response.get("raw_output")
            used_optional_llm = True
    else:
        logger.debug(
            f"[OPTIONAL LLM] skipped has_product={has_active_product_signal} "
            f"reasons={optional_trigger_reasons}"
        )

    # =========================================================
    # CALL LLM2
    # =========================================================
    raw_response = str(raw_response) if raw_response is not None else ""

    sanitized = sanitize_llm_response(
        user_message=user_message,
        user_intent=detected_intent,
        context=" | ".join(m["content"] for m in context_messages) if context_messages else "",
        raw_response=raw_response,
        category_product=final_category_product,
        company=company,
        negotiated_price=negotiated_price
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
    final_response = re.sub(r'[ \t]+([\U0001F300-\U0001FAFF])', r'\1', final_response)
    final_response = _restore_atas_nama_from_prompt_fallback(final_response)
    llm2_prompt = sanitized.get("prompt")
    llm2_output = sanitized.get("raw_output")

    logger.debug(f"[LLM2 RESULT] response_preview={str(final_response)[:100]}")
    logger.debug(f"[FINAL CATEGORY PRODUCT] {final_category_product}")
    logger.debug(
        f"[PIPELINE SUMMARY] "
        f"ROUTING_V2 -> LLM1/OPTIONAL -> LLM2 | "
        f"confidence={confidence_score} | "
        f"category={final_category_product}"
    )

    resolved_assistant_intent = None
    resolved_assistant_category = None

    if not assistant_node_id:
        assistant_context = list(context_messages or [])
        assistant_context.append({
            "role": "user",
            "content": user_message
        })
        resolved_assistant_intent, resolved_assistant_category = all_intent_and_category(
            final_response,
            "assistant",
            assistant_context
        )

        logger.debug(
            f"[ASSISTANT FALLBACK CLASSIFIER] "
            f"intent={resolved_assistant_intent} | "
            f"category={resolved_assistant_category}"
        )

        if not resolved_assistant_intent:
            resolved_assistant_intent = "assistant_generated_response"
        if not resolved_assistant_category:
            resolved_assistant_category = user_category or "conversational_closing"

    return {
        "response": final_response,
        "node_id": assistant_node_id,
        "best_assistant_node_id": best_assistant_node_id,
        "assistant_candidates": metadata.get("assistant_candidate_scores", []),
        "metadata": metadata,
        "category_product": normalize_category_product(final_category_product),
        "knowledge_relevant": knowledge_relevant,
        "confidence_score": confidence_score,
        "used_optional_llm": used_optional_llm,
        "llm1_prompt": llm1_prompt,
        "optional_llm_prompt": optional_llm_prompt,
        "llm2_prompt": llm2_prompt,
        "knowledge_context": knowledge_context,
        "sensitive_found": sanitized.get("sensitive_found"),
        "price_corrected": sanitized.get("price_corrected"),
        "wants_examples": wants_examples,
        "website_examples_used": examples_replaced,
        "negotiated_price_detected": negotiated_price is not None,
        "negotiated_price_value": negotiated_price,
        "llm1_output": llm1_output,
        "optional_llm_output": optional_llm_output,
        "llm2_output": llm2_output,
        "resolved_assistant_intent": resolved_assistant_intent,
        "resolved_assistant_category": resolved_assistant_category
    }

# ======================
# SAVE MESSAGE TO DB
# ======================
def save_message(session_id, role, content):
    conn = get_db("output/analysis.db")
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO session_history (session_id, role, chat)
        VALUES (?, ?, ?)
    """, (session_id, role, content))

    conn.commit()
    conn.close()

def load_recent_messages(session_id, limit=10):
    # 1) Prefer Smartchat history (authoritative)
    try:
        params = {
            "conversation_id": session_id,
            "limit": limit * 5
        }
        headers = {
            "Authorization": f"Bearer {SMARTCHAT_TOKEN}"
        }
        resp = requests.get(
            SMARTCHAT_LATEST_CHAT_API,
            params=params,
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            messages = data.get("last_chat", []) or []
            history = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("chat")
                if role not in ["user", "assistant"]:
                    continue
                if role == "assistant":
                    need_confirmation = msg.get("need_confirmation", 1)
                    is_delivered = msg.get("is_delivered", 0)
                    if need_confirmation != 0 or is_delivered != 1:
                        continue
                if not content:
                    continue
                history.append({
                    "role": role,
                    "content": content.strip()
                })

            history.reverse()

            # Merge consecutive same-role messages
            merged = []
            for msg in history:
                if not merged:
                    merged.append(msg)
                    continue
                last = merged[-1]
                if msg["role"] == last["role"]:
                    last["content"] += "\n" + msg["content"]
                else:
                    merged.append(msg)

            return merged[-limit:]
    except Exception as e:
        logger.warning(f"[SMARTCHAT] load_recent_messages failed: {e}")

    # 2) Fallback to local session_history
    conn = get_db("output/analysis.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT role, chat
        FROM session_history
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (session_id, limit))
    rows = cur.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]

def _find_first_key(obj, keys):
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key]:
                return obj[key]
        for val in obj.values():
            found = _find_first_key(val, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_first_key(item, keys)
            if found:
                return found
    return None

def _collect_key_paths(obj, prefix="", out=None, max_items=60):
    if out is None:
        out = []
    if len(out) >= max_items:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if len(out) >= max_items:
                break
            path = f"{prefix}.{k}" if prefix else str(k)
            out.append(path)
            _collect_key_paths(v, path, out, max_items)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if len(out) >= max_items:
                break
            path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.append(path)
            _collect_key_paths(item, path, out, max_items)
    return out

def _normalize_company_tokens(text):
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    tokens = [t for t in cleaned.split() if len(t) >= 3 and t not in {"pt", "tbk"}]
    return set(tokens)

def _split_knowledge_blocks(knowledge_context):
    if not knowledge_context:
        return []
    parts = knowledge_context.split("\n- ")
    blocks = []
    for i, part in enumerate(parts):
        if i == 0 and part.startswith("- "):
            part = part[2:]
        blocks.append(part)
    return blocks


def _restore_atas_nama_from_prompt_fallback(text):
    if not text or "{{$company}}" not in text:
        return text
    # Use ONLY payment account owner names defined in the prompt fallback.
    fallback = {
        "01-351-00-16-00004-3": "PT EBYB GLOBAL MARKETPLACE",
        "878-0532239": "EBYB GLOBAL MARKETPLACE",
        "118-00-1500440-0": "PT EBYB GLOBAL MARKETPLACE",
        "050201000623569": "EBYB GLOBAL MARKETPLACE",
        "03 7958 3999": "PT. ASA INOVASI SOFTWARE",
        "099999 555 3": "PT EKSA DIGITAL AGENCY",
    }
    lines = text.splitlines()
    current_account = None

    def _normalize_account_number(value):
        cleaned = re.sub(r"[^0-9]", "", str(value))
        return cleaned if cleaned else None

    fallback_by_digits = {
        _normalize_account_number(account): owner
        for account, owner in fallback.items()
    }

    for i, line in enumerate(lines):
        acct_match = re.search(r"(?i)nomor rekening\s*:\s*([0-9\- ]+)", line)
        if acct_match:
            current_account = acct_match.group(1).strip()
            continue
        bare_account_match = re.fullmatch(r"\s*([0-9][0-9\- ]{5,})\s*", line)
        if bare_account_match:
            current_account = bare_account_match.group(1).strip()
            continue

        if "{{$company}}" in line and re.search(r"(?i)\b(atas nama|a\s*/\s*n)\b", line):
            normalized_account = _normalize_account_number(current_account)
            owner_name = fallback_by_digits.get(normalized_account)
            if owner_name:
                label_match = re.match(r"^(\s*)([^:]+:\s*)", line)
                prefix = ""
                if label_match:
                    prefix = f"{label_match.group(1)}{label_match.group(2)}"
                else:
                    prefix = "Atas nama: "
                lines[i] = f"{prefix}{owner_name}"
    return "\n".join(lines)


def fetch_smartchat_company(conversation_id):
    url = f"{SMARTCHAT_DETAIL_API}/{conversation_id}"
    headers = {
        "Authorization": f"Bearer {SMARTCHAT_TOKEN}"
    }
    whatsapp_company_map = {
        1: "PT. Asa Inovasi Software (Asain)",
        2: "PT. Eksa Digital Agency (EDA)",
        3: "PT EBYB Global Marketplace",
        4: "PT EBYB Global Marketplace",
        5: "PT EBYB Global Marketplace"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        logger.debug(f"[SMARTCHAT DETAIL] Status Code: {response.status_code}")
        logger.debug(f"[SMARTCHAT DETAIL] Raw Response: {response.text}")

        if response.status_code != 200:
            logger.error("[SMARTCHAT DETAIL] API returned non-200 status")
            return None

        data = response.json()
    except Exception as e:
        logger.error(f"[SMARTCHAT DETAIL] API request failed: {str(e)}")
        return None

    key_paths = _collect_key_paths(data)
    logger.debug(f"[SMARTCHAT DETAIL] key_paths_sample={key_paths}")

    whatsapp_id = _find_first_key(data, keys=["whatsapp_id"])
    try:
        whatsapp_id_int = int(whatsapp_id)
    except (TypeError, ValueError):
        whatsapp_id_int = None

    company = whatsapp_company_map.get(whatsapp_id_int)
    logger.debug(
        f"[SMARTCHAT DETAIL] whatsapp_id={whatsapp_id} mapped_company={company}"
    )
    logger.debug(f"[SMARTCHAT DETAIL] company={company}")
    return company

# ======================
# MAIN ENTRY
# ======================
def chat_with_session(user_message, session_id, reset=False):

    logger.debug(f"[SESSION START] session_id={session_id}")
    logger.debug(f"[USER MESSAGE] {user_message}")

    save_message(session_id, "user", user_message)

    with SESSION_LOCK:
        session_data = dict(SESSION_STORE.get(session_id, {}))
        company = session_data.get("company")

        # BUILD CONTEXT
        context_messages = load_recent_messages(session_id, limit=10)

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
                "category_product": [],
                "company": company
            }
            SESSION_STORE[session_id] = session_data
        else:
            prev_node_id = session_data.get("prev_node_id")
            assistant_category = session_data.get("assistant_category")

            if "category_product" not in session_data:
                session_data["category_product"] = []

        if not company:
            company = fetch_smartchat_company(session_id)

        if company:
            session_data["company"] = company

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
        company=company,
    )

    final_response = result.get("response", "")
    assistant_node_id = result.get("node_id")
    metadata = result.get("metadata", {})
    metadata = metadata or {} 
    best_assistant_node_id = result.get("best_assistant_node_id")
    assistant_candidates = result.get("assistant_candidates", [])
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
    wants_examples = result.get("wants_examples")
    website_examples_used = result.get("website_examples_used")
    negotiated_price_detected = result.get("negotiated_price_detected")
    negotiated_price_value = result.get("negotiated_price_value")
    resolved_assistant_intent = result.get("resolved_assistant_intent")
    resolved_assistant_category = result.get("resolved_assistant_category")
    
    assistant_node = NODES.get(assistant_node_id, {}) if assistant_node_id else {}
    final_assistant_intent = (
        assistant_node.get("intent")
        or resolved_assistant_intent
        or "assistant_generated_response"
    )
    final_assistant_category = (
        assistant_node.get("category")
        or resolved_assistant_category
        or user_category
        or "conversational_closing"
    )

    # =================
    # UPDATE SESSION 
    # =================
    with SESSION_LOCK:
        session_data["prev_node_id"] = assistant_node_id
        session_data["assistant_category"] = final_assistant_category
        session_data["category_product"] = normalize_category_product(
            result.get("category_product", [])
        )
        SESSION_STORE[session_id] = session_data

    logger.debug(
        f"[PRODUCT MEMORY FINALIZED] "
        f"stored={session_data.get('category_product')}"
    )

    # ======================
    # DEBUG INFO
    # ======================
    debug_info = {
        "user_intent": user_intent,
        "user_category": user_category,
        "company": company,

        # ROUTING
        "match_type": metadata.get("best_type"),
        "best_similarity": round(metadata.get("best_sim", 0), 3),
        "best_user_node_id": metadata.get("best_user_node_id"),
        "best_assistant_node_id": best_assistant_node_id,
        "assistant_candidates": metadata.get("assistant_candidate_scores", assistant_candidates),

        # RESULT
        "assistant_intent": final_assistant_intent,
        "assistant_category": final_assistant_category,

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
        "price_corrected": price_corrected,

        # SPECIAL FLAGS
        "wants_examples": wants_examples,
        "website_examples_used": website_examples_used,
        "negotiated_price_detected": negotiated_price_detected,
        "negotiated_price_value": negotiated_price_value
    }

    logger.debug(f"[FINAL RESPONSE] {final_response}")
    logger.debug(f"[ASSISTANT NODE ID] {assistant_node_id}")
    logger.debug(f"[SESSION CATEGORY PRODUCT] {session_data.get('category_product')}")
    logger.debug(f"[SESSION END] session_id={session_id}")

    save_message(session_id, "assistant", final_response)

    return {
        "response": final_response,
        "node_id": assistant_node_id,
        "debug": debug_info
    }
