import os
import re
import sqlite3

DB_PATH = "output/data_website.db"
MAX_SAMPLES = 5

def _get_db(path):
    return sqlite3.connect(path, check_same_thread=False)

def _normalize_text(t: str):
    return re.sub(r"\s+", " ", str(t).lower().strip())

def _extract_package(text: str):
    t = _normalize_text(text)
    if "silver" in t:
        return "Silver"
    if "gold" in t:
        return "Gold"
    if "diamond" in t:
        return "Diamond"
    if "platinum" in t:
        return "Platinum"
    return None

def _extract_business_hint(text: str):
    t = _normalize_text(text)
    patterns = [
        r"(?:bidang|jenis)\s+usaha(?:nya)?\s*(?:di|:)?\s*([a-z0-9 &/\\-]{3,80})",
        r"contoh\s+web(?:site)?\s+(?:untuk|di\s+bidang)\s+([a-z0-9 &/\\-]{3,80})",
        r"website\s+untuk\s+([a-z0-9 &/\\-]{3,80})"
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1).strip(" .,:;")
    return None

def _get_distinct_values(conn, column):
    cur = conn.cursor()
    cur.execute(f"SELECT DISTINCT {column} FROM websites")
    return [r[0] for r in cur.fetchall() if r and r[0]]

def _match_values_in_text(text, values):
    t = _normalize_text(text)
    matches = []
    for v in values:
        v_norm = _normalize_text(v)
        if v_norm and v_norm in t:
            matches.append(v)
    return matches

def _query_examples(conn, company, kategori=None, tema=None, paket=None, use_like=False, limit=MAX_SAMPLES):
    params = [company]
    where = ["Company = ?"]
    if kategori:
        if use_like:
            where.append("Kategori LIKE ?")
            params.append(f"%{kategori}%")
        else:
            placeholders = ",".join(["?"] * len(kategori))
            where.append(f"Kategori IN ({placeholders})")
            params.extend(kategori)
    if tema:
        if use_like:
            where.append("Tema LIKE ?")
            params.append(f"%{tema}%")
        else:
            placeholders = ",".join(["?"] * len(tema))
            where.append(f"Tema IN ({placeholders})")
            params.extend(tema)
    if paket:
        where.append("Paket = ?")
        params.append(paket)

    where_sql = " AND ".join(where)
    sql = f"SELECT Domain, Paket, Kategori, Tema FROM websites WHERE {where_sql} LIMIT ?"
    params.append(limit)
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()

def maybe_build_examples_response(
    user_message,
    context_summary,
    llm1_output,
    company,
    wants_examples=False,
    db_path=DB_PATH,
    max_samples=MAX_SAMPLES
):
    if not wants_examples:
        return None

    if not company or not os.path.exists(db_path):
        return None

    company_code = str(company).upper().strip()
    combined_context = f"{context_summary} {user_message}"
    paket = _extract_package(combined_context)
    hint = _extract_business_hint(combined_context)

    conn = _get_db(db_path)
    try:
        kategori_values = _get_distinct_values(conn, "Kategori")
        tema_values = _get_distinct_values(conn, "Tema")

        kategori_matches = _match_values_in_text(combined_context, kategori_values)
        tema_matches = _match_values_in_text(combined_context, tema_values)

        results = []
        paket_relaxed = False

        # 1) Prefer kategori
        if kategori_matches or hint:
            if kategori_matches:
                results = _query_examples(conn, company_code, kategori=kategori_matches, paket=paket, limit=max_samples)
            elif hint:
                results = _query_examples(conn, company_code, kategori=hint, paket=paket, use_like=True, limit=max_samples)

            if paket and not results:
                paket_relaxed = True
                if kategori_matches:
                    results = _query_examples(conn, company_code, kategori=kategori_matches, limit=max_samples)
                elif hint:
                    results = _query_examples(conn, company_code, kategori=hint, use_like=True, limit=max_samples)

        # 2) Fallback ke tema
        if not results and (tema_matches or hint):
            if tema_matches:
                results = _query_examples(conn, company_code, tema=tema_matches, paket=paket, limit=max_samples)
            elif hint:
                results = _query_examples(conn, company_code, tema=hint, paket=paket, use_like=True, limit=max_samples)

            if paket and not results:
                paket_relaxed = True
                if tema_matches:
                    results = _query_examples(conn, company_code, tema=tema_matches, limit=max_samples)
                elif hint:
                    results = _query_examples(conn, company_code, tema=hint, use_like=True, limit=max_samples)

        # 3) If no jenis usaha detected, fallback by company (and paket if specified)
        if not results and not kategori_matches and not tema_matches and not hint:
            results = _query_examples(conn, company_code, paket=paket, limit=max_samples)
            if paket and not results:
                paket_relaxed = True
                results = _query_examples(conn, company_code, limit=max_samples)

        if not results:
            if hint:
                fallback = f"Baik kak, mohon ditunggu yaa saya carikan dulu di database terkait contoh website di bidang {hint}."
            else:
                fallback = (
                    "Baik kak, mohon maaf sebelumnya boleh saya tau usahanya di bidang apa kak? "
                    "Supaya saya bisa kirimkan contoh website buatan kami yang sesuai😊"
                )
            return {"response": fallback, "replaced": True}

        # Build response
        jenis = None
        if kategori_matches:
            jenis = kategori_matches[0]
        elif tema_matches:
            jenis = tema_matches[0]
        elif hint:
            jenis = hint

        lines = []
        if paket and jenis:
            lines.append(f"Baik kak, berikut beberapa contoh website paket {paket} di bidang usaha {jenis}:")
        elif jenis:
            lines.append(f"Baik kak, berikut beberapa contoh website buatan kami di bidang usaha {jenis}:")
        else:
            lines.append("Baik kak, berikut beberapa contoh website buatan kami:")

        if paket and paket_relaxed:
            lines.insert(
                0,
                f"Mohon maaf kak, untuk bidang ini paket {paket} belum ada. Ini saya kirimkan dari paket lain ya kak."
            )

        for i, row in enumerate(results[:max_samples], start=1):
            domain = row[0]
            lines.append(f"{i}. {domain}")

        lines.append("Silahkan bisa dilihat-lihat terlebih dahulu😊")

        return {"response": "\n".join(lines), "replaced": True}
    finally:
        conn.close()
