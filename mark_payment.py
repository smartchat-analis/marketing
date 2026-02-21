import pandas as pd
import re, json, time
from urllib.parse import quote
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_one_conversation_payment(cid, df, client, tools, url_regex):
    conv_df = df[df["conversation_id"] == cid].copy()
    media_rows = conv_df[conv_df["role"].str.lower() == "media"]

    if media_rows.empty:
        return cid, df, {
            "found_receipt": False,
            "results": [],
            "message": "no media"
        }

    conv_results = []
    found_receipt = False
    receipt_index = None

    for idx, row in media_rows.iterrows():
        chat_text = str(row.get("chat", "")).strip()
        urls = url_regex.findall(chat_text)

        for url in urls:
            encoded_url = quote(url.strip(), safe=":/?&=#.%")

            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", 
                             "content": 
                                            "You are a highly strict and structure-aware financial image validator.\n"
                                            "Your job is ONLY to identify real payment receipts or genuine money transfer proofs.\n"
                                            "================================================================================\n"
                                            "REJECT the image (is_receipt=false) if it is:\n"
                                            "- any advertisement, promotion, poster, banner, marketing design\n"
                                            "- images showing big numbers like '4 JT', '5.000.000', '50%', etc. without transaction context\n"
                                            "- pictures of people, models, products, or call center numbers\n"
                                            "- screenshots of websites, catalogs, WhatsApp chats, or social media posts\n"
                                            "- edited images, aesthetic graphics, or anything with handwriting\n"
                                            "================================================================================\n"
                                            "================================================================================\n"
                                            "ONLY classify an image as a REAL receipt if BOTH conditions are satisfied:\n"
                                            "(1) TEXT-BASED RECEIPT ELEMENTS (should appear clearly):\n"
                                            "    - sender or payer information (name/ID/phone)\n"
                                            "    - receiver or merchant name\n"
                                            "    - bank/platform (BCA, BRI, Mandiri, QRIS, Dana, OVO, ShopeePay, Seabank, etc.)\n"
                                            "    - transaction date and time\n"
                                            "    - reference number / transaction ID / authorization code\n"
                                            "    - transaction amount (Total / Amount Paid / Jumlah / Nominal)\n"
                                            "    - payment method (transfer, QRIS, VA, debit, mobile banking)\n"
                                            "If multiple of these are missing → NOT a receipt.\n"
                                            "(2) STRUCTURAL VISUAL FEATURES (detect at least TWO):\n"
                                            "    - printed/machine-generated font (NOT handwritten)\n"
                                            "    - structured receipt layout (header area, aligned rows, spacing)\n"
                                            "    - merchant/bank logo in a header position\n"
                                            "    - QR code or barcode block\n"
                                            "    - consistent banking-app UI elements (uniform typography, aligned sections)\n"
                                            "If structural features do NOT resemble real receipts → NOT a receipt.\n"
                                            "================================================================================\n"
                                            "ADDITIONAL RULES:\n"
                                            "- Fake receipts or manually designed templates must be rejected.\n"
                                            "- A photo containing only a large number must be rejected.\n"
                                            "- If uncertain, always classify as NOT a receipt.\n"
                                            "RETURN RULE:\n"
                                            "Return is_receipt=true ONLY when BOTH textual and structural criteria match real receipts.\n"
                                            "Otherwise ALWAYS return is_receipt=false and payment_value=null.\n"
                                            "Extract only the actual total amount paid by the customer (exclude admin fees if visible)."
                             },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Analyze this payment receipt image."},
                                    {"type": "image_url", "image_url": {"url": encoded_url}}
                                ]
                            }
                        ],
                        tools=tools,
                        tool_choice="auto",
                        temperature=0
                    )

                    msg = resp.choices[0].message
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not tool_calls:
                        break

                    parsed = json.loads(tool_calls[0].function.arguments)
                    is_receipt = bool(parsed.get("is_receipt"))
                    payment_value = parsed.get("payment_value")

                    conv_results.append({
                        "row_index": idx,
                        "image_url": encoded_url,
                        "is_receipt": is_receipt,
                        "payment_value": payment_value
                    })

                    if is_receipt:
                        df.loc[idx, "payment_marker"] = "receipt found"
                        print(f"Receipt found at row {idx} in conversation {cid}")
                        found_receipt = True
                        receipt_index = idx
                    break

                except Exception:
                    if attempt == 2:
                        conv_results.append({
                            "row_index": idx,
                            "image_url": encoded_url,
                            "error": "API error"
                        })
                    else:
                        time.sleep(1)

            if found_receipt:
                break
        if found_receipt:
            break

    return cid, df, {
        "found_receipt": found_receipt,
        "receipt_index": receipt_index,
        "results": conv_results
    }

def mark_payment_df_parallel(df_bubble: pd.DataFrame, client: OpenAI, max_workers=3):
    df = df_bubble.copy()
    if "payment_marker" not in df.columns:
        df["payment_marker"] = ""
    conversation_ids = df["conversation_id"].dropna().unique().tolist()
    url_regex = re.compile(
        r'https?://[^\n\r]+?\.(?:jpg|jpeg|png|webp|gif)',
        re.IGNORECASE
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "analyze_image_for_receipt_and_value",
                "description": "Analyze a payment receipt image and extract the total value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_receipt": {"type": "boolean"},
                        "payment_value": {"type": ["number", "null"]},
                    },
                    "required": ["is_receipt", "payment_value"],
                },
            },
        }
    ]
    overall_results = {}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(conversation_ids))) as executor:
        futures = [
            executor.submit(
                _process_one_conversation_payment,
                cid, df, client, tools, url_regex
            )
            for cid in conversation_ids
        ]

        for fut in as_completed(futures):
            cid, df, result = fut.result()
            overall_results[str(cid)] = result

    return df, overall_results