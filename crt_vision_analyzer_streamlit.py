import base64
import json
import os
from io import BytesIO
from typing import Any

import requests
import streamlit as st
from PIL import Image


OPENAI_URL = "https://api.openai.com/v1/responses"
API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

SYSTEM_PROMPT = """You are a strict CRT scalping analyst reading a trading chart screenshot.
Your job is to analyze the provided screenshot and optional notes using:
- HTF bias
- draw on liquidity (DOL)
- CRT range context
- key levels / PD arrays
- liquidity sweep
- MSS/CISD
- session profile
- realistic scalp TP logic

Rules:
- No DOL = no trade.
- No sweep + no MSS/CISD = no trade.
- Middle of range = usually no trade.
- Long requires sweep of lows + bullish confirmation.
- Short requires sweep of highs + bearish confirmation.
- SL goes beyond sweep extreme or MSS invalidation.
- TP1 = nearest internal liquidity
- TP2 = CRT midpoint
- TP3 = opposite side of CRT range or next external liquidity
- Minimum RR to TP1 should be 1.5 unless clearly impossible.
- If the chart is unreadable or important context is missing, say so in warnings and missing_data.
- Never invent precise chart levels unless they can be reasonably inferred from the screenshot.
- Be conservative.

Return ONLY valid JSON in this schema:
{
  "bias": {
    "direction": "bullish | bearish | neutral",
    "explanation": "string"
  },
  "dol": {
    "target_side": "above | below | mixed | unclear",
    "liquidity_targets": ["string"],
    "confidence": "high | medium | low"
  },
  "crt_context": {
    "active_range_name": "string",
    "crt_high": number | null,
    "crt_low": number | null,
    "crt_mid": number | null,
    "session_context": "string",
    "profile": "continuation | reversal | expansion | compression | lunch_behavior | unclear"
  },
  "key_levels": [
    {
      "type": "OB | BB | RB | FVG | IFVG | HIGH | LOW | PDH | PDL | SESSION_HIGH | SESSION_LOW | OTHER",
      "price": number | null,
      "timeframe": "string",
      "notes": "string"
    }
  ],
  "long_setup": {
    "valid": boolean,
    "entry_zone": [number | null, number | null],
    "confirmation_trigger": "string",
    "stop_loss": number | null,
    "tp1": number | null,
    "tp2": number | null,
    "tp3": number | null,
    "rr_tp1": number | null,
    "rr_tp2": number | null,
    "rr_tp3": number | null,
    "confidence": "high | medium | low",
    "probability_score": number,
    "reason": "string"
  },
  "short_setup": {
    "valid": boolean,
    "entry_zone": [number | null, number | null],
    "confirmation_trigger": "string",
    "stop_loss": number | null,
    "tp1": number | null,
    "tp2": number | null,
    "tp3": number | null,
    "rr_tp1": number | null,
    "rr_tp2": number | null,
    "rr_tp3": number | null,
    "confidence": "high | medium | low",
    "probability_score": number,
    "reason": "string"
  },
  "best_trade_decision": {
    "decision": "LONG | SHORT | NO_TRADE",
    "score": number,
    "reasoning": ["string"]
  },
  "missing_data": ["string"],
  "warnings": ["string"]
}"""


def image_to_data_url(image: Image.Image, format_hint: str = "PNG") -> str:
    buffer = BytesIO()
    fmt = format_hint.upper()
    if fmt not in {"PNG", "JPEG", "JPG", "WEBP"}:
        fmt = "PNG"
    save_fmt = "JPEG" if fmt == "JPG" else fmt
    image.save(buffer, format=save_fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if save_fmt == "JPEG" else f"image/{save_fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def extract_output_text(response_json: dict[str, Any]) -> str:
    if response_json.get("output_text"):
        return response_json["output_text"]

    texts: list[str] = []
    for block in response_json.get("output", []):
        for item in block.get("content", []):
            if item.get("type") == "output_text" and item.get("text"):
                texts.append(item["text"])
    return "\n".join(texts)


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return json.loads(text[first:last + 1])
        raise


def analyze_chart(
    image: Image.Image,
    model: str,
    symbol: str,
    session_hint: str,
    timeframe_note: str,
    notes: str,
) -> dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment.")

    image_url = image_to_data_url(image)
    user_text = (
        f"Analyze this trading chart screenshot with CRT scalping logic.\n"
        f"Symbol: {symbol or 'UNKNOWN'}\n"
        f"Session hint: {session_hint or 'auto'}\n"
        f"Timeframe note: {timeframe_note or 'not provided'}\n"
        f"Extra notes: {notes or 'none'}\n"
        f"Return strict JSON only."
    )

    payload = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "text": {"format": {"type": "json_object"}},
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    }

    response = requests.post(
        OPENAI_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    response_json = response.json()
    raw = extract_output_text(response_json)
    if not raw:
        raise RuntimeError("Model response did not contain output_text.")
    return parse_model_json(raw)


def fmt_num(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def fmt_zone(zone: Any) -> str:
    if isinstance(zone, list) and len(zone) >= 2:
        return f"{fmt_num(zone[0])} – {fmt_num(zone[1])}"
    return "—"


def fmt_rr(setup: dict[str, Any]) -> str:
    return (
        f"TP1 {setup.get('rr_tp1', '—')}R • "
        f"TP2 {setup.get('rr_tp2', '—')}R • "
        f"TP3 {setup.get('rr_tp3', '—')}R"
    )


def render_setup(title: str, setup: dict[str, Any], accent: str) -> None:
    st.markdown(f"### {title}")
    c1, c2 = st.columns(2)
    c1.metric("Valid", "Yes" if setup.get("valid") else "No")
    c2.metric("Probability", f"{setup.get('probability_score', 0)}/100")

    c3, c4 = st.columns(2)
    c3.metric("Entry Zone", fmt_zone(setup.get("entry_zone")))
    c4.metric("Stop Loss", fmt_num(setup.get("stop_loss")))

    c5, c6 = st.columns(2)
    c5.metric("TP1 / TP2 / TP3", f"{fmt_num(setup.get('tp1'))} / {fmt_num(setup.get('tp2'))} / {fmt_num(setup.get('tp3'))}")
    c6.metric("RR", fmt_rr(setup))

    st.caption(f"Confidence: {setup.get('confidence', 'low')}")
    reasons = [setup.get("reason"), setup.get("confirmation_trigger")]
    for item in [x for x in reasons if x]:
        st.write(f"- {item}")
    st.divider()


st.set_page_config(page_title="CRT Vision Analyzer", layout="wide")
st.title("CRT Vision Analyzer")
st.write(
    "Upload a TradingView screenshot, then let the app send the chart image to the OpenAI Responses API using a server-side environment variable."
)

if not API_KEY:
    st.error("OPENAI_API_KEY is missing on the server. Set it in your environment before running this app.")

with st.sidebar:
    st.header("Request Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    symbol = st.text_input("Symbol", value="XAUUSD")
    session_hint = st.selectbox(
        "Session profile hint",
        ["auto", "continuation", "reversal", "expansion", "compression", "lunch_behavior", "unclear"],
        index=0,
    )
    timeframe_note = st.text_input("Timeframe note", value="HTF + execution from screenshot")
    notes = st.text_area(
        "Extra notes",
        placeholder="Optional context: session, suspected sweep, MSS, CRT model, London / NY behavior, etc.",
        height=120,
    )
    st.info("The API key is read only from OPENAI_API_KEY on the server side. The browser does not send its own key.")

uploaded = st.file_uploader("Upload chart screenshot", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded chart", use_container_width=True)

    if st.button("Analyze Screenshot", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing chart screenshot..."):
                result = analyze_chart(
                    image=image,
                    model=model,
                    symbol=symbol,
                    session_hint=session_hint,
                    timeframe_note=timeframe_note,
                    notes=notes,
                )
            st.success("Analysis complete.")

            col1, col2, col3 = st.columns(3)
            bias = result.get("bias", {})
            dol = result.get("dol", {})
            best = result.get("best_trade_decision", {})
            col1.metric("Bias", str(bias.get("direction", "neutral")).upper())
            col2.metric("DOL", str(dol.get("target_side", "unclear")).upper())
            col3.metric("Best Decision", str(best.get("decision", "NO_TRADE")))

            st.subheader("Bias and Decision")
            st.write(bias.get("explanation", "No explanation returned."))
            for item in best.get("reasoning", []) or []:
                st.write(f"- {item}")

            left, right = st.columns(2)
            with left:
                render_setup("Long Scenario", result.get("long_setup", {}), "good")
            with right:
                render_setup("Short Scenario", result.get("short_setup", {}), "bad")

            st.subheader("Warnings")
            warnings = result.get("warnings", []) or ["No warnings returned."]
            for item in warnings:
                st.write(f"- {item}")

            st.subheader("Missing Data")
            missing = result.get("missing_data", []) or ["No missing data returned."]
            for item in missing:
                st.write(f"- {item}")

            st.subheader("Raw JSON")
            st.json(result)

        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"API request failed: {detail}")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
else:
    st.info("Upload a chart screenshot to begin.")
