# -*- coding: utf-8 -*-
# Bidirectional Translation Tutor - Arabic <-> English
# Strong feedback (span-level), verifier pass, rubric weights, optional instructor/student flow.

import os
import io
import json
import re
from typing import Dict, List, Any, Tuple

import streamlit as st
from gtts import gTTS

# Optional imports so the app still runs without them
try:
    import openai
except Exception:
    openai = None

import language_tool_python as lt

import nltk
from nltk.corpus import wordnet as wn
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

# Optional DB
try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None

# Risk checks
try:
    import dateutil.parser as du
except Exception:
    du = None

st.set_page_config(page_title="Arabic <-> English Translation Tutor", page_icon="🗣️", layout="wide")

# =============================
# Arabic helpers & resources
# =============================
AR_NEGATIONS = {"لا", "لم", "لن", "ما", "ليس", "بدون", "غير"}
EN_NEG_TOKENS = {" not ", "n't", " never ", " no "}

DIACRITICS = re.compile("[\u0617-\u061A\u064B-\u0652\u0657-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"

# False friends (Arabic cue -> English word)
FALSE_FRIENDS_AR_EN = {
    "فعليًا": ("actually", "Use 'actually' for emphasis/contrast, not for 'currently'. For 'currently', use 'currently/at the moment'."),
    "حساس": ("sensitive", "'Sensitive' is not 'sensible'. 'Sensible' = reasonable; 'sensitive' = easily affected."),
    "أخيرا": ("finally", "Use 'finally' for the end of a process, not for 'eventually' (at some point in the future)."),
}

OVERUSED_ENGLISH = {
    "good": ["strong", "effective", "solid", "helpful", "beneficial", "favorable"],
    "very": ["extremely", "highly", "particularly", "remarkably"],
    "important": ["crucial", "vital", "essential", "key", "significant"],
    "big": ["large", "major", "substantial", "considerable"],
    "small": ["minor", "slight", "limited", "compact"],
    "make": ["create", "produce", "cause", "prepare", "build"],
    "do": ["perform", "carry out", "conduct", "execute"],
}

OVERUSED_ARABIC = {
    "جيد": ["ممتاز", "قوي", "فعّال", "مفيد", "ملائم"],
    "كبير": ["ضخم", "واسع", "بارز", "هائل", "ملحوظ"],
    "مهم": ["أساسي", "محوري", "جوهري", "حيوي", "رئيسي"],
    "جداً": ["للغاية", "إلى حد كبير", "بشكل كبير"],
    "عمل": ["أنجز", "قام بـ", "نفّذ", "أدى"],
    "جعل": ["حوّل", "سبّب", "أدى إلى", "أحدث"],
}

PREP_TIPS_EN = [
    ("in", "Months/years and large areas: in 2020, in July, in the city"),
    ("on", "Days/dates and surfaces: on Monday, on 5 May, on the table"),
    ("at", "Precise times/points: at 7 pm, at the door, at work"),
]

PREP_TIPS_AR = [
    ("في", "تستخدم مع الزمن العام/الاماكن الواسعة: في 2020، في يوليو، في المدينة"),
    ("على", "تستخدم مع الاسطح/بعض التراكيب؛ ملاحظة: on Monday -> يوم الاثنين"),
    ("عند", "تستخدم مع الاوقات/النقاط المحددة: عند الساعة السابعة، عند الباب"),
]

# =============================
# LanguageTool setup (rate-limit safe)
# =============================

def get_lt_tool(lang_code: str = "en-US"):
    api_url = st.secrets.get("LT_API_URL", os.getenv("LT_API_URL"))
    api_key = st.secrets.get("LT_API_KEY", os.getenv("LT_API_KEY"))
    try:
        return lt.LanguageToolPublicAPI(lang_code, api_key=api_key, api_url=api_url)  # type: ignore
    except Exception:
        try:
            return lt.LanguageToolPublicAPI(lang_code)  # type: ignore
        except Exception:
            return lt.LanguageTool(lang_code)

def safe_lt_check(text: str, lang_code: str = "en-US") -> List[Any]:
    tool = get_lt_tool(lang_code)
    try:
        return tool.check(text)
    except Exception:
        st.warning("Grammar server is busy (rate limit). Using lightweight checks this run.")
        return []

# =============================
# Simple local checks (fallback for English)
# =============================

def fallback_text_checks_en(text: str) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for m in re.finditer(r"\b(\w+)\s+\1\b", text, flags=re.IGNORECASE):
        issues.append({
            "type": "fluency",
            "message": f"Repeated word '{m.group(1)}'.",
            "offset": m.start(),
            "length": len(m.group(0)),
            "replacements": [m.group(1)]
        })
    for m in re.finditer(r"  +", text):
        issues.append({
            "type": "style",
            "message": "Multiple consecutive spaces.",
            "offset": m.start(),
            "length": len(m.group(0)),
            "replacements": [" "]
        })
    for line in [s.strip() for s in text.split("\n") if s.strip()]:
        if len(line.split()) >= 6 and not re.search(r"[.!?]$", line):
            issues.append({
                "type": "punctuation",
                "message": "Consider ending the sentence with a period.",
                "offset": 0,
                "length": 0,
                "replacements": ["."]
            })
    return issues

# =============================
# Utility
# =============================

def ar_normalize(s: str) -> str:
    s = DIACRITICS.sub("", s or "")
    s = s.replace(TATWEEL, "")
    s = re.sub("[\u0622\u0623\u0625]", "ا", s)
    s = s.replace("ى", "ي").replace("ئ", "ي")
    return s

# =============================
# OpenAI helpers (optional)
# =============================

def has_openai() -> bool:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return bool(key)

def _model_name():
    return os.getenv("OPENAI_MODEL", "gpt-4o")

def init_openai():
    if not has_openai():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")))
        return client
    except Exception:
        if openai:
            openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
            return "legacy"
        return None

# =============================
# Core analysis helpers
# =============================

def analyze_with_languagetool_en(text: str) -> List[Dict[str, Any]]:
    matches = safe_lt_check(text, "en-US")
    if not matches:
        return fallback_text_checks_en(text)
    issues: List[Dict[str, Any]] = []
    for m in matches:
        issues.append({
            "type": getattr(m, "ruleIssueType", None) or "grammar",
            "message": m.message,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": [r.value for r in m.replacements][:5]
        })
    return issues

def wordnet_synonyms_en(word: str) -> List[str]:
    syns = set()
    for synset in wn.synsets(word, lang="eng"):
        for lemma in synset.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower():
                syns.add(w)
    return sorted(list(syns))[:10]

# =============================
# Heuristics: Arabic -> English
# =============================

def heuristics_ar_to_en(ar_source: str, en_student: str) -> Tuple[List[str], Dict[str, List[str]]]:
    notes: List[str] = []
    ar_norm = ar_normalize(ar_source)
    en_low = (en_student or "").lower()

    if any(tok in ar_norm for tok in AR_NEGATIONS):
        if not any(tok in en_low for tok in EN_NEG_TOKENS) and not en_low.startswith("no "):
            notes.append("Source has negation; ensure the English includes 'not/never/no'.")

    if re.search(r"\bال\w+", ar_norm) and re.search(r"\b(the)\b", en_low) is None:
        notes.append("Arabic definite 'ال' may require 'the' in English; check article use.")

    for p, tip in PREP_TIPS_EN:
        if re.search(fr"\b{p}\b", en_low):
            notes.append(f"Preposition '{p}': {tip}")
            break

    for ar_key, (ff, msg) in FALSE_FRIENDS_AR_EN.items():
        if ar_key in ar_norm and re.search(fr"\b{ff}\b", en_low):
            notes.append(f"Possible false friend: '{ff}'. {msg}")

    synonym_suggestions: Dict[str, List[str]] = {}
    tokens = re.findall(r"[a-zA-Z']+", en_student or "")
    counts = {}
    for t in tokens:
        wl = t.lower()
        counts[wl] = counts.get(wl, 0) + 1
    for w, options in OVERUSED_ENGLISH.items():
        if counts.get(w, 0) >= 2 or w in counts:
            merged = list(dict.fromkeys(options + wordnet_synonyms_en(w)))[:8]
            if merged:
                synonym_suggestions[w] = merged

    return notes, synonym_suggestions

# =============================
# Heuristics: English -> Arabic
# =============================

def heuristics_en_to_ar(en_source: str, ar_student: str) -> Tuple[List[str], Dict[str, List[str]]]:
    notes: List[str] = []
    en_low = f" {(en_source or '').lower()} "
    ar_norm = ar_normalize(ar_student or "")

    if any(tok in en_low for tok in EN_NEG_TOKENS) and not any(tok in ar_norm for tok in AR_NEGATIONS):
        notes.append("Source is negative; add Arabic negation (لا/لم/لن/ما/ليس) where appropriate.")

    if " the " in en_low and not re.search(r"\bال\w+", ar_norm):
        notes.append("Source uses 'the'; consider the Arabic definite article 'ال' when needed.")

    for p, tip in PREP_TIPS_AR:
        if p in (ar_student or ""):
            notes.append(f"تلميح حرف الجر '{p}': {tip}")
            break

    if re.search(r"\d", en_source or "") and not re.search(r"\d", ar_student or ""):
        notes.append("تأكد من نقل الأرقام كما هي (أو بالأرقام العربية إذا لزم).")

    if re.search(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", en_source or ""):
        notes.append("حافظ على الأسماء العلم منقولة صوتيًا بدقة أو مترجمة إذا لها مقابل شائع.")

    synonym_suggestions: Dict[str, List[str]] = {}
    tokens = re.findall(r"[\u0600-\u06FF]+", ar_student or "")
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    for w, options in OVERUSED_ARABIC.items():
        if counts.get(w, 0) >= 2 or w in counts:
            synonym_suggestions[w] = options[:8]
    return notes, synonym_suggestions

# =============================
# Quick risk scan (heuristic)
# =============================

def quick_risk_scan(source: str, student: str) -> Dict[str, Any]:
    risks = {"numbers": [], "dates": [], "named_entities": [], "negation_flip": False}
    nums_src = re.findall(r"\d+[\d,\.]*", source or "")
    nums_tgt = re.findall(r"\d+[\d,\.]*", student or "")
    for n in nums_src:
        if n not in nums_tgt:
            risks["numbers"].append(n)
    if du:
        for m in re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}|\d{1,2}\s+\w+\s+\d{4}", source or ""):
            try:
                du.parse(m)
                if m not in (student or ""):
                    risks["dates"].append(m)
            except Exception:
                pass
    for name in re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", source or ""):
        if name not in (student or ""):
            risks["named_entities"].append(name)
    src_neg = bool(re.search(r"\bnot\b|n't|never|no\b", (source or "").lower()) or re.search(r"لا|لم|لن|ما|ليس", source or ""))
    tgt_neg = bool(re.search(r"\bnot\b|n't|never|no\b", (student or "").lower()) or re.search(r"لا|لم|لن|ما|ليس", student or ""))
    risks["negation_flip"] = src_neg != tgt_neg
    return risks

# =============================
# LLM-enhanced coaching (primary + verifier)
# =============================

SCHEMA_HINT = {
  "corrected": "string",
  "alt_rewrites": ["string"],
  "scores": {"fluency": 0, "adequacy": 0, "fidelity": 0, "style": 0, "overall": 0},
  "issues": [
    {"start": 0, "end": 0, "text": "", "type": "", "severity": "", "explanation": "", "suggestion": ""}
  ],
  "risk_flags": {"named_entities": [], "numbers": [], "dates": [], "negation_flip": False, "hallucination": False},
  "spoken_feedback_en": "",
  "spoken_feedback_ar": ""
}

VERIFY_SYSTEM = (
    "You are a strict verifier for translation feedback. Given source, student, and a proposed corrected version, "
    "identify mismatches in named entities, numbers, dates, and polarity (negation). Respond in JSON."
)

def llm_feedback(student_text: str, source_text: str, mode: str) -> Dict[str, Any]:
    data = {
        "corrected": student_text,
        "alt_rewrites": [],
        "scores": {"fluency": 70, "adequacy": 70, "fidelity": 70, "style": 70, "overall": 70},
        "issues": [],
        "risk_flags": {"named_entities": [], "numbers": [], "dates": [], "negation_flip": False, "hallucination": False},
        "spoken_feedback_en": "",
        "spoken_feedback_ar": "",
    }

    # Heuristics + LT pre-pass
    if mode == "ar->en":
        lt_issues = analyze_with_languagetool_en(student_text)
        notes, syns = heuristics_ar_to_en(source_text, student_text)
    else:
        lt_issues = []
        notes, syns = heuristics_en_to_ar(source_text, student_text)

    # Primary LLM pass
    if has_openai():
        client = init_openai()
        if client:
            target_lang = "English" if mode == "ar->en" else "Arabic"
            system_prompt = (
                f"You are a meticulous Arabic-English translation coach. Target language: {target_lang}.\n"
                "Return precise, span-labeled feedback. Keep corrections minimal (preserve meaning).\n"
                "JSON only, matching this schema: " + json.dumps(SCHEMA_HINT)
            )
            user_payload = {
                "direction": mode,
                "source": source_text,
                "student": student_text,
                "requirements": [
                    "Use char offsets start/end relative to the STUDENT string",
                    "Classify: grammar|syntax|fluency|word_choice|adequacy|fidelity|register",
                    "Severity: minor|major|critical",
                    "Provide corrected + up to 2 alternative rewrites",
                    "Scores 0-100 for fluency, adequacy, fidelity, style, overall",
                    "Spoken feedback: brief; one English, one Arabic"
                ]
            }
            try:
                if client == "legacy":
                    resp = openai.chat.completions.create(
                        model=_model_name(), temperature=0.2,
                        messages=[{"role":"system","content":system_prompt},{"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}],
                        response_format={"type":"json_object"}
                    )
                    parsed = json.loads(resp.choices[0].message.content)
                else:
                    resp = client.chat.completions.create(
                        model=_model_name(), temperature=0.2,
                        messages=[{"role":"system","content":system_prompt},{"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}],
                        response_format={"type":"json_object"}
                    )
                    parsed = json.loads(resp.choices[0].message.content)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        data[k] = v
            except Exception as e:
                st.warning(f"LLM feedback temporarily unavailable; kept rule-based. ({e})")

    # Convert LT issues to span issues
    for li in (lt_issues or [])[:10]:
        start = li.get("offset", 0)
        end = start + li.get("length", 0)
        data.setdefault("issues", []).append({
            "start": start, "end": end,
            "text": student_text[start:end],
            "type": li.get("type", "grammar"),
            "severity": "minor",
            "explanation": li.get("message", ""),
            "suggestion": (li.get("replacements") or [""])[0] if isinstance(li.get("replacements"), list) else ""
        })

    # Add heuristics notes + synonyms
    if notes:
        data.setdefault("word_choice_notes", [])
        data["word_choice_notes"].extend(notes)
    if syns:
        data.setdefault("synonym_suggestions", {})
        for k, v in syns.items():
            existing = data["synonym_suggestions"].get(k, [])
            data["synonym_suggestions"][k] = list(dict.fromkeys(existing + v))[:10]

    # Verifier pass
    if has_openai() and data.get("corrected"):
        client = init_openai()
        payload_v = {"source": source_text, "student": student_text, "corrected": data["corrected"]}
        try:
            if client == "legacy":
                r = openai.chat.completions.create(
                    model=_model_name(), temperature=0,
                    messages=[{"role":"system","content":VERIFY_SYSTEM},{"role":"user","content": json.dumps(payload_v, ensure_ascii=False)}],
                    response_format={"type":"json_object"}
                )
                ver = json.loads(r.choices[0].message.content)
            else:
                r = client.chat.completions.create(
                    model=_model_name(), temperature=0,
                    messages=[{"role":"system","content":VERIFY_SYSTEM},{"role":"user","content": json.dumps(payload_v, ensure_ascii=False)}],
                    response_format={"type":"json_object"}
                )
                ver = json.loads(r.choices[0].message.content)
            if isinstance(ver, dict):
                rf = data.get("risk_flags", {})
                for k in ["named_entities", "numbers", "dates", "negation_flip", "hallucination"]:
                    if k in ver:
                        rf[k] = ver[k]
                data["risk_flags"] = rf
        except Exception:
            pass

    # Safety clamps for scores
    sc = data.setdefault("scores", {})
    for k in ["fluency", "adequacy", "fidelity", "style", "overall"]:
        try:
            s = int(sc.get(k, 70))
            sc[k] = min(100, max(0, s))
        except Exception:
            sc[k] = 70

    return data

# =============================
# TTS
# =============================

def tts_bytes(text: str, lang_code: str = "en") -> bytes:
    fp = io.BytesIO()
    tts = gTTS(text=text, lang=lang_code)
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

# =============================
# Optional: Supabase client
# =============================

@st.cache_resource
def supabase_client():
    if not create_client:
        return None
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_SERVICE_ROLE")
    if not url or not key:
        return None
    return create_client(url, key)

# =============================
# UI
# =============================

st.title("Arabic <-> English Translation Tutor")

with st.sidebar:
    role = st.selectbox("Role", ["Student", "Instructor"], index=0)
    st.header("Rubric weights")
    w_flu = st.slider("Fluency weight", 0, 100, 25)
    w_ade = st.slider("Adequacy weight", 0, 100, 25)
    w_fid = st.slider("Fidelity weight", 0, 100, 25)
    w_sty = st.slider("Style weight", 0, 100, 25)
    minimal = st.toggle("Prefer minimal edits", value=True)

mode = st.radio(
    "Direction",
    ["Arabic -> English", "English -> Arabic"],
    index=0,
    horizontal=True
)

# Instructor panel
if role == "Instructor":
    st.subheader("Create exercise")
    ex_title = st.text_input("Title")
    ex_dir = st.radio("Direction", ["ar->en", "en->ar"], horizontal=True)
    ex_source = st.text_area("Source text", height=140)
    ex_level = st.selectbox("Level", ["B1", "B2", "C1", "C2"], index=1)
    sb = supabase_client()
    if st.button("Save exercise"):
        if not sb:
            st.error("Supabase is not configured.")
        elif not (ex_title and ex_source):
            st.warning("Title and source are required.")
        else:
            sb.table("exercises").insert({
                "title": ex_title, "direction": ex_dir, "source": ex_source, "level": ex_level
            }).execute()
            st.success("Exercise saved.")
    st.divider()
    if sb:
        st.subheader("Existing exercises")
        rows = sb.table("exercises").select("id,title,direction,level,created_at").order("created_at", desc=True).execute().data
        for r in rows:
            st.write(f"• {r['title']} ({r['direction']}, {r['level']}) — {r['id']}")

# Student / ad-hoc panel
if role == "Student":
    sb = supabase_client()
    chosen = None
    if sb:
        st.subheader("Pick an exercise (optional)")
        rows = sb.table("exercises").select("id,title,direction,source").order("created_at", desc=True).execute().data
        if rows:
            idx = st.selectbox("Exercise", [-1] + list(range(len(rows))), format_func=lambda k: "None (use custom text)" if k == -1 else f"{rows[k]['title']} — {rows[k]['direction']}")
            if idx != -1:
                chosen = rows[idx]

    if mode == "Arabic -> English" or (chosen and chosen["direction"] == "ar->en"):
        st.markdown("Provide Arabic source and your English translation.")
        col1, col2 = st.columns(2)
        with col1:
            source_text = st.text_area("النصّ العربي (Source in Arabic)", height=180, value=(chosen["source"] if chosen and chosen["direction"] == "ar->en" else ""))
        with col2:
            student_text = st.text_area("Your English translation", height=180)
        tts_target_lang = "en"
        mode_key = "ar->en"
    else:
        st.markdown("Provide English source and your Arabic translation.")
        col1, col2 = st.columns(2)
        with col1:
            source_text = st.text_area("Source in English", height=180, value=(chosen["source"] if chosen and chosen["direction"] == "en->ar" else ""))
        with col2:
            student_text = st.text_area("ترجمتك العربية", height=180)
        tts_target_lang = "ar"
        mode_key = "en->ar"

    st.divider()
    if st.button("Analyze & Coach"):
        if not source_text.strip():
            st.warning("Please paste the source text.")
            st.stop()
        working_text = (student_text or "").strip()
        if not working_text:
            st.warning("Paste your translation to analyze.")
            st.stop()

        enriched = llm_feedback(working_text, source_text, mode_key)

        # Scores & overall
        sc = enriched.get("scores", {})
        overall = (
            sc.get("fluency", 0) * w_flu + sc.get("adequacy", 0) * w_ade + sc.get("fidelity", 0) * w_fid + sc.get("style", 0) * w_sty
        ) / max(1, (w_flu + w_ade + w_fid + w_sty))
        st.metric("Overall score", f"{overall:.0f}/100")

        st.subheader("Corrected Version (target)")
        st.write(enriched.get("corrected", working_text))

        if enriched.get("alt_rewrites"):
            with st.expander("Alternative rewrites (style variants)"):
                for i, alt in enumerate(enriched["alt_rewrites"], 1):
                    st.markdown(f"**Alt {i}:** {alt}")

        st.subheader(f"Feedback (Fluency: {sc.get('fluency',0)}/100, Adequacy: {sc.get('adequacy',0)}/100, Fidelity: {sc.get('fidelity',0)}/100, Style: {sc.get('style',0)}/100)")
        issues = enriched.get("issues", [])
        if issues:
            st.markdown("Highlighted issues")
            for i, iss in enumerate(issues, 1):
                frag = iss.get("text", "")
                st.markdown(
                    f"**{i}. {iss.get('type','issue')} ({iss.get('severity','minor')})** — {iss.get('explanation','')}\n\n"
                    f"Suggestion: _{iss.get('suggestion','')}_\n\n"
                    f"Span: `{iss.get('start',0)}–{iss.get('end',0)}` -> `{frag}`"
                )
        else:
            st.write("No span-level issues found.")

        wc = enriched.get("word_choice_notes", [])
        if wc:
            st.subheader("Notes & Strategy")
            for note in wc:
                st.write(f"- {note}")

        syns = enriched.get("synonym_suggestions", {})
        if syns:
            st.subheader("Synonym Suggestions")
            for w, options in syns.items():
                st.markdown(f"- **{w}** -> {', '.join(options)}")

        # Risk scan (heuristic) on corrected
        qr = quick_risk_scan(source_text, enriched.get("corrected", working_text))
        if any([qr["numbers"], qr["dates"], qr["named_entities"], qr["negation_flip"]]):
            st.warning(f"Risk scan — numbers missing: {qr['numbers']}, dates missing: {qr['dates']}, names missing: {qr['named_entities']}, negation flip: {qr['negation_flip']}")

        # Spoken feedback (EN + AR)
        st.subheader("Listen to Feedback")
        fb_en = enriched.get("spoken_feedback_en") or "Good effort. Review the corrections and try again."
        fb_ar = enriched.get("spoken_feedback_ar") or "عمل جيد. راجع التصحيحات ثم حاول مجددا."
        try:
            mp3_target = tts_bytes(enriched.get("corrected", working_text), lang_code=tts_target_lang)
            st.audio(mp3_target, format="audio/mp3")
            st.download_button("Download corrected (target)", mp3_target, file_name="corrected_target.mp3")
        except Exception as e:
            st.warning(f"TTS failed on corrected text: {e}")
        try:
            mp3_en = tts_bytes(fb_en, lang_code="en")
            st.audio(mp3_en, format="audio/mp3")
            st.download_button("Download feedback (EN)", mp3_en, file_name="feedback_en.mp3")
        except Exception as e:
            st.warning(f"English TTS failed: {e}")
        try:
            mp3_ar = tts_bytes(fb_ar, lang_code="ar")
            st.audio(mp3_ar, format="audio/mp3")
            st.download_button("Download feedback (AR)", mp3_ar, file_name="feedback_ar.mp3")
        except Exception as e:
            st.warning(f"Arabic TTS failed: {e}")

        # Persist submission if exercise chosen
        if chosen and supabase_client():
            sb.table("submissions").insert({
                "exercise_id": chosen["id"],
                "student_id": st.session_state.get("student_id", "anon"),
                "answer": working_text,
                "feedback": enriched
            }).execute()
            st.info("Submission saved.")
else:
    st.caption("Tip: Add OPENAI_API_KEY in Secrets to unlock richer feedback. Supabase is optional for exercises.")
