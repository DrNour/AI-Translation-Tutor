# Bidirectional Translation Tutor: Arabic ↔ English
# Heuristics for negation transfer, article use, false friends, prepositions, and synonym nudges.

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

st.set_page_config(page_title="Arabic ↔ English Translation Tutor", page_icon="🗣️", layout="centered")

# =============================
# Arabic helpers & resources
# =============================
AR_NEGATIONS = {"لا", "لم", "لن", "ما", "ليس", "بدون", "غير"}
EN_NEG_TOKENS = {" not ", "n't", " never ", " no "}

DIACRITICS = re.compile("[\u0617-\u061A\u064B-\u0652\u0657-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"

# False friends (triggered when certain English words appear with an Arabic source cue, or vice versa)
FALSE_FRIENDS_AR_EN = {
    "فعليًا": ("actually", "Use ‘actually’ for emphasis/contrast, not for ‘currently’. For ‘currently’, use ‘currently/at the moment’."),
    "حساس": ("sensitive", "‘Sensitive’ is not ‘sensible’. ‘Sensible’ = reasonable; ‘sensitive’ = easily affected."),
    "أخيرا": ("finally", "Use ‘finally’ for end of a process, not for ‘eventually’. ‘Eventually’ = at some point in the future."),
}

# Curated English synonyms for overused words
OVERUSED_ENGLISH = {
    "good": ["strong", "effective", "solid", "helpful", "beneficial", "favorable"],
    "very": ["extremely", "highly", "particularly", "remarkably"],
    "important": ["crucial", "vital", "essential", "key", "significant"],
    "big": ["large", "major", "substantial", "considerable"],
    "small": ["minor", "slight", "limited", "compact"],
    "make": ["create", "produce", "cause", "prepare", "build"],
    "do": ["perform", "carry out", "conduct", "execute"],
}

# Curated Arabic synonyms to reduce repetition
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
    ("في", "تُستخدم مع الزمن العام/الأماكن الواسعة: في 2020، في يوليو، في المدينة"),
    ("على", "تُستخدم مع الأيام/الأسطح: على الطاولة، على الجدار (انتبه: ‘on Monday’ → ‘يوم الاثنين’)"),
    ("عند", "تُستخدم مع الأوقات/النقاط المحددة: عند الساعة السابعة، عند الباب"),
]

def ar_normalize(s: str) -> str:
    s = DIACRITICS.sub("", s or "")
    s = s.replace(TATWEEL, "")
    s = re.sub("[\u0622\u0623\u0625]", "ا", s)  # unify alef forms
    s = s.replace("ى", "ي").replace("ئ", "ي")
    return s

# =============================
# LanguageTool setup (rate-limit safe)
# =============================

def get_lt_tool(lang_code: str = "en-US"):
    """
    Prefer the hosted Public API (works on Streamlit Cloud).
    If LT_API_URL and/or LT_API_KEY are provided via secrets/env, use them.
    Falls back to default public endpoint, then local server.
    """
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
        # Common: rate limit/timeouts → return empty and let fallback kick in
        st.warning("Grammar server is busy (rate limit). Using lightweight checks for now.")
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
# OpenAI helpers (optional)
# =============================

def has_openai() -> bool:
    key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return bool(key)

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
    out = sorted(list(syns))
    return out[:10]

# =============================
# Heuristics: Arabic → English
# =============================

def heuristics_ar_to_en(ar_source: str, en_student: str) -> Tuple[List[str], Dict[str, List[str]]]:
    notes: List[str] = []
    ar_norm = ar_normalize(ar_source)
    en_low = (en_student or "").lower()

    # Negation transfer
    if any(tok in ar_norm for tok in AR_NEGATIONS):
        if not any(tok in en_low for tok in EN_NEG_TOKENS) and not en_low.startswith("no "):
            notes.append("Source has negation; ensure the English includes ‘not/never/no’.")

    # Definite article heuristic: Arabic definite ‘ال’ often maps to ‘the’
    if re.search(r"\bال\w+", ar_norm) and re.search(r"\b(the)\b", en_low) is None:
        notes.append("Arabic definite ‘ال’ may require ‘the’ in English; check article use.")

    # Preposition reminders
    for p, tip in PREP_TIPS_EN:
        if re.search(fr"\b{p}\b", en_low):
            notes.append(f"Preposition ‘{p}’: {tip}")
            break

    # False friends
    for ar_key, (ff, msg) in FALSE_FRIENDS_AR_EN.items():
        if ar_key in ar_norm and re.search(fr"\b{ff}\b", en_low):
            notes.append(f"Possible false friend: ‘{ff}’. {msg}")

    # Overuse → synonyms
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
# Heuristics: English → Arabic
# =============================

def heuristics_en_to_ar(en_source: str, ar_student: str) -> Tuple[List[str], Dict[str, List[str]]]:
    notes: List[str] = []
    en_low = f" { (en_source or '').lower() } "
    ar_norm = ar_normalize(ar_student or "")

    # Negation transfer: English not/never/no → Arabic negation
    if any(tok in en_low for tok in EN_NEG_TOKENS) and not any(tok in ar_norm for tok in AR_NEGATIONS):
        notes.append("Source is negative; add Arabic negation (لا/لم/لن/ما/ليس) where appropriate.")

    # Definite article: English "the" → Arabic definite "ال"
    if " the " in en_low and not re.search(r"\bال\w+", ar_norm):
        notes.append("Source uses ‘the’; consider the Arabic definite article ‘ال’ when needed.")

    # Prepositions: quick reminders
    for p, tip in PREP_TIPS_AR:
        if p in ar_student:
            notes.append(f"تلميح حرف الجر ‘{p}’: {tip}")
            break

    # Numbers & named entities: encourage faithful transfer
    if re.search(r"\d", en_source or "") and not re.search(r"\d", ar_student or ""):
        notes.append("تأكد من نقل الأرقام كما هي (أو بالأرقام العربية إذا لزم).")
    # Minimal check for transliteration vs. translation of names (very rough)
    if re.search(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", en_source or ""):
        notes.append("حافظ على الأسماء العلم إمّا منقولة صوتيًا بدقة أو مترجمة إذا لها مقابل شائع.")

    # Overuse → Arabic synonyms
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
# LLM-enhanced coaching
# =============================

def llm_feedback(student_text: str, source_text: str, mode: str) -> Dict[str, Any]:
    """
    mode: "ar→en" or "en→ar"
    Returns keys:
      corrected, explanations[], fluency_score, word_choice_notes[], synonym_suggestions{}, spoken_feedback_en, spoken_feedback_ar
    """
    data = {
        "corrected": "",
        "explanations": [],
        "fluency_score": 0,
        "word_choice_notes": [],
        "synonym_suggestions": {},
        "spoken_feedback_en": "",
        "spoken_feedback_ar": "",
    }

    # Baseline grammar/style only when English is the TARGET (we have tooling there)
    if mode == "ar→en":
        lt_issues = analyze_with_languagetool_en(student_text)
    else:
        lt_issues = []  # Arabic target: rely on heuristics + LLM

    if has_openai():
        client = init_openai()
        if client:
            target_lang = "English" if mode == "ar→en" else "Arabic"
            system_prompt = (
                f"You are a bilingual Arabic↔English translation tutor. "
                f"Target language: {target_lang}. "
                "Compare the student's translation to the source, then return concise, classroom-friendly feedback. "
                "Return JSON with keys: corrected, explanations[{type,original,suggestion,explanation}], "
                "fluency_score(0-100), word_choice_notes[], synonym_suggestions{word:[...]}, "
                "spoken_feedback_en, spoken_feedback_ar."
            )
            user_payload = {
                "direction": mode,
                "source_text": source_text,
                "student_translation": student_text,
                "focus": ["grammar", "syntax", "fluency", "word choice", "faithfulness"],
                "style": "brief and constructive",
            }
            try:
                if client == "legacy":
                    resp = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.3,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                        ],
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content
                else:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        temperature=0.3,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                        ],
                        response_format={"type": "json_object"}
                    )
                    raw = resp.choices[0].message.content
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    data.update(parsed)
            except Exception as e:
                st.warning(f"LLM feedback unavailable, using rule-based checks only. ({e})")

    # Fallback/populate if LLM missing fields
    if not data.get("corrected"):
        data["corrected"] = student_text
        if mode == "ar→en":
            exps = []
            for i in lt_issues[:12]:
                exps.append({
                    "type": i.get("type", "issue"),
                    "original": "",
                    "suggestion": (i.get("replacements") or [""])[0] if isinstance(i.get("replacements"), list) else "",
                    "explanation": i.get("message", ""),
                })
            data["explanations"] = exps
            data["fluency_score"] = max(0, 100 - len(lt_issues)*3)
            data["spoken_feedback_en"] = (
                "Good effort. Watch articles (a/the), tense consistency, and prepositions. "
                "Try the synonym suggestions to avoid repetition, then read the corrected line aloud."
            )
            data["spoken_feedback_ar"] = (
                "عمل جيد. انتبه لاستخدام أدوات التعريف، واتساق الأزمنة، وحروف الجر. "
                "جرّب المرادفات المقترحة ثم أعد قراءة الجملة المصححة."
            )
        else:  # en→ar
            data["explanations"] = data.get("explanations", [])
            data["fluency_score"] = 75  # soft default without Arabic grammar scoring
            data["spoken_feedback_en"] = (
                "Nice translation. Check definite articles, negation, and names/numbers. "
                "Aim for concise, natural Arabic phrasing."
            )
            data["spoken_feedback_ar"] = (
                "ترجمة موفّقة. راجع أدوات التعريف (ال)، ونقل النفي بدقة، والأسماء/الأرقام، "
                "وحاول صياغة عربية طبيعية ومقتضبة."
            )

    # Merge heuristics & synonyms based on direction
    if mode == "ar→en":
        notes, syns = heuristics_ar_to_en(source_text, student_text)
    else:
        notes, syns = heuristics_en_to_ar(source_text, student_text)

    if notes:
        data.setdefault("word_choice_notes", [])
        data["word_choice_notes"].extend(notes)
    if syns:
        data.setdefault("synonym_suggestions", {})
        for k, v in syns.items():
            existing = data["synonym_suggestions"].get(k, [])
            merged = list(dict.fromkeys(existing + v))[:10]
            data["synonym_suggestions"][k] = merged

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
# UI
# =============================

st.title("🗣️ Arabic ↔ English Translation Tutor")

mode = st.radio(
    "Direction",
    ["Arabic → English", "English → Arabic"],
    index=0,
    horizontal=True
)

if mode == "Arabic → English":
    st.markdown(
        "Speak or paste your **English** translation. Provide the **Arabic** source.\n\n"
        "You’ll get corrections on **grammar, syntax, fluency, and word choice**, plus synonym suggestions. "
        "You can also listen to the feedback (English & Arabic)."
    )
    col1, col2 = st.columns(2)
    with col1:
        source_text = st.text_area("النصّ العربي (Source in Arabic)", height=180, placeholder="ألصِق النص العربي هنا…")
    with col2:
        student_text = st.text_area("Your English translation", height=180, placeholder="Type or paste your English translation…")
    tts_target_lang = "en"
    mode_key = "ar→en"

else:
    st.markdown(
        "Speak or paste your **Arabic** translation. Provide the **English** source.\n\n"
        "You’ll get targeted feedback for **natural Arabic phrasing**, faithfulness, and common interference issues."
    )
    col1, col2 = st.columns(2)
    with col1:
        source_text = st.text_area("Source in English", height=180, placeholder="Paste the English source here…")
    with col2:
        student_text = st.text_area("ترجمتك العربية", height=180, placeholder="ألصِق ترجمتك العربية هنا…")
    tts_target_lang = "ar"
    mode_key = "en→ar"

st.divider()
rec = st.audio_input("Optional: Record your translation (mic)")

if st.button("Analyze & Coach"):
    if not source_text.strip():
        st.warning("Please paste the source text.")
        st.stop()

    working_text = (student_text or "").strip()
    if not working_text and rec is None:
        st.warning("Paste text or record audio for your translation.")
        st.stop()

    if rec is not None and not working_text:
        st.info("Audio provided but STT is not wired in this build. Paste text, or add your OpenAI key and connect Whisper if needed.")

    # Main feedback
    enriched = llm_feedback(working_text, source_text, mode_key)

    st.subheader("✅ Corrected Version")
    st.write(enriched.get("corrected", working_text))

    st.subheader(f"🧭 Feedback (Fluency: {enriched.get('fluency_score', 0)}/100)")
    expander = st.expander("Detailed issues")
    with expander:
        exps = enriched.get("explanations", [])
        if not exps:
            st.write("No issues found.")
        else:
            for i, e in enumerate(exps, 1):
                st.markdown(f"**{i}. {e.get('type','issue')}** — {e.get('explanation','')}")
                if e.get("original"): st.markdown(f"- Original: `{e['original']}`")
                if e.get("suggestion"): st.markdown(f"- Suggestion: `{e['suggestion']}`")

    wc = enriched.get("word_choice_notes", [])
    if wc:
        st.subheader("🎯 Notes & Strategy")
        for note in wc:
            st.write(f"- {note}")

    syns = enriched.get("synonym_suggestions", {})
    if syns:
        st.subheader("🧩 Synonym Suggestions")
        # For readability, show English terms bolded in ar→en, Arabic terms bolded in en→ar
        for w, options in syns.items():
            st.markdown(f"- **{w}** → {', '.join(options)}")

    # Spoken feedback (EN + AR)
    st.subheader("🔊 Listen to Feedback")
    fb_en = enriched.get("spoken_feedback_en") or "Good effort. Review the corrections and try again."
    fb_ar = enriched.get("spoken_feedback_ar") or "عمل جيد. راجع التصحيحات ثم حاول مجدداً."

    try:
        mp3_main = tts_bytes(enriched.get("corrected", working_text), lang_code=tts_target_lang)
        st.audio(mp3_main, format="audio/mp3")
        st.download_button(
            "Download corrected (target language)",
            mp3_main,
            file_name="corrected_target.mp3"
        )
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

    st.success("Done. Paste a new sample whenever you’re ready.")
else:
    st.caption("Tip: This build is bidirectional. Add OPENAI_API_KEY in Secrets to unlock richer feedback.")
