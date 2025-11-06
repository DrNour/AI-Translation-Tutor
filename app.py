# Heuristics for negation transfer from Arabic (لا/لم/لن/ما/ليس) -> English (not/never/no)
import os
import io
import json
import re
from typing import Dict, List, Any

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

st.set_page_config(page_title="Arabic→English Translation Tutor", page_icon="🗣️", layout="centered")

# =============================
# Arabic helpers
# =============================
AR_NEGATIONS = {"لا", "لم", "لن", "ما", "ليس", "بدون", "غير"}
DIACRITICS = re.compile("[\u0617-\u061A\u064B-\u0652\u0657-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"

FALSE_FRIENDS = {
    # Arabic token (normalized) -> suggestion note if English uses a common false friend
    # These fire when the English translation contains the false-friend word
    "فعليًا": ("actually", "Use ‘actually’ for emphasis/contrast, not for ‘currently’. For ‘currently’, use ‘currently/at the moment’."),
    "حساس": ("sensitive", "‘Sensitive’ is not ‘sensible’. ‘Sensible’ = reasonable; ‘sensitive’ = easily affected."),
    "أخيرا": ("finally", "Use ‘finally’ for end of a process, not for ‘eventually’. ‘Eventually’ = at some point in the future."),
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

PREP_TIPS = [
    ("in", "Months/years and large areas: in 2020, in July, in the city"),
    ("on", "Days/dates and surfaces: on Monday, on 5 May, on the table"),
    ("at", "Precise times/points: at 7 pm, at the door, at work"),
]

# Prefer the public API to avoid Java requirement on Streamlit Cloud
def get_lt_tool(lang_code: str = "en-US"):
    try:
        return lt.LanguageToolPublicAPI(lang_code)
    except Exception:
        return lt.LanguageTool(lang_code)

def ar_normalize(s: str) -> str:
    # Remove diacritics and tatweel; unify alef and ya forms
    s = DIACRITICS.sub("", s or "")
    s = s.replace(TATWEEL, "")
    s = re.sub("[\u0622\u0623\u0625]", "ا", s)  # different alef forms -> ا
    s = s.replace("ى", "ي").replace("ئ", "ي")
    return s

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
# Core analysis
# =============================

def analyze_with_languagetool(text: str, lang_code: str = "en-US") -> List[Dict[str, Any]]:
    tool = get_lt_tool(lang_code)
    matches = tool.check(text)
    issues = []
    for m in matches:
        issues.append({
            "type": m.ruleIssueType or "grammar",
            "message": m.message,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": [r.value for r in m.replacements][:5]
        })
    return issues

def wordnet_synonyms(word: str) -> List[str]:
    syns = set()
    for synset in wn.synsets(word, lang="eng"):
        for lemma in synset.lemmas():
            w = lemma.name().replace("_", " ")
            if w.lower() != word.lower():
                syns.add(w)
    out = sorted(list(syns))
    return out[:10]

def ar_en_heuristics(ar_source: str, en_student: str) -> Dict[str, Any]:
    """Lightweight checks tailored for Arabic→English."""
    notes = []
    ar_norm = ar_normalize(ar_source)
    en_low = (en_student or "").lower()

    # Negation transfer
    if any(tok in ar_norm for tok in AR_NEGATIONS):
        if not (" not " in en_low or "n't" in en_low or " never " in en_low or en_low.startswith("no ")):
            notes.append("Source has negation; ensure the English includes ‘not/never/no’.")

    # Definite article heuristic: Arabic definite ‘ال’ often maps to ‘the’
    if re.search(r"\bال\w+", ar_norm):
        if re.search(r"\b(the)\b", en_low) is None:
            notes.append("Arabic definite ‘ال’ may require ‘the’ in English; check article use.")

    # Preposition reminders (common errors from Arabic interference)
    for p, tip in PREP_TIPS:
        if re.search(fr"\b{p}\b", en_low):
            notes.append(f"Preposition ‘{p}’: {tip}")
            break

    # False friends
    for ar_key, (ff, msg) in FALSE_FRIENDS.items():
        if ar_key in ar_norm and re.search(fr"\b{ff}\b", en_low):
            notes.append(f"Possible false friend: ‘{ff}’. {msg}")

    # Overuse flags → synonym nudges
    synonym_suggestions = {}
    tokens = re.findall(r"[a-zA-Z']+", en_student or "")
    counts = {}
    for t in tokens:
        wl = t.lower()
        counts[wl] = counts.get(wl, 0) + 1
    for w, options in OVERUSED_ENGLISH.items():
        if counts.get(w, 0) >= 2 or w in counts:
            # merge WordNet to curated list
            merged = list(dict.fromkeys(options + wordnet_synonyms(w)))[:8]
            if merged:
                synonym_suggestions[w] = merged

    return {"notes": notes, "synonyms": synonym_suggestions}

def llm_feedback(student_text: str, source_text_ar: str) -> Dict[str, Any]:
    """LLM-enhanced feedback tailored to Arabic→English when API is available."""
    data = {
        "corrected": "",
        "explanations": [],
        "fluency_score": 0,
        "word_choice_notes": [],
        "synonym_suggestions": {},
        "spoken_feedback_en": "",
        "spoken_feedback_ar": "",
    }

    lt_issues = analyze_with_languagetool(student_text)

    if has_openai():
        client = init_openai()
        if client:
            system_prompt = (
                "You are a bilingual Arabic→English translation tutor. "
                "Compare the student's English translation to the Arabic source. "
                "Return concise, practical feedback suitable for classroom use. "
                "Return JSON with keys: corrected, explanations[{type,original,suggestion,explanation}], "
                "fluency_score(0-100), word_choice_notes[], synonym_suggestions{word:[...]}, "
                "spoken_feedback_en, spoken_feedback_ar."
            )
            user_payload = {
                "source_language": "Arabic",
                "target_language": "English",
                "source_text_ar": source_text_ar,
                "student_translation_en": student_text,
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

    # Fallback/populate if missing
    if not data.get("corrected"):
        data["corrected"] = student_text
        exps = []
        for i in lt_issues[:10]:
            exps.append({
                "type": i["type"],
                "original": "",
                "suggestion": i["replacements"][0] if i["replacements"] else "",
                "explanation": i["message"],
            })
        data["explanations"] = exps
        data["fluency_score"] = max(0, 100 - len(lt_issues)*3)
        data["spoken_feedback_en"] = (
            "Good effort. Focus on articles (a/the), tense consistency, and prepositions. "
            "Try the synonym suggestions to avoid repetition, then read the corrected line aloud."
        )
        data["spoken_feedback_ar"] = (
            "عمل جيد. انتبه لاستخدام أدوات التعريف، واتساق الأزمنة، وحروف الجر. "
            "جرّب المرادفات المقترحة ثم أعد قراءة الجملة المصححة."
        )

    # Merge Arabic→English heuristics
    heur = ar_en_heuristics(source_text_ar, student_text)
    extra_notes = heur.get("notes", [])
    if extra_notes:
        data.setdefault("word_choice_notes", [])
        data["word_choice_notes"].extend(extra_notes)
    syns = heur.get("synonyms", {})
    if syns:
        data.setdefault("synonym_suggestions", {})
        # don't clobber LLM suggestions; merge
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

st.title("🗣️ Arabic → English Translation Tutor")

st.markdown(
    "Speak or paste your **English** translation. Provide the **Arabic** source.\n\n"
    "You’ll get corrections on **grammar, syntax, fluency, and word choice**, plus synonym suggestions. "
    "You can also listen to the feedback (English & Arabic)."
)

col1, col2 = st.columns(2)
with col1:
    source_text_ar = st.text_area("النصّ العربي (Source in Arabic)", height=140, placeholder="ألصِق النص العربي هنا…")
with col2:
    student_text_en = st.text_area("Your English translation", height=140, placeholder="Type or paste your English translation…")

st.divider()

rec = st.audio_input("Optional: Record your English translation (mic)")

if st.button("Analyze & Coach"):
    if not source_text_ar.strip():
        st.warning("Please paste the Arabic source text.")
        st.stop()

    # This build uses text input for analysis.
    # If you want STT, wire Whisper using your OPENAI_API_KEY.
    working_text = (student_text_en or "").strip()
    if not working_text and rec is None:
        st.warning("Paste text or record audio for your translation.")
        st.stop()

    if rec is not None and not working_text:
        st.info("Audio was provided but STT is not enabled in this build. Paste text, or add your OpenAI key and wire Whisper if needed.")

    # Main feedback
    enriched = llm_feedback(working_text, source_text_ar)

    st.subheader("✅ Corrected Version (English)")
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
        st.subheader("🧩 Synonym Suggestions (English)")
        for w, options in syns.items():
            st.markdown(f"- **{w}** → {', '.join(options)}")

    # Spoken feedback (English + Arabic)
    st.subheader("🔊 Listen to Feedback")
    fb_en = enriched.get("spoken_feedback_en") or enriched.get("spoken_feedback") or "Good effort. Review the corrections and try again."
    fb_ar = enriched.get("spoken_feedback_ar") or "عمل جيد. راجع التصحيحات ثم حاول مجدداً."
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

    # Optional: corrected sentence TTS (English)
    st.subheader("🔁 Listen to Corrected Sentence (EN)")
    try:
        mp3_corr = tts_bytes(enriched.get("corrected", working_text), lang_code="en")
        st.audio(mp3_corr, format="audio/mp3")
        st.download_button("Download corrected (EN)", mp3_corr, file_name="corrected_en.mp3")
    except Exception as e:
        st.warning(f"TTS failed on corrected text: {e}")

    st.success("Done. Paste new text whenever you’re ready.")
else:
    st.caption("Tip: This build focuses on Arabic→English. To enable speech-to-text, add OPENAI_API_KEY in Secrets and wire Whisper.")
