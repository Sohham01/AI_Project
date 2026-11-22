"""Pure-stdlib Indian-Reddit Language, Script & Code-Mix Identifier (Task 5).

Supports English, Hindi, Bengali, Odia, Hinglish (hi-en), Banglish (bn-en),
Roman-Odia code-mixed (or-en), and other major Indian scripts with zero
third-party dependencies so it runs identically in offline ML pipelines,
Docker, and Vercel serverless endpoints.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_ALPHA_TOK_RE = re.compile(r"[a-zA-Z]+")

# Unicode block detectors for Indic scripts
_SCRIPT_RANGES: Dict[str, tuple[str, str]] = {
    "devanagari": ("\u0900", "\u097f"),
    "bengali": ("\u0980", "\u09ff"),
    "odia": ("\u0b00", "\u0b7f"),
    "gurmukhi": ("\u0a00", "\u0a7f"),
    "gujarati": ("\u0a80", "\u0aff"),
    "tamil": ("\u0b80", "\u0bff"),
    "telugu": ("\u0c00", "\u0c7f"),
    "kannada": ("\u0c80", "\u0cff"),
    "malayalam": ("\u0d00", "\u0d7f"),
}

_SCRIPT_TO_LANG: Dict[str, str] = {
    "devanagari": "hi",
    "bengali": "bn",
    "odia": "or",
    "gurmukhi": "pa",
    "gujarati": "gu",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
}

# Unambiguous Romanized Hindi / Hinglish markers (strictly excluding English collisions)
STRONG_HINGLISH = frozenset({
    "hai", "hain", "nahi", "nhi", "kya", "kyu", "kyun", "kaise", "kaisa", "kaisi",
    "bhai", "bhaiya", "yaar", "toh", "bhi", "mein", "mera", "tera", "aur", "lekin",
    "agar", "accha", "achha", "acha", "theek", "thik", "karo", "karna", "karke",
    "karega", "karenge", "waise", "waisa", "itna", "itni", "boht", "zyaada",
    "gaya", "gayi", "gaye", "dekho", "dekhna", "dekha", "bola", "bolna", "padho",
    "padhna", "padhai", "mila", "mile", "milta", "chahiye", "paisa", "paise",
    "sahi", "galat", "bahut", "bohot", "zyada", "abhi", "pehle", "phir", "kuch",
    "kuchh", "koi", "kaun", "wala", "wali", "wale", "kaam", "banda", "bande",
    "jugaad", "haan", "shukriya", "hoga", "hogi", "honge", "raha", "rahi", "rahe",
    "hota", "hoti", "jao", "jaana", "jaao", "rha", "rhi", "rhe", "krna", "kro",
    "mtlb", "matlab", "shayad", "bilkul", "thoda", "batao", "pucha", "pata",
    "aayi", "aayegi", "aaegi", "aagyi", "baat", "yaha", "waha", "kitna", "kab",
    "udhar", "idhar", "unka", "unko", "unhe", "apna", "apne", "khud", "lagta",
    "chalo", "fir", "yhi", "kiska", "kisko", "kahan", "dhyan", "bkl", "abe",
    "arre", "chutiyapa", "dost", "paglu", "bhoot", "jaruri", "isliye", "kyuki",
    "taaki", "kudh", "sabhi", "horaha", "uthare", "utha", "nahin", "harkatein",
    "dekhkar", "karwa", "hogyi", "karun", "sakti", "mujhe", "farak", "padhta",
    "hoon", "aap", "dekh", "wli", "phele", "daali", "ismi", "khilao", "chor",
    "bhar", "bahar", "puch", "karle", "lelo", "bata", "hum", "tum", "woh", "yeh",
})

# Distinctive Romanized Bengali / Banglish markers (avoiding English & Hindi collisions)
STRONG_BANGLISH = frozenset({
    "keno", "kothay", "kokhon", "kemon", "hobe", "holo", "hoyeche", "hoye",
    "ache", "achhe", "nei", "ami", "aami", "tumi", "apni", "amar", "tomar",
    "bhalo", "bhaloi", "kharap", "korcho", "korbe", "korbo", "korte", "koreche",
    "bolcho", "bolbo", "jabe", "gelo", "ekhane", "okhane", "ekhon", "giye",
    "pawa", "taka", "bujhte", "parchi", "thake", "koro", "jani", "jano",
    "dada", "bangla", "kolkata", "kemon", "ektu", "onek", "shob", "kichu",
})

# Distinctive Romanized Odia markers (avoiding English & Hindi collisions)
STRONG_ROMAN_ODIA = frozenset({
    "kana", "kemiti", "kebe", "kouthi", "achi", "achhi", "tume", "amme",
    "apana", "bhala", "kariba", "karibi", "jiba", "jibi", "etha", "setha",
    "ebe", "hau", "aaji", "thila", "thile", "heba", "hela", "mote", "taku",
    "amara", "tamara", "bujhili", "heichi", "katha", "pila", "pilanka",
    "bhubaneswar", "odia", "bhaina", "bhauni",
})

# Secondary shared particles that boost Bengali/Odia confidence when paired with a strong cue
_SECONDARY_BN_OR = frozenset({"ki", "na", "jay", "to", "ta", "ora", "tara", "se"})

ENGLISH_OR_MBA_ANCHORS = frozenset({
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "have",
    "not", "but", "from", "they", "will", "would", "could", "should", "what", "when",
    "where", "which", "about", "there", "just", "like", "more", "some", "very", "only",
    "also", "can", "all", "any", "get", "got", "one", "how", "why", "yes", "no", "ok",
    "okay", "same", "true", "agree", "thanks", "thank", "congrats", "congratulations",
    "please", "pls", "bro", "sir", "check", "mail", "call", "email", "link", "dm",
    "iim", "cat", "xat", "gmat", "jee", "ipmat", "wl", "waitlist", "waitlisted",
    "fees", "fee", "loan", "roi", "lpa", "ctc", "sip", "placement", "placements",
    "hostel", "mess", "campus", "mba", "ug", "bs", "dsai", "mpp", "sambalpur",
    "mumbai", "lucknow", "shillong", "udaipur", "trichy", "raipur", "ranchi",
    "sirmaur", "jammu", "bodhgaya", "kashipur", "vizag", "nagpur", "tapmi", "gim",
    "glim", "imi", "imt", "fore", "nmims", "isb", "iit", "nit", "college", "clg",
    "offer", "list", "round", "merit", "gen", "general", "obc", "ews", "sc", "st",
    "gem", "gnem", "gef", "gnef", "profile", "percentile", "score", "rank", "cutoff",
    "interview", "pi", "gd", "wat", "cap", "sap", "batch", "sem", "semester", "term",
    "classes", "prof", "faculty", "exam", "exams", "cgpa", "refund", "portal", "seat",
    "admission", "admissions", "convert", "converted", "rejected", "cfbr", "imo",
    "tbh", "idk", "ig", "rn", "fr", "ngl", "lol", "lmao", "wtf", "fyi", "ama",
    "actually", "hectic", "worth", "summer", "literally", "seniors", "helped",
})


def detect_script_profile(text: str) -> Dict[str, Any]:
    """Count Latin and Indic Unicode script characters deterministically."""
    raw = str(text or "")
    counts: Dict[str, int] = {}
    latin_chars = 0
    for ch in raw:
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            latin_chars += 1
            continue
        for script_name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= ch <= hi:
                counts[script_name] = counts.get(script_name, 0) + 1
                break
    if latin_chars:
        counts["latin"] = latin_chars

    if not counts:
        return {"primary_script": "unknown", "scripts_present": [], "char_counts": {}}

    indic_present = [s for s in counts if s != "latin"]
    if indic_present and latin_chars >= 2:
        dominant_indic = max(indic_present, key=lambda s: counts[s])
        return {
            "primary_script": "mixed",
            "dominant_indic_script": dominant_indic,
            "scripts_present": sorted(counts.keys()),
            "char_counts": counts,
        }
    primary = max(counts.keys(), key=lambda s: counts[s])
    return {
        "primary_script": primary,
        "dominant_indic_script": primary if primary != "latin" else None,
        "scripts_present": sorted(counts.keys()),
        "char_counts": counts,
    }


def detect_indian_reddit_language(text: str, script_detector: Any = None) -> Dict[str, Any]:
    """Identify language, script, and code-mixing for Indian student Reddit discussions.

    Handles native Indic scripts (Devanagari, Bengali, Odia, etc.), script-mixed
    text, Romanized Hinglish (`hi-en`), Banglish (`bn-en`), Roman-Odia (`or-en`),
    and English (`en`) without false European-language misclassifications.
    """
    raw = str(text or "")
    script_profile = detect_script_profile(raw)
    primary_script = script_profile["primary_script"]
    dominant_indic = script_profile.get("dominant_indic_script")

    # Fallback to external script_detector only if provided and our profile is unknown
    if primary_script == "unknown" and script_detector is not None:
        s_res = script_detector.detect(raw)
        primary_script = str(getattr(s_res, "script", "unknown") or "unknown").lower()

    latin_toks = [t.lower() for t in _ALPHA_TOK_RE.findall(raw)]
    n_tok = len(latin_toks)

    # 1. Native Indic scripts (Devanagari, Bengali, Odia, Tamil, Telugu, etc.)
    if dominant_indic and dominant_indic in _SCRIPT_TO_LANG:
        indic_lang = _SCRIPT_TO_LANG[dominant_indic]
        if n_tok >= 1 and primary_script == "mixed":
            return {
                "language": f"{indic_lang}-en",
                "script": "mixed",
                "is_code_mixed": True,
                "detected_languages": [indic_lang, "en"],
                "indic_script": dominant_indic,
                "matched_markers": [],
            }
        return {
            "language": indic_lang,
            "script": dominant_indic,
            "is_code_mixed": False,
            "detected_languages": [indic_lang],
            "indic_script": dominant_indic,
            "matched_markers": [],
        }

    # 2. Romanized Indic code-mixing (Hinglish, Banglish, Roman-Odia)
    hi_hits = [t for t in latin_toks if t in STRONG_HINGLISH]
    bn_hits = [t for t in latin_toks if t in STRONG_BANGLISH]
    or_hits = [t for t in latin_toks if t in STRONG_ROMAN_ODIA]
    sec_hits = [t for t in latin_toks if t in _SECONDARY_BN_OR]
    en_hits = [t for t in latin_toks if t in ENGLISH_OR_MBA_ANCHORS]

    # Pick strongest Romanized Indic signal (distinctive Bengali/Odia win over Hindi if stronger)
    bn_score = len(bn_hits) + (0.5 * len(sec_hits) if bn_hits else 0.0)
    or_score = len(or_hits) + (0.5 * len(sec_hits) if or_hits else 0.0)
    hi_score = float(len(hi_hits))

    best_lang = None
    best_hits: List[str] = []
    if bn_hits and bn_score >= max(hi_score, or_score) and (len(bn_hits) >= 2 or (len(bn_hits) >= 1 and len(sec_hits) >= 1) or (n_tok <= 16 and len(bn_hits) >= 1)):
        best_lang = "bn"
        best_hits = sorted(set(bn_hits))
    elif or_hits and or_score >= max(hi_score, bn_score) and (len(or_hits) >= 2 or (len(or_hits) >= 1 and len(sec_hits) >= 1) or (n_tok <= 16 and len(or_hits) >= 1)):
        best_lang = "or"
        best_hits = sorted(set(or_hits))
    elif hi_hits and (len(hi_hits) >= 2 or (n_tok <= 16 and len(hi_hits) >= 1)):
        best_lang = "hi"
        best_hits = sorted(set(hi_hits))

    if best_lang is not None:
        raw_hits_count = len(bn_hits) if best_lang == "bn" else (len(or_hits) if best_lang == "or" else len(hi_hits))
        indic_ratio = raw_hits_count / max(n_tok, 1)
        if indic_ratio >= 0.55 and len(en_hits) <= 1:
            return {
                "language": best_lang,
                "script": "latin",
                "is_code_mixed": False,
                "detected_languages": [best_lang],
                "indic_script": None,
                "matched_markers": best_hits,
            }
        return {
            "language": f"{best_lang}-en",
            "script": "latin",
            "is_code_mixed": True,
            "detected_languages": [best_lang, "en"],
            "indic_script": None,
            "matched_markers": best_hits,
        }

    # 3. English / Indian MBA-Reddit English vs. short undetermined replies
    if en_hits or n_tok >= 4:
        return {
            "language": "en",
            "script": "latin",
            "is_code_mixed": False,
            "detected_languages": ["en"],
            "indic_script": None,
            "matched_markers": [],
        }

    return {
        "language": "und",
        "script": primary_script if primary_script else "unknown",
        "is_code_mixed": False,
        "detected_languages": [],
        "indic_script": None,
        "matched_markers": [],
    }
