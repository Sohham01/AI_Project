"""Pure-stdlib Indian-Reddit Language, Script, Span & Romanized-Text Processor (V6).

Supports English, Hindi, Bengali, Odia, Hinglish (`hi-en`), Banglish (`bn-en`),
Roman-Odia (`or-en`), and other major Indian scripts with zero third-party
dependencies so it runs identically in offline ML pipelines, Docker, and Vercel
serverless endpoints.

V6 capabilities:
1. Document-level language & script separation (Latin-script Hindi is never
   auto-labelled English).
2. Span/sentence-level language & script detection (`detect_language_spans`)
   preserving exact character offsets and representing code-mixing + uncertainty
   explicitly rather than forcing one language label.
3. Explicit confidence estimation & abstention flags (`confidence`, `abstain`,
   `uncertainty_flag`).
4. Non-destructive Romanized Hindi/Bengali/Odia phonetic-variant & spelling
   normalization (`normalize_romanized_indic`), strictly preserving `original_text`
   and returning structured `transformations` metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

_ALPHA_TOK_RE = re.compile(r"[a-zA-Z]+")
_SPAN_SPLIT_RE = re.compile(r"(?<=[.!?\n;।])\s+|\n+")

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
    "hai", "hain", "nahi", "nhi", "nai", "nahin", "kya", "kyu", "kyun", "kaise", "kaisa", "kaisi",
    "bhai", "bhaiya", "yaar", "yarr", "toh", "bhi", "mein", "mera", "tera", "aur", "lekin",
    "agar", "accha", "achha", "acha", "theek", "thik", "karo", "kro", "karna", "krna", "karke",
    "karega", "karenge", "waise", "waisa", "itna", "itni", "boht", "bht", "zyaada",
    "gaya", "gayi", "gaye", "gya", "gyi", "dekho", "dekhna", "dekha", "bola", "bolna", "padho",
    "padhna", "padhai", "mila", "mile", "milta", "chahiye", "paisa", "paise", "rupiye", "laakhon",
    "sahi", "galat", "bahut", "bohot", "zyada", "abhi", "pehle", "phele", "phir", "fir", "kuch",
    "kuchh", "koi", "kaun", "wala", "wali", "wale", "kaam", "banda", "bande",
    "jugaad", "haan", "shukriya", "hoga", "hogi", "honge", "raha", "rahi", "rahe",
    "hota", "hoti", "jao", "jaana", "jaao", "rha", "rhi", "rhe",
    "mtlb", "matlab", "shayad", "bilkul", "thoda", "batao", "pucha", "pata",
    "aayi", "aayegi", "aaegi", "aagyi", "aaya", "baat", "yaha", "waha", "kitna", "kab",
    "udhar", "idhar", "unka", "unko", "unhe", "inka", "inke", "apna", "apne", "khud", "lagta",
    "chalo", "yhi", "kiska", "kisko", "kahan", "dhyan", "bkl", "abe",
    "arre", "chutiyapa", "dost", "paglu", "bhoot", "jaruri", "isliye", "kyuki",
    "taaki", "kudh", "sabhi", "horaha", "uthare", "utha", "harkatein",
    "dekhkar", "karwa", "hogya", "hogyi", "karun", "sakti", "mujhe", "farak", "padhta",
    "hoon", "aap", "dekh", "wli", "daali", "ismi", "khilao", "chor",
    "bhar", "bharo", "bahar", "puch", "karle", "lelo", "bata", "hum", "tum", "woh", "yeh",
    "sirf", "lenge", "lete", "alag", "bach", "jaise", "karne", "lage", "khaas", "usme", "reh",
    "baar", "chud", "guru", "mat", "kar", "kardo", "bhej", "samajh", "sab", "bas", "bass",
    "achhi", "acchi", "jab", "tabhi", "mil", "jaye", "jaaye", "bdhiya", "badhiya", "zindagi",
    "padega", "padegi", "karoge", "sakta", "sakte", "bina", "liye", "kyunki", "unki", "unke",
    "meri", "mere", "teri", "tere", "tumhe", "tujhe", "usko", "isko", "iski", "uski",
    "humara", "hamara", "tumhara", "yahan", "wahan", "bahot", "shi", "lga", "lag", "rehta", "rehti",
})

# Distinctive Romanized Bengali / Banglish markers (avoiding English & Hindi collisions)
STRONG_BANGLISH = frozenset({
    "keno", "kothay", "kothai", "kokhon", "kemon", "hobe", "hbe", "holo", "hlo", "hoyeche", "hoye",
    "ache", "achhe", "nei", "ami", "aami", "tumi", "apni", "amar", "tomar",
    "bhalo", "bhaloi", "valo", "kharap", "korcho", "korbe", "korbo", "korte", "koreche", "korche",
    "bolcho", "bolbo", "jabe", "gelo", "ekhane", "okhane", "ekhon", "giye",
    "pawa", "taka", "bujhte", "parchi", "thake", "koro", "jani", "jano",
    "dada", "bangla", "kolkata", "ektu", "onek", "shob", "sob", "kichu", "bochor", "khub",
})

# Distinctive Romanized Odia markers (avoiding English & Hindi collisions; "achhi" is shared with Hindi so excluded)
STRONG_ROMAN_ODIA = frozenset({
    "kana", "kemiti", "kemti", "kebe", "kouthi", "achi", "tume", "amme",
    "apana", "bhala", "kariba", "karibi", "jiba", "jibi", "etha", "setha",
    "ebe", "hau", "aaji", "aji", "thila", "thile", "heba", "hela", "mote", "taku",
    "amara", "tamara", "bujhili", "heichi", "heigala", "katha", "pila", "pilanka",
    "bhubaneswar", "odia", "bhaina", "bhauni", "besi", "ra", "sabu", "pasuchu",
    "karuchi", "jaichi", "kichi",
})

# Secondary shared particles that boost Bengali/Odia confidence when paired with a strong cue
_SECONDARY_BN_OR = frozenset({"ki", "na", "jay", "to", "ta", "ora", "tara", "se", "ei", "er", "ar"})

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
    "package", "packages", "wait", "mostly", "try", "time", "iims", "job", "work", "class",
})

# Non-destructive phonetic & spelling variant normalizer table for Romanized Indic text
ROMANIZED_VARIANT_RULES: Dict[str, Tuple[str, str, str]] = {
    # token -> (canonical_form, language, category)
    # Romanized Hindi (Hinglish) phonetic variants
    "nhi": ("nahi", "hi", "phonetic_contraction"),
    "nahin": ("nahi", "hi", "phonetic_variant"),
    "nai": ("nahi", "hi", "phonetic_variant"),
    "bohot": ("bahut", "hi", "phonetic_variant"),
    "boht": ("bahut", "hi", "phonetic_contraction"),
    "bht": ("bahut", "hi", "phonetic_contraction"),
    "acha": ("accha", "hi", "phonetic_variant"),
    "achha": ("accha", "hi", "phonetic_variant"),
    "krna": ("karna", "hi", "phonetic_contraction"),
    "kro": ("karo", "hi", "phonetic_contraction"),
    "kyu": ("kyun", "hi", "phonetic_variant"),
    "kyuki": ("kyunki", "hi", "phonetic_variant"),
    "rha": ("raha", "hi", "phonetic_contraction"),
    "rhi": ("rahi", "hi", "phonetic_contraction"),
    "rhe": ("rahe", "hi", "phonetic_contraction"),
    "mtlb": ("matlab", "hi", "phonetic_contraction"),
    "yarr": ("yaar", "hi", "elongated_spelling"),
    "hogya": ("ho gaya", "hi", "fused_compound"),
    "hogyi": ("ho gayi", "hi", "fused_compound"),
    "gya": ("gaya", "hi", "phonetic_contraction"),
    "gyi": ("gayi", "hi", "phonetic_contraction"),
    "kuchh": ("kuch", "hi", "phonetic_variant"),
    "thik": ("theek", "hi", "phonetic_variant"),
    "phele": ("pehle", "hi", "phonetic_variant"),
    "fir": ("phir", "hi", "phonetic_variant"),
    "zyaada": ("zyada", "hi", "elongated_spelling"),
    "kudh": ("khud", "hi", "transposition_typo"),
    "aagyi": ("aa gayi", "hi", "fused_compound"),
    "horaha": ("ho raha", "hi", "fused_compound"),
    "wli": ("wali", "hi", "phonetic_contraction"),
    "yhi": ("yahi", "hi", "phonetic_contraction"),
    "paissaa": ("paisa", "hi", "elongated_spelling"),
    "paisssaa": ("paisa", "hi", "elongated_spelling"),
    # Romanized Bengali (Banglish) phonetic variants
    "valo": ("bhalo", "bn", "phonetic_variant"),
    "bhaloi": ("bhalo", "bn", "emphatic_variant"),
    "achhe": ("ache", "bn", "phonetic_variant"),
    "aami": ("ami", "bn", "elongated_spelling"),
    "kothai": ("kothay", "bn", "phonetic_variant"),
    "hbe": ("hobe", "bn", "phonetic_contraction"),
    "hlo": ("holo", "bn", "phonetic_contraction"),
    "korche": ("koreche", "bn", "phonetic_variant"),
    "shob": ("sob", "bn", "phonetic_variant"),
    # Romanized Odia phonetic variants
    "kemti": ("kemiti", "or", "phonetic_contraction"),
    "aaji": ("aji", "or", "elongated_spelling"),
    "achhi": ("acchi", "hi", "phonetic_variant"),
    "bdhiya": ("badhiya", "hi", "phonetic_contraction"),
    "shi": ("sahi", "hi", "phonetic_contraction"),
    "lga": ("laga", "hi", "phonetic_contraction"),
    # Indian student Reddit abbreviations (non-destructive, excluding CAT/IIM acronyms)
    "clg": ("college", "en-in", "abbreviation"),
    "clgs": ("colleges", "en-in", "abbreviation"),
    "govt": ("government", "en-in", "abbreviation"),
    "univ": ("university", "en-in", "abbreviation"),
    "sem": ("semester", "en-in", "abbreviation"),
    "prof": ("professor", "en-in", "abbreviation"),
    "profs": ("professors", "en-in", "abbreviation"),
}

_ELONGATED_CHAR_RE = re.compile(r"([a-zA-Z])\1{2,}")


def normalize_romanized_indic(text: str) -> Dict[str, Any]:
    """Non-destructively normalize Romanized Hindi/Bengali/Odia spelling variants.

    Strictly preserves ``original_text`` and returns full ``transformations`` metadata.
    """
    original = str(text or "")
    if not original.strip():
        return {
            "original_text": original,
            "normalized_text": original,
            "changed": False,
            "transformations": [],
        }

    transformations: List[Dict[str, Any]] = []
    parts: List[str] = []
    pos = 0

    for m in _ALPHA_TOK_RE.finditer(original):
        start, end = m.span()
        parts.append(original[pos:start])
        tok = m.group(0)
        low = tok.lower()
        collapsed = _ELONGATED_CHAR_RE.sub(r"\1\1", low)

        rule = ROMANIZED_VARIANT_RULES.get(low) or ROMANIZED_VARIANT_RULES.get(collapsed)
        if rule is not None:
            canon, lang, cat = rule
            parts.append(canon)
            transformations.append({
                "original": tok,
                "normalized": canon,
                "language": lang,
                "category": cat,
                "span": [start, end],
            })
        elif collapsed != low and len(low) >= 4:
            # Collapse 3+ repeated letters (e.g., "soooo" -> "soo", "bhaiiii" -> "bhaii")
            # Check if reducing to single char hits a known Hinglish/Banglish/Odia token
            single_collapsed = _ELONGATED_CHAR_RE.sub(r"\1", low)
            target = (
                single_collapsed
                if (single_collapsed in STRONG_HINGLISH or single_collapsed in STRONG_BANGLISH or single_collapsed in STRONG_ROMAN_ODIA or single_collapsed in ENGLISH_OR_MBA_ANCHORS)
                else collapsed
            )
            parts.append(target)
            transformations.append({
                "original": tok,
                "normalized": target,
                "language": "multilingual",
                "category": "elongated_spelling",
                "span": [start, end],
            })
        else:
            parts.append(tok)
        pos = end

    parts.append(original[pos:])
    normalized = "".join(parts)
    return {
        "original_text": original,
        "normalized_text": normalized,
        "changed": normalized != original,
        "transformations": transformations,
    }


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
        return {"primary_script": "unknown", "dominant_indic_script": None, "scripts_present": [], "char_counts": {}}

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


def _classify_segment(raw: str, script_detector: Any = None) -> Dict[str, Any]:
    """Core language/script/confidence classifier for one span or document."""
    script_profile = detect_script_profile(raw)
    primary_script = script_profile["primary_script"]
    dominant_indic = script_profile.get("dominant_indic_script")

    if primary_script == "unknown" and script_detector is not None:
        s_res = script_detector.detect(raw)
        primary_script = str(getattr(s_res, "script", "unknown") or "unknown").lower()

    latin_toks = [t.lower() for t in _ALPHA_TOK_RE.findall(raw)]
    n_tok = len(latin_toks)

    # 1. Native Indic scripts (Devanagari, Bengali, Odia, Tamil, Telugu, etc.)
    if dominant_indic and dominant_indic in _SCRIPT_TO_LANG:
        indic_lang = _SCRIPT_TO_LANG[dominant_indic]
        indic_chars = script_profile["char_counts"].get(dominant_indic, 0)
        conf = round(min(0.99, 0.85 + 0.015 * min(indic_chars, 10)), 4)
        if n_tok >= 1 and primary_script == "mixed":
            return {
                "language": f"{indic_lang}-en",
                "script": "mixed",
                "is_code_mixed": True,
                "detected_languages": [indic_lang, "en"],
                "indic_script": dominant_indic,
                "matched_markers": [],
                "confidence": conf,
                "abstain": False,
                "uncertainty_flag": "mixed_span",
            }
        return {
            "language": indic_lang,
            "script": dominant_indic,
            "is_code_mixed": False,
            "detected_languages": [indic_lang],
            "indic_script": dominant_indic,
            "matched_markers": [],
            "confidence": conf,
            "abstain": False,
            "uncertainty_flag": "clear",
        }

    # 2. Romanized Indic code-mixing (Hinglish, Banglish, Roman-Odia)
    # Guard proper noun "Bodh Gaya" (IIM Bodh Gaya) from colliding with Hindi verb "gaya"
    hi_hits = [
        t for idx, t in enumerate(latin_toks)
        if t in STRONG_HINGLISH and not (t == "gaya" and idx > 0 and latin_toks[idx - 1] == "bodh")
    ]
    bn_hits = [t for t in latin_toks if t in STRONG_BANGLISH]
    or_hits = [t for t in latin_toks if t in STRONG_ROMAN_ODIA]
    sec_hits = [t for t in latin_toks if t in _SECONDARY_BN_OR]
    en_hits = [t for t in latin_toks if t in ENGLISH_OR_MBA_ANCHORS]

    # Guard: "ra" in Roman Odia only counts when paired with another Odia marker
    or_core = [t for t in or_hits if t != "ra"]
    or_effective = or_hits if or_core else []

    bn_score = len(bn_hits) + (0.5 * len(sec_hits) if bn_hits else 0.0)
    or_score = len(or_effective) + (0.5 * len(sec_hits) if or_effective else 0.0)
    hi_score = float(len(hi_hits))

    best_lang = None
    best_hits: List[str] = []
    if bn_hits and bn_score >= max(hi_score, or_score) and (
        len(bn_hits) >= 2 or (len(bn_hits) >= 1 and len(sec_hits) >= 1) or (n_tok <= 16 and len(bn_hits) >= 1)
    ):
        best_lang = "bn"
        best_hits = sorted(set(bn_hits))
    elif or_effective and or_score >= max(hi_score, bn_score) and (
        len(or_core) >= 2 or (len(or_core) >= 1 and len(sec_hits) >= 1) or (n_tok <= 16 and len(or_core) >= 1)
    ):
        best_lang = "or"
        best_hits = sorted(set(or_effective))
    elif hi_hits and (len(hi_hits) >= 2 or (n_tok <= 16 and len(hi_hits) >= 1)):
        best_lang = "hi"
        best_hits = sorted(set(hi_hits))

    if best_lang is not None:
        raw_hits_count = len(bn_hits) if best_lang == "bn" else (len(or_effective) if best_lang == "or" else len(hi_hits))
        indic_ratio = raw_hits_count / max(n_tok, 1)
        conf = round(min(0.96, 0.62 + 0.08 * min(raw_hits_count, 4) + 0.12 * min(indic_ratio, 1.0)), 4)
        # Pure Romanized Indic requires zero substantive English anchor words
        if indic_ratio >= 0.50 and len(en_hits) == 0:
            return {
                "language": best_lang,
                "script": "latin",
                "is_code_mixed": False,
                "detected_languages": [best_lang],
                "indic_script": None,
                "matched_markers": best_hits,
                "confidence": conf,
                "abstain": False,
                "uncertainty_flag": "clear",
            }
        return {
            "language": f"{best_lang}-en",
            "script": "latin",
            "is_code_mixed": True,
            "detected_languages": [best_lang, "en"],
            "indic_script": None,
            "matched_markers": best_hits,
            "confidence": conf,
            "abstain": False,
            "uncertainty_flag": "mixed_span",
        }

    # 3. English / Indian MBA-Reddit English vs. short undetermined replies
    if en_hits or n_tok >= 4:
        en_ratio = len(en_hits) / max(n_tok, 1)
        conf = round(min(0.97, 0.55 + 0.25 * min(en_ratio * 2.0, 1.0) + 0.03 * min(n_tok, 6)), 4)
        abstain = (len(en_hits) == 0 and n_tok <= 4)
        return {
            "language": "en",
            "script": "latin",
            "is_code_mixed": False,
            "detected_languages": ["en"],
            "indic_script": None,
            "matched_markers": [],
            "confidence": conf,
            "abstain": abstain,
            "uncertainty_flag": "ambiguous_short" if abstain else "clear",
        }

    # Explicit abstention for numeric-only, punctuation-only, or ultra-terse unknown strings
    return {
        "language": "und",
        "script": primary_script if primary_script else "unknown",
        "is_code_mixed": False,
        "detected_languages": [],
        "indic_script": None,
        "matched_markers": [],
        "confidence": 0.25 if n_tok > 0 else 0.0,
        "abstain": True,
        "uncertainty_flag": "low_evidence_abstain",
    }


def detect_language_spans(text: str, script_detector: Any = None) -> List[Dict[str, Any]]:
    """Segment text into sentence/clause spans and identify language, script & uncertainty per span."""
    raw = str(text or "")
    if not raw.strip():
        return []

    spans: List[Dict[str, Any]] = []
    cursor = 0
    for m in _SPAN_SPLIT_RE.finditer(raw):
        seg = raw[cursor : m.start()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            trail = len(seg) - len(seg.rstrip())
            s_start = cursor + lead
            s_end = m.start() - trail
            seg_clean = raw[s_start:s_end]
            res = _classify_segment(seg_clean, script_detector=script_detector)
            spans.append({
                "text": seg_clean,
                "start": s_start,
                "end": s_end,
                **res,
            })
        cursor = m.end()

    tail = raw[cursor:]
    if tail.strip():
        lead = len(tail) - len(tail.lstrip())
        trail = len(tail) - len(tail.rstrip())
        s_start = cursor + lead
        s_end = len(raw) - trail
        seg_clean = raw[s_start:s_end]
        res = _classify_segment(seg_clean, script_detector=script_detector)
        spans.append({
            "text": seg_clean,
            "start": s_start,
            "end": s_end,
            **res,
        })
    return spans


def detect_indian_reddit_language(text: str, script_detector: Any = None) -> Dict[str, Any]:
    """Identify document-level AND span-level language, script, code-mixing, and uncertainty.

    Backward-compatible with all V5/V6.1 callers while exposing V6 span-level
    language detection (`spans`), `confidence`, `abstain`, and `uncertainty_flag`.
    """
    raw = str(text or "")
    doc_res = _classify_segment(raw, script_detector=script_detector)
    spans = detect_language_spans(raw, script_detector=script_detector)

    # Reconcile multi-span code-switching: if separate sentences use distinct languages
    # (e.g., Sentence 1 in English, Sentence 2 in Hindi/Bengali/Odia), never force a single label!
    span_langs: List[str] = []
    for sp in spans:
        for lg in sp.get("detected_languages", []):
            if lg and lg not in span_langs:
                span_langs.append(lg)

    if len(span_langs) >= 2 and not doc_res["is_code_mixed"]:
        indic = next((lg for lg in span_langs if lg != "en"), None)
        if indic and "en" in span_langs:
            doc_res["language"] = f"{indic}-en"
            doc_res["is_code_mixed"] = True
            doc_res["detected_languages"] = [indic, "en"]
            doc_res["uncertainty_flag"] = "mixed_span"
            doc_res["abstain"] = False
        elif len(span_langs) >= 2:
            doc_res["is_code_mixed"] = True
            doc_res["detected_languages"] = span_langs
            doc_res["uncertainty_flag"] = "mixed_span"

    doc_res["spans"] = spans
    doc_res["span_count"] = len(spans)
    doc_res["has_multi_language_spans"] = len({sp["language"] for sp in spans if sp["language"] != "und"}) >= 2
    return doc_res
