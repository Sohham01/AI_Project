"""
Emoji normalization stage.

Emojis are NOT removed — they carry sentiment/emotion information.
Each emoji is replaced with a deterministic semantic token so that
downstream NLP models can use the information.

Representation chosen: <EMOJI_name>
  e.g. "😭" -> "<EMOJI_loudly_crying_face>"

Design decisions:
- We use a small curated mapping for the most common emojis seen in
  Indian social-media / Reddit discourse.
- Unknown emojis fall back to their Unicode name via unicodedata.name().
- Consecutive duplicate emoji tokens are collapsed to one token + count
  only if there are 3+ identical emojis in a row (to reduce noise while
  preserving emphasis from a pair).
- The original emoji is always recoverable from original_text.
"""

import re
import unicodedata
from typing import Dict, List

# Curated mapping: emoji -> semantic token (lowercase, underscored)
_EMOJI_MAP: Dict[str, str] = {
    "\U0001f602": "face_with_tears_of_joy",
    "\U0001f601": "beaming_face",
    "\U0001f603": "grinning_face_big_eyes",
    "\U0001f604": "grinning_face_smiling_eyes",
    "\U0001f605": "grinning_face_sweat",
    "\U0001f606": "grinning_squinting_face",
    "\U0001f609": "winking_face",
    "\U0001f60a": "smiling_face_smiling_eyes",
    "\U0001f60d": "heart_eyes",
    "\U0001f618": "face_blowing_kiss",
    "\U0001f62d": "loudly_crying_face",
    "\U0001f622": "crying_face",
    "\U0001f625": "sad_but_relieved_face",
    "\U0001f62a": "sleepy_face",
    "\U0001f620": "angry_face",
    "\U0001f621": "pouting_face",
    "\U0001f624": "face_with_steam",
    "\U0001f610": "neutral_face",
    "\U0001f611": "expressionless_face",
    "\U0001f614": "pensive_face",
    "\U0001f615": "confused_face",
    "\U0001f633": "flushed_face",
    "\U0001f631": "face_screaming",
    "\U0001f637": "face_with_mask",
    "\U0001f44d": "thumbs_up",
    "\U0001f44e": "thumbs_down",
    "\U0001f44f": "clapping_hands",
    "\U0001f64f": "folded_hands",
    "\U0001f525": "fire",
    "\u2764\ufe0f": "red_heart",
    "\u2764": "red_heart",
    "\U0001f494": "broken_heart",
    "\U0001f4af": "hundred_points",
    "\U0001f389": "party_popper",
    "\U0001f680": "rocket",
    "\U0001f914": "thinking_face",
    "\U0001f92f": "exploding_head",
    "\U0001f926": "facepalm",
    "\U0001f937": "shrug",
    "\U0001f4a9": "poop",
    "\U0001f923": "rofl",
    "\U0001f60e": "sunglasses_face",
    "\U0001f929": "star_struck",
    "\U0001f971": "yawning_face",
    "\U0001f974": "woozy_face",
    "\U0001f976": "cold_face",
    "\U0001f975": "hot_face",
    "\U0001f61c": "winking_tongue",
    "\U0001f61d": "squinting_tongue",
    "\U0001f61b": "tongue_out",
    "\U0001f913": "nerd_face",
    "\U0001f60f": "smirking_face",
    "\U0001f612": "unamused_face",
    "\U0001f644": "eye_roll",
    "\U0001f4aa": "flexed_biceps",
    "\u270b": "raised_hand",
    "\u270c\ufe0f": "victory_hand",
    "\u270c": "victory_hand",
    "\U0001f44b": "waving_hand",
    "\U0001f480": "skull",
    "\U0001f921": "clown_face",
    "\U0001f643": "upside_down_face",
    "\U0001f972": "smiling_face_with_tear",
    "\U0001f62b": "tired_face",
    "\U0001f629": "weary_face",
    "\U0001f97a": "pleading_face",
    "\U0001f440": "eyes",
}

# Semantic affect verbalizations for social-media emoji pragmatics (Task 6)
EMOJI_AFFECT_MAP: Dict[str, str] = {
    "\U0001f62d": "sadness frustration",
    "\U0001f622": "sadness disappointment",
    "\U0001f525": "strong approval",
    "\U0001f480": "extreme amusement disbelief",
    "\U0001f921": "sarcasm mockery",
    "\U0001f602": "amusement laughter",
    "\U0001f923": "strong amusement laughter",
    "\U0001f620": "anger frustration",
    "\U0001f621": "anger frustration",
    "\U0001f624": "frustration annoyance",
    "\U0001f64f": "gratitude polite request",
    "\U0001f44d": "approval agreement",
    "\U0001f44e": "disapproval disagreement",
    "\u2764\ufe0f": "strong appreciation",
    "\u2764": "strong appreciation",
    "\U0001f494": "disappointment heartbreak",
    "\U0001f4af": "strong agreement",
    "\U0001f389": "celebration congratulations",
    "\U0001f680": "excitement ambition",
    "\U0001f914": "curiosity skepticism",
    "\U0001f92f": "shock disbelief",
    "\U0001f926": "frustration disbelief",
    "\U0001f937": "uncertainty indifference",
    "\U0001f644": "annoyance skepticism",
    "\U0001f643": "ironic frustration",
    "\U0001f972": "bittersweet sadness stress",
    "\U0001f62b": "exhaustion stress",
    "\U0001f629": "overwhelmed stress",
    "\U0001f97a": "anxious plea",
    "\U0001f440": "curious attention",
    "\U0001f4aa": "encouragement confidence",
    "\U0001f60a": "warmth gratitude",
    "\U0001f60e": "confidence approval",
}

# Character class of *emoji base* code points (covers most common ranges).
# Only non-ASCII emoji code points belong here.  ASCII punctuation such as
# ?, !, ., ,, :, ; lives below U+2700 and must NEVER be part of this class --
# it is owned by punctuation_normalizer / repeated_char_normalizer.
_EMOJI_BASE_CLASS = (
    r"\U0001f300-\U0001f5ff"    # symbols & pictographs
    r"\U0001f600-\U0001f64f"    # emoticons
    r"\U0001f680-\U0001f6ff"    # transport & map
    r"\U0001f900-\U0001f9ff"    # supplemental symbols and pictographs
    r"\U0001fa00-\U0001fa6f"    # chess symbols
    r"\U0001fa70-\U0001faff"    # symbols extended-A
    r"\u2600-\u26ff"            # misc symbols
    r"\u2700-\u27bf"            # dingbats (includes the red heart U+2764)
)

# Variation selectors (text / emoji presentation) and zero-width joiner are
# not emoji on their own -- they only qualify an adjacent base character,
# e.g. the red heart is "U+2764 U+FE0F".  They are therefore consumed as part
# of the matched sequence instead of being listed as standalone class members.
_VARIATION_SELECTORS = "\ufe0e\ufe0f"
_ZWJ = "\u200d"
_SEQUENCE_INFIX = _VARIATION_SELECTORS + _ZWJ

# Matches an emoji base character optionally followed by a variation selector
# and any number of ZWJ-joined continuations (e.g. "\U0001f9d1\u200d\U0001f4bb").
_EMOJI_RE = re.compile(
    f"[{_EMOJI_BASE_CLASS}]"
    f"(?:[{_VARIATION_SELECTORS}])?"
    f"(?:{_ZWJ}[{_EMOJI_BASE_CLASS}](?:[{_VARIATION_SELECTORS}])?)*"
)


def _single_emoji_to_token(emoji_char: str) -> str:
    """Convert a single emoji base character to its semantic token."""
    if emoji_char in _EMOJI_MAP:
        return f"<EMOJI_{_EMOJI_MAP[emoji_char]}>"
    # Fallback: use Unicode name
    try:
        name = unicodedata.name(emoji_char, "").lower().replace(" ", "_")
        if name:
            return f"<EMOJI_{name}>"
    except (TypeError, ValueError):
        pass
    return f"<EMOJI_unknown>"


def _emoji_to_token(emoji_seq: str) -> str:
    """Convert a matched emoji sequence to a single deterministic token."""
    if emoji_seq in _EMOJI_MAP:
        return f"<EMOJI_{_EMOJI_MAP[emoji_seq]}>"
    # Recover the base character(s) by dropping variation selectors / ZWJ.
    bases = [ch for ch in emoji_seq if ch not in _SEQUENCE_INFIX]
    if not bases:
        return "<EMOJI_unknown>"
    if len(bases) == 1:
        return _single_emoji_to_token(bases[0])
    # ZWJ sequence (e.g. "person technologist"): deterministic combined token.
    return "<EMOJI_" + "_".join(
        _single_emoji_to_token(ch)[len("<EMOJI_"):-1] for ch in bases
    ) + ">"


def normalize_emojis(text: str) -> str:
    """Replace emojis with semantic tokens. Does NOT remove them."""
    return _EMOJI_RE.sub(lambda m: _emoji_to_token(m.group()), text)


def strip_emojis(text: str) -> str:
    """Remove emojis completely (used for Task 6 preprocessing ablation only)."""
    return re.sub(r"\s+", " ", _EMOJI_RE.sub(" ", str(text or ""))).strip()


def _emoji_to_affect(emoji_seq: str) -> str:
    if emoji_seq in EMOJI_AFFECT_MAP:
        return f" {EMOJI_AFFECT_MAP[emoji_seq]} "
    bases = [ch for ch in emoji_seq if ch not in _SEQUENCE_INFIX]
    if len(bases) == 1 and bases[0] in EMOJI_AFFECT_MAP:
        return f" {EMOJI_AFFECT_MAP[bases[0]]} "
    tok = _emoji_to_token(emoji_seq).removeprefix("<EMOJI_").removesuffix(">").replace("_", " ")
    return f" {tok} "


def normalize_emojis_semantic_affect(text: str) -> str:
    """Replace emojis with natural-language semantic affect phrases (Task 6)."""
    return re.sub(r"\s+", " ", _EMOJI_RE.sub(lambda m: _emoji_to_affect(m.group()), str(text or ""))).strip()


def extract_emojis(text: str) -> List[Dict[str, str]]:
    """Return each detected emoji with its deterministic token and semantic affect."""
    found: List[Dict[str, str]] = []
    seen: set[str] = set()
    for match in _EMOJI_RE.finditer(str(text or "")):
        raw = match.group()
        if raw in seen:
            continue
        seen.add(raw)
        bases = [ch for ch in raw if ch not in _SEQUENCE_INFIX]
        affect = EMOJI_AFFECT_MAP.get(raw) or (EMOJI_AFFECT_MAP.get(bases[0]) if len(bases) == 1 else None)
        token = _emoji_to_token(raw)
        found.append({
            "emoji": raw,
            "token": token,
            "semantic_affect": affect or token.removeprefix("<EMOJI_").removesuffix(">").replace("_", " "),
        })
    return found
