import re

# Unicode block ranges used for language detection
_RE_HANGUL  = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")
_RE_HIRAGANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")

CONFIG = {
    "split_pattern": (
        r",|;"
        r"| and "
        r"| then "
        r"| also "
        r"| and then "
        # Vietnamese
        r"| và "
        r"| rồi "
        r"| sau đó "
        # Korean
        r"| 그리고 "
        r"| 그리고 나서 "
        r"| 그 다음 "
        # Japanese
        r"| そして "
        r"| それから "
        r"| そのあと "
    ),

    # ── Pronouns (trigger coreference resolution) ─────────────────────────
    "pronouns": [
        # English
        "him", "her", "them", "it", "that", "this", "his", "their",
        # Vietnamese
        "anh ấy", "cô ấy", "ông ấy", "bà ấy", "họ", "nó", "đó",
        # Korean
        "그", "그녀", "그들", "그것",
        # Japanese
        "彼", "彼女", "彼ら", "それ",
    ],

    # ── Intent-bearing verbs (used to decide split vs. merge) ─────────────
    "verbs_en": [
        "open", "close", "play", "stop", "pause",
        "tell", "show", "find", "navigate",
        "turn", "switch", "enable", "disable",
        "unlock", "lock", "check", "search",
        "call", "send", "read",
        "start", "launch",
        "increase", "decrease",
        "set", "change",
    ],
    "verbs_vi": [
        "mở", "đóng", "bật", "tắt",
        "phát", "dừng",
        "kể", "nói",
        "tìm", "kiểm tra",
        "dẫn", "chỉ",
        "gọi", "nhắn",
        "tăng", "giảm",
    ],
    "verbs_ko": [
        "열", "닫", "켜", "꺼",
        "재생", "멈춰",
        "말", "알려",
        "찾", "검색",
        "안내",
        "전화",
    ],
    "verbs_jp": [
        "開け", "閉め",
        "再生", "停止",
        "教え",
        "探",
        "案내",
        "電話",
    ],

    # ── CJK-aware split patterns (method 5) ──────────────────────────────
    # Korean: split on verb-connective suffix 고 (e.g. 열고 → 열 | 고)
    # The pattern keeps the verb stem by splitting AFTER the 고/서 ending.
    # We insert a split boundary after verb-stem+고 sequences.
    # Regex: look for a Hangul character followed by 고/서 then space or end.
    "split_pattern_ko": (
        r"(?<=고)\s+"          # after 고 + whitespace  (열고 음악)
        r"|(?<=서)\s+"         # after 서 + whitespace  (해서 ...)
        r"|(?<=고)(?=[가-힣])" # after 고 directly before next Hangul (no space)
    ),

    # Japanese: split on て-form connectives and Japanese comma
    # 開けて音楽  →  開けて | 音楽
    "split_pattern_jp": (
        r"(?<=て)(?=[ぁ-ん\u30A0-\u30FF\u4E00-\u9FFF])"  # after て before CJK/kana
        r"|(?<=して)(?=[ぁ-ん\u30A0-\u30FF\u4E00-\u9FFF])"
        r"|[、，]"             # Japanese / fullwidth comma
    ),

    # ── Japanese postpositions ending in て (method 7) ───────────────────
    # These are grammatical fixed forms, NOT verb connective て.
    # Method 7 protects them before splitting so they don't create
    # spurious fragment boundaries.
    # Examples:  について (about)  にとって (for)  によって (by/through)
    "jp_postpositions": [
        "について",   # about
        "にとって",   # for / to (someone)
        "によって",   # by / through / depending on
        "に対して",   # against / toward
        "に関して",   # regarding
        "をもって",   # with / by means of
        "において",   # in / at (formal)
    ],

    # ── spaCy models (method 6) ───────────────────────────────────────────
    # Official models for en/ko/jp; vi falls back to verb-list (no official model).
    # Install before running method 6:
    #   pip install spacy
    #   python -m spacy download en_core_web_sm
    #   python -m spacy download ko_core_news_sm
    #   pip install spacy[ja] && python -m spacy download ja_core_news_sm
    "spacy_models": {
        "en": "en_core_web_sm",
        "ko": "ko_core_news_sm",
        "jp": "ja_core_news_sm",
        "vi": None,             # no official model → falls back to has_verb()
    },

    # ── Semantic thresholds ───────────────────────────────────────────────
    # Cosine similarity above which two consecutive fragments are merged
    "similarity_threshold": 0.8,

    # Fragments with fewer words than this are merged into the previous one
    "min_words_merge": 5,
}

# Flat list of all verbs across languages — built once from CONFIG
ALL_VERBS: list[str] = (
    CONFIG["verbs_en"]
    + CONFIG["verbs_vi"]
    + CONFIG["verbs_ko"]
    + CONFIG["verbs_jp"]
)

class MultiIntentParser:

    def __init__(self, config: dict):
        """
        Initialize parser with config.
        Loads spaCy models once and stores config globally.
        """
        self.cfg = config

        # expose config globally for existing functions
        global CONFIG
        CONFIG = config

        # load spaCy models once
        self._load_spacy_models()


    def detect_lang(self, text: str) -> str:
        """
        Heuristic language detection based on Unicode character blocks.
        Returns one of: 'ko', 'jp', 'other'.

        Korean (Hangul) and Japanese (Hiragana / Katakana) are the two scripts
        that need CJK-aware splitting; everything else falls through to the
        standard regex split.

        Detection is intentionally lightweight — a single character is enough
        to identify the script.
        """
        if _RE_HANGUL.search(text):
            return "ko"
        if _RE_HIRAGANA.search(text):
            return "jp"
        return "other"

    # -------------------------------------------------
    # spaCy loader
    # -------------------------------------------------
    def _load_spacy_models(self):

        global SPACY_NLPS
        SPACY_NLPS = {}

        import spacy

        for lang, model_name in self.cfg["spacy_models"].items():

            if model_name is None:
                SPACY_NLPS[lang] = None
                continue

            try:
                SPACY_NLPS[lang] = spacy.load(model_name)
            except OSError:
                print(f"[WARN] spaCy model not found for {lang}: {model_name}")
                SPACY_NLPS[lang] = None

    def split_basic(self, text: str, cfg: dict) -> list[str]:
        """
        Step 1 shared by all methods.
        Split *text* on the regex pattern defined in cfg["split_pattern"].
        Returns a list of non-empty stripped fragments.
        """
        return [
            p.strip()
            for p in re.split(cfg["split_pattern"], text)
            if p.strip()
        ]

    def split_cjk_aware(self, text: str, cfg: dict) -> list[str]:
        """
        Language-aware tokenisation for method 5.

        For Korean and Japanese the standard split_basic often returns a single
        fragment because those languages use verb-connective suffixes (고, て)
        rather than standalone conjunction words.

        Algorithm
        ---------
        1. Detect script (Korean / Japanese / other).
        2a. Korean  : split on cfg["split_pattern_ko"]  (고/서 verb endings)
                    then also split on the standard pattern (handles 그리고 etc.)
        2b. Japanese: split on cfg["split_pattern_jp"]  (て-form + 、)
                    then also split on the standard pattern (そして etc.)
        2c. Other   : fall back to split_basic (unchanged behaviour).
        3. Strip and drop empty fragments.

        Note: CJK splits are applied *first*, then the standard pattern is applied
        to each resulting fragment, so both levels of splitting cooperate.
        """
        lang = self.detect_lang(text)

        if lang == "ko":
            primary = cfg["split_pattern_ko"]
            protected = text
        elif lang == "jp":
            primary = cfg["split_pattern_jp"]
            # Protect postpositions ending in て before applying て-split
            protected = self.protect_jp_postpositions(text, cfg)
        else:
            return self.split_basic(text, cfg)

        # Primary CJK split
        rough_parts = [p.strip() for p in re.split(primary, protected) if p.strip()]

        # Restore any protected て characters
        if lang == "jp":
            rough_parts = [self.restore_jp_postpositions(p) for p in rough_parts]

        # Secondary standard split on each rough fragment (catches 그리고, そして …)
        final = []
        for part in rough_parts:
            sub = self.split_basic(part, cfg)
            final.extend(sub)

        return [p for p in final if p]

    def has_verb(self, text: str, verbs: list[str]) -> bool:
        """
        Return True if *text* (lowercased) contains at least one entry
        from *verbs*.  Used to decide whether a fragment carries its own
        intent-bearing verb.
        """
        t = text.lower()
        return any(v in t for v in verbs)

    def spacy_has_verb(self, text: str, lang: str) -> bool:
        """
        Return True if *text* contains at least one VERB or AUX token according
        to the spaCy POS tagger for *lang*.

        Falls back to has_verb() (verb-list check) when:
        - The spaCy model for *lang* was not loaded (OSError at load time)
        - *lang* is "vi" (no official model)
        - *lang* is not in SPACY_NLPS at all

        Universal POS tags used:
        - VERB  : main verbs  (열다, 開ける, open, mở)
        - AUX   : auxiliaries (있다, する, will, đã)
        """
        nlp = SPACY_NLPS.get(lang)
        if nlp is None:
            # Graceful fallback — use the original verb-list method
            return self.has_verb(text, ALL_VERBS)
        doc = nlp(text)
        return any(token.pos_ in ("VERB", "AUX") for token in doc)

    def parse (self, text: str, cfg: dict) -> list[str]:
        
        lang  = self.detect_lang(text)
        parts = self.split_cjk_aware(text, cfg)
    
        if len(parts) <= 1:
            return parts
    
        # ── Korean: spaCy verb detection, min_words_merge fixed at 2 ───────────
        if lang == "ko":
            KO_MIN_WORDS = 2
            result = [parts[0]]
            for fragment in parts[1:]:
                is_short          = len(fragment.split()) < KO_MIN_WORDS
                fragment_has_verb = self.spacy_has_verb(fragment, lang)
                if is_short and not fragment_has_verb:
                    result[-1] += " " + fragment
                else:
                    result.append(fragment)
            # Forward-scan collapse for object-list sentences
            return result
    
        # ── Japanese / other CJK: spaCy verb detection + forward-scan collapse ─
        result = [parts[0]]
        for fragment in parts[1:]:
            is_short          = len(fragment.split()) < cfg["min_words_merge"]
            fragment_has_verb = self.spacy_has_verb(fragment, lang)
            if is_short and not fragment_has_verb:
                result[-1] += " " + fragment
            else:
                result.append(fragment)
        return result

    def protect_jp_postpositions(self, text: str, cfg: dict) -> str:
        """
        Replace the final て of each known Japanese postposition with a
        private-use placeholder (\\uE000) so the て-split regex skips it.

        Example:  'マイケルジャクソンについて教えて'
                → 'マイケルジャクソンについ\\uE000教えて'
        The placeholder is restored after splitting via restore_jp_postpositions().
        """
        for pp in cfg["jp_postpositions"]:
            # pp ends in て; replace only the final て with the placeholder
            text = text.replace(pp, pp[:-1] + "\uE000")
        return text

    def restore_jp_postpositions(self, text: str) -> str:
        """Restore the て placeholder inserted by protect_jp_postpositions."""
        return text.replace("\uE000", "て")

    

if name == main

    parser = MultiIntentParser(CONFIG)

    text = "open door, window and trunk and tell me joke"

    parts = parser.parse(text)

    print(parts)