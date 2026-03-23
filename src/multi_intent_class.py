import sys

sys.path.append("")

import re
import spacy
from src.utils import config_parser

class MultiIntentParser:

    def __init__(self, config_path: str):
        """
        Initialize parser with config.
        Loads spaCy models once and stores config globally.
        """
        self.config = config_parser(config_path)['multi_intent']

        # Flat list of all verbs across languages — built once from CONFIG
        self.all_verbs: list[str] = (
            self.config["verbs_en"]
            + self.config["verbs_vi"]
            + self.config["verbs_ko"]
            + self.config["verbs_jp"]
        )

        # load spaCy models once
        self._load_spacy_models()

    # -------------------------------------------------
    # spaCy loader
    # -------------------------------------------------
    def _load_spacy_models(self):

        self.spacy_nlps = {}


        for lang, model_name in self.config["spacy_models"].items():

            if model_name is None:
                self.spacy_nlps[lang] = None
                continue

            try:
                self.spacy_nlps[lang] = spacy.load(model_name)
            except OSError:
                print(f"[WARN] spaCy model not found for {lang}: {model_name}")
                self.spacy_nlps[lang] = None

    def spacy_has_verb(self, text: str, lang: str) -> bool:
        """
        Return True if *text* contains at least one VERB or AUX token according
        to the spaCy POS tagger for *lang*.

        Falls back to has_verb() (verb-list check) when:
        - The spaCy model for *lang* was not loaded (OSError at load time)
        - *lang* is "vi" (no official model)
        - *lang* is not in self.spacy_nlps at all

        Universal POS tags used:
        - VERB  : main verbs  (열다, 開ける, open, mở)
        - AUX   : auxiliaries (있다, する, will, đã)
        """
        nlp = self.spacy_nlps.get(lang)
        if nlp is None:
            # Graceful fallback — use the original verb-list method
            return self.has_verb(text, self.all_verbs)
        doc = nlp(text)
        return any(token.pos_ in ("VERB", "AUX") for token in doc)
    
    def parse (self, text: str) -> list[str]:
        
        lang  = self.detect_lang(text)
        parts = self.split_cjk_aware(text)
    
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
            is_short          = len(fragment.split()) < self.config["min_words_merge"]
            fragment_has_verb = self.spacy_has_verb(fragment, lang)
            if is_short and not fragment_has_verb:
                result[-1] += " " + fragment
            else:
                result.append(fragment)
        return result
    

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
        # Unicode block ranges used for language detection
        _RE_HANGUL  = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")
        _RE_HIRAGANA = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")

        if _RE_HANGUL.search(text):
            return "ko"
        if _RE_HIRAGANA.search(text):
            return "jp"
        return "other"

    

    def split_basic(self, text: str) -> list[str]:
        """
        Step 1 shared by all methods.
        Split *text* on the regex pattern defined in self.config["split_pattern"].
        Returns a list of non-empty stripped fragments.
        """
        return [
            p.strip()
            for p in re.split(self.config["split_pattern"], text)
            if p.strip()
        ]

    def split_cjk_aware(self, text: str) -> list[str]:
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
            primary = self.config["split_pattern_ko"]
            protected = text
        elif lang == "jp":
            primary = self.config["split_pattern_jp"]
            # Protect postpositions ending in て before applying て-split
            protected = self.protect_jp_postpositions(text)
        else:
            return self.split_basic(text)

        # Primary CJK split
        rough_parts = [p.strip() for p in re.split(primary, protected) if p.strip()]

        # Restore any protected て characters
        if lang == "jp":
            rough_parts = [self.restore_jp_postpositions(p) for p in rough_parts]

        # Secondary standard split on each rough fragment (catches 그리고, そして …)
        final = []
        for part in rough_parts:
            sub = self.split_basic(part)
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


    def protect_jp_postpositions(self, text: str) -> str:
        """
        Replace the final て of each known Japanese postposition with a
        private-use placeholder (\\uE000) so the て-split regex skips it.

        Example:  'マイケルジャクソンについて教えて'
                → 'マイケルジャクソンについ\\uE000教えて'
        The placeholder is restored after splitting via restore_jp_postpositions().
        """
        for pp in self.config["jp_postpositions"]:
            # pp ends in て; replace only the final て with the placeholder
            text = text.replace(pp, pp[:-1] + "\uE000")
        return text

    def restore_jp_postpositions(self, text: str) -> str:
        """Restore the て placeholder inserted by protect_jp_postpositions."""
        return text.replace("\uE000", "て")

    
if __name__ == "__main__":

    parser = MultiIntentParser(config_path="config/routing_config.yaml")

    texts = [
        # English (3)
        "open door and close window",
        "turn on light, play music and tell me joke",
        "start engine, open trunk and check fuel",

        # Korean (3)
        "문 열고 창문 닫고 음악 틀어",
        "엔진 켜고 트렁크 열고 농담 말해",
        "문 잠그고 라디오 켜고 에어컨 켜",

        # Japanese (3)
        "ドアを開けて窓を閉めて音楽をかけて",
        "エンジンをかけてトランクを開けてジョークを言って",
        "ドアをロックしてライトをつけてラジオをつけて",

        # Vietnamese (1)
        "mở cửa, bật đèn và phát nhạc",
    ]

    for text in texts:
        parts = parser.parse(text)
        print("TEXT:", text)
        print("PARSED:", parts)
        print("-" * 40)