
import re

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


class TextUtilsMixin:
    """String/regex helpers: IFSC matching, garbage/devanagari detection, repetition."""

    def _normalize_alpha(self, text: str) -> str:
        return re.sub(r'[^a-z]', '', text.lower())

    def _strip_devanagari(self, text: str) -> str:
        if not text:
            return text
        tokens = text.split()
        kept = [t for t in tokens if not DEVANAGARI_RE.search(t)]
        return ' '.join(kept).strip()

    def _is_mostly_devanagari(self, text: str, ratio=0.3) -> bool:
        if not text:
            return False
        hits = len(DEVANAGARI_RE.findall(text))
        return hits > len(text) * ratio

    def _is_probable_garbage(self, text: str, confidence: float = 1.0, conf_thresh: float = 0.55) -> bool:
        t = text.strip()
        if not t:
            return True
        if DEVANAGARI_RE.search(t):
            return False
        if re.search(r'[^\x00-\x7F]', t):
            return True
        letters = [c for c in t if c.isalpha()]
        if len(letters) >= 4:
            vowel_ratio = sum(1 for c in letters if c.lower() in 'aeiou') / len(letters)
            if vowel_ratio < 0.12 and confidence < conf_thresh:
                return True
        return False

    def _is_repetitive(self, text: str, min_repeats=3) -> bool:
        tokens = [t.lower() for t in re.findall(r'\b\w+\b', text)]
        if len(tokens) < min_repeats * 2:
            return False
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return max(counts.values()) >= min_repeats

    def _ifsc_candidates(self, raw: str):
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
        seen = {cleaned}
        yield cleaned
        for i, ch in enumerate(cleaned):
            for a, b in self.confusion_subs:
                if ch == a:
                    variant = cleaned[:i] + b + cleaned[i + 1:]
                    if variant not in seen:
                        seen.add(variant)
                        yield variant

    def _has_foreign_long_digit_run(self, text: str, code: str, min_run: int = 9) -> bool:
        """True if `text` contains a digit run of min_run+ characters that
        isn't part of the matched IFSC code itself. A real IFSC line never
        contains a run this long (its own digits are at most 6-7 chars);
        account numbers are 9-18 digits. This catches the case where OCR
        merges an 'A/c No.' label with the adjacent account-number digits
        into one line/box, which can coincidentally match the IFSC regex
        (4 letters + 0/O + alnum) purely from misread label characters."""
        for run in re.findall(r'\d+', text):
            if len(run) >= min_run and run not in code:
                return True
        return False

    def try_match_ifsc(self, raw: str):
        if not raw:
            return None

        search_pattern = re.compile(r'[A-Z]{4}[0O][A-Z0-9]{5,6}')

        for candidate in self._ifsc_candidates(raw):
            cleaned_stripped = re.sub(r'(RTGS|NEFT|IFSC|IFS|C0DE|CODE)', '', candidate)
            for target in (cleaned_stripped, candidate):
                match = search_pattern.search(target)
                if not match:
                    continue
                found = match.group(0)
                ifsc_code = found[:4] + '0' + found[5:]
                tail = ifsc_code[5:]
                if sum(c.isdigit() for c in tail) < 4:
                    continue
                return ifsc_code

        return None