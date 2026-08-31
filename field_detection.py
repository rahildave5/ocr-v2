import re


class FieldDetectionMixin:
    """Text classifiers: bank name, person name, address, annotation labels, overlay marks."""

    def _has_bank_evidence(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        if 'bank' in t:
            return True
        if self.bank_continuation_re.search(text):
            return True
        return any(token in t for token in self.known_bank_tokens)

    def _looks_like_person_name(self, text: str) -> bool:
        t = text.strip()
        if not t or self._has_bank_evidence(t):
            return False
        if self.person_initial_re.search(t):
            return True
        if self.person_title_re.match(t):
            return True
        words = t.split()
        if 2 <= len(words) <= 5:
            name_like = all(
                re.match(r'^[A-Z]\.?$', w) or re.match(r'^[A-Z][a-z]+(?:-[A-Z][a-z]+)?$', w)
                for w in words
            )
            corp_markers = ('ltd', 'limited', 'corp', 'corporation', 'bank', 'india', 'co')
            if name_like and not any(m in t.lower() for m in corp_markers):
                return True
        return False

    def _looks_like_bank_name(self, text: str) -> bool:
        if not text or len(text.strip()) < 3:
            return False
        if self._looks_like_person_name(text):
            return False
        return self._has_bank_evidence(text)

    def _is_annotation_label(self, text: str) -> bool:
        normalized = re.sub(r'[^a-z_]', '', text.lower())
        return normalized in self.annotation_labels

    def _is_overlay_mark(self, text: str) -> bool:
        text_l = text.lower()
        direct_keywords = [
            'cancel', 'cancelled', 'void', 'specimen', 'sample',
            'not negotiable', 'duplicate', 'draft', 'not valid',
        ]
        if any(kw in text_l for kw in direct_keywords):
            return True
        cleaned = self._normalize_alpha(text)
        if len(cleaned) < 5:
            return False
        if cleaned.startswith('canc') and cleaned.endswith(('led', 'lled', 'ed')):
            return True
        if 'camcel' in cleaned or 'cancell' in cleaned or 'cancl' in cleaned:
            return True
        return False

    def is_cancelled(self, extracted) -> bool:
        cancel_keywords = ('cancelled', 'cancel', 'void')
        for item in extracted:
            text = item.get("text", "")
            text_l = text.lower()
            if any(kw in text_l for kw in cancel_keywords):
                return True
            cleaned = self._normalize_alpha(text)
            if len(cleaned) < 5:
                continue
            if cleaned.startswith('canc') and cleaned.endswith(('led', 'lled', 'ed')):
                return True
            if 'camcel' in cleaned or 'cancell' in cleaned or 'cancl' in cleaned:
                return True
        return False

    def detect_annotated_upload(self, extracted) -> bool:
        hits = sum(
            1 for item in extracted
            if self._is_annotation_label(item.get("text", ""))
        )
        return hits >= 2

    def is_address(self, text: str) -> bool:
        text_lower = text.lower()
        if self._is_overlay_mark(text) or self._is_annotation_label(text):
            return False
        if self._is_repetitive(text) or any(kw in text_lower for kw in self.non_field_keywords):
            return False
        grid_keywords = ['valid for', 'months only', 'cash transaction', 'd d m m', 'y y y y']
        if any(kw in text_lower for kw in grid_keywords):
            return False
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        if digits > 0 and letters == 0:
            return False
        if re.search(r'\b\d{6}\b', text) or re.search(r'\b\d{3}\s\d{3}\b', text):
            return True
        keywords = [
            'branch', 'road', 'marg', 'street', 'nagar', 'dist', 'district', 'plot',
            'complex', 'floor', 'bldg', 'building', 'mumbai', 'delhi', 'bangalore',
            'bengaluru', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'sector',
            'opp', 'near', 'post', 'pin', 'pincode', 'tel', 'fax', 'hub', 'centre',
            'center', 'chowk', 'lane', 'estate', 'park', 'cross', 'extn', 'extension', 'main'
        ]
        if any(kw in text_lower for kw in keywords):
            return True
        if self.branch_code_re.search(text):
            return True
        if ',' in text and re.search(r'\d+', text):
            return True
        return False

    def _clean_bank_name(self, text: str) -> str:
        t = text
        t = re.sub(r'IFS?C?\s*CODE\s*[:\-]?\s*[A-Z]{4}0[A-Z0-9]{6}', ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', ' ', t)
        t = re.sub(r'SWIFT\s*:?', ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'\(\d{4,5}\)[^,]*', ' ', t)
        t = re.sub(r'\b\d{6}\b', ' ', t)
        address_kw_pattern = r'\b(branch|road|marg|street|nagar|dist|district|vpo|teh|tehsil|near|opp|pin|pincode|tel|fax|gujrat|gujarat)\b.*'
        t = re.sub(address_kw_pattern, ' ', t, flags=re.IGNORECASE)
        t = re.sub(r'[^A-Za-z0-9\s\.\'-]', ' ', t)
        t = ' '.join(t.split())
        if 'bank' in t.lower():
            m = re.search(r'([A-Za-z][\w\s\.\'-]*bank[\w\s\.\'-]*)', t, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return t.strip()

    def _looks_like_bank_continuation(self, text: str) -> bool:
        txt_l = text.lower()
        if self.bank_continuation_re.search(text):
            return True
        if "bank" in txt_l:
            return True
        return len(text.strip()) <= 20 and sum(c.isalpha() for c in text) >= 2