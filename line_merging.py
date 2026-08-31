import re


class LineMergingMixin:
    """Groups raw OCR boxes into lines/blocks: same-line merge, address block,
    header (bank name) block, IFSC bbox recovery, payer-name extraction."""

    def _bank_name_parts_from_block(self, block):
        texts = []
        xs, ys = [], []
        for line in block:
            for member in line.get("members", [line]):
                txt = member.get("text", "")
                txt_l = txt.lower()
                if self._is_annotation_label(txt) or self._is_overlay_mark(txt):
                    continue
                if self._is_probable_garbage(txt, member.get("confidence", 1.0)):
                    continue
                if self.is_address(txt) or self.try_match_ifsc(txt):
                    continue
                if (
                    "bank" in txt_l
                    or self.bank_continuation_re.search(txt)
                    or re.search(r'\b(state|syndicate|hdfc|icici|axis|sbi|punjab|baroda)\b', txt_l)
                ):
                    texts.append(txt.strip())
                    xs.extend(p[0] for p in member["bbox"])
                    ys.extend(p[1] for p in member["bbox"])
        if xs:
            text = " ".join(t for t in texts if t)
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            return text, bbox
        return None, None

    def _bank_name_bbox_from_block(self, block):
        _, bbox = self._bank_name_parts_from_block(block)
        return bbox

    def _watermark_bank_fallback(self, extracted):
        bank_tokens = [
            it for it in extracted
            if "bank" in it["text"].lower() and not self._is_annotation_label(it["text"])
        ]
        if len(bank_tokens) < 3:
            return None

        bank_tokens.sort(key=lambda it: (self._row_center(it), self._bbox_left(it["bbox"])))
        anchor = bank_tokens[0]
        anchor_y = self._row_center(anchor)
        anchor_h = max(self._row_height(anchor), 1)

        same_row = [
            it for it in extracted
            if abs(self._row_center(it) - anchor_y) < anchor_h * 0.8
        ]
        same_row.sort(key=lambda it: self._bbox_left(it["bbox"]))
        anchor_idx = next((i for i, it in enumerate(same_row) if it is anchor), None)
        if anchor_idx is None:
            window = [anchor]
        else:
            window = same_row[max(0, anchor_idx - 1):anchor_idx + 2]
        if not window:
            return None

        text = " ".join(it["text"].strip() for it in window if it["text"].strip())
        xs = [p[0] for it in window for p in it["bbox"]]
        ys = [p[1] for it in window for p in it["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(it["confidence"] for it in window) / max(len(window), 1)
        return {"text": text.strip(), "bbox": bbox, "confidence": conf}

    def _ifsc_token_bbox(self, ifsc_code, source):
        members = source.get("members") or [source]
        for member in members:
            if self.try_match_ifsc(member.get("text", "")) == ifsc_code:
                return member["bbox"]
            tight = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
            if ifsc_code in tight:
                return member["bbox"]

        text = source.get("text", "")
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        idx = cleaned.find(ifsc_code)
        if idx >= 0:
            members_sorted = sorted(members, key=lambda m: self._bbox_left(m["bbox"])) if members else []
            offset = 0
            for member in members_sorted:
                m_cleaned = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
                m_len = len(m_cleaned)
                if offset <= idx < offset + m_len:
                    xs = [p[0] for p in member["bbox"]]
                    ys = [p[1] for p in member["bbox"]]
                    x0, x1 = min(xs), max(xs)
                    y0, y1 = min(ys), max(ys)
                    local_idx = idx - offset
                    span = max(m_len, 1)
                    char_w = (x1 - x0) / span
                    tx0 = x0 + local_idx * char_w
                    tx1 = min(tx0 + len(ifsc_code) * char_w, x1)
                    return [[tx0, y0], [tx1, y0], [tx1, y1], [tx0, y1]]
                offset += m_len
            overlap_bboxes = []
            offset = 0
            for member in members_sorted:
                m_cleaned = re.sub(r'[^A-Za-z0-9]', '', member.get("text", "")).upper()
                m_len = len(m_cleaned)
                if offset < idx + len(ifsc_code) and offset + m_len > idx:
                    overlap_bboxes.append(member["bbox"])
                offset += m_len
            if overlap_bboxes:
                xs = [p[0] for b in overlap_bboxes for p in b]
                ys = [p[1] for b in overlap_bboxes for p in b]
                return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        return source["bbox"]

    def merge_same_line(self, extracted, overlap_ratio_threshold=0.5, max_x_gap_ratio=4.0, max_x_gap_width_ratio=0.12):
        if not extracted:
            return []
        all_xs = [p[0] for it in extracted for p in it["bbox"]]
        all_ys = [p[1] for it in extracted for p in it["bbox"]]
        width_est = max(all_xs) if all_xs else 1
        height_est = max(all_ys) if all_ys else 1
        items = sorted(extracted, key=self._row_center)
        lines = []
        for item in items:
            item_top = min(p[1] for p in item["bbox"])
            item_bottom = max(p[1] for p in item["bbox"])
            item_left = min(p[0] for p in item["bbox"])
            item_right = max(p[0] for p in item["bbox"])
            item_height = max(item_bottom - item_top, 1)
            item_y_ratio = self._row_center({"bbox": item["bbox"]}) / max(height_est, 1)

            placed = False
            for line in lines:
                overlap = min(line["bottom"], item_bottom) - max(line["top"], item_top)
                shorter = min(item_height, line["bottom"] - line["top"])
                if overlap <= 0 or overlap / shorter < overlap_ratio_threshold:
                    continue
                if item_left > line["right"]:
                    x_gap = item_left - line["right"]
                elif line["left"] > item_right:
                    x_gap = line["left"] - item_right
                else:
                    x_gap = 0
                height_based_cap = max(item_height, line["bottom"] - line["top"], 10) * max_x_gap_ratio
                width_based_cap = width_est * max_x_gap_width_ratio
                max_allowed_gap = min(height_based_cap, width_based_cap)
                line_y_ratio = ((line["top"] + line["bottom"]) / 2) / max(height_est, 1)
                if item_y_ratio < 0.28 or line_y_ratio < 0.28:
                    max_allowed_gap = min(max_allowed_gap, width_est * 0.045)
                if x_gap > max_allowed_gap:
                    continue
                line["members"].append(item)
                line["top"] = min(line["top"], item_top)
                line["bottom"] = max(line["bottom"], item_bottom)
                line["left"] = min(line["left"], item_left)
                line["right"] = max(line["right"], item_right)
                placed = True
                break
            if not placed:
                lines.append({"top": item_top, "bottom": item_bottom,
                               "left": item_left, "right": item_right, "members": [item]})

        merged = []
        for line in lines:
            line_sorted = sorted(line["members"], key=lambda it: self._bbox_left(it["bbox"]))
            text = " ".join(it["text"].strip() for it in line_sorted)
            text_tight = "".join(it["text"].strip() for it in line_sorted)
            xs = [p[0] for it in line_sorted for p in it["bbox"]]
            ys = [p[1] for it in line_sorted for p in it["bbox"]]
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            conf = sum(it["confidence"] for it in line_sorted) / len(line_sorted)
            merged.append({
                "text": text, "text_tight": text_tight, "bbox": bbox,
                "confidence": conf, "members": line_sorted,
            })
        merged.sort(key=self._row_center)
        return merged

    def merge_address_block(self, lines, img_height, max_gap_ratio=0.05):
        candidates = [
            l for l in lines
            if self._row_center(l) / img_height < 0.30
            and not self._is_overlay_mark(l["text"])
            and not self._is_annotation_label(l["text"])
            and not self._is_probable_garbage(l["text"], l["confidence"])
        ]
        candidates.sort(key=self._row_center)
        blocks = []
        current = []
        for l in candidates:
            if not self.is_address(l["text"]) and not self.branch_code_re.search(l["text"]):
                continue
            if not current:
                current = [l]
                continue
            gap = self._row_center(l) - self._row_center(current[-1])
            if gap / img_height <= max_gap_ratio:
                current.append(l)
            else:
                blocks.append(current)
                current = [l]
        if current:
            blocks.append(current)

        results = []
        for block in blocks:
            text = ", ".join(b["text"].strip() for b in block if b["text"].strip())
            xs = [p[0] for b in block for p in b["bbox"]]
            ys = [p[1] for b in block for p in b["bbox"]]
            bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
            conf = sum(b["confidence"] for b in block) / len(block)
            results.append({"text": text, "bbox": bbox, "confidence": conf})
        return results

    def _extract_address_by_geometry(self, lines, bank_bbox, ifsc_bbox, width_est, pad_ratio=0.35, x_limit_ratio=0.62):
        if not bank_bbox or not ifsc_bbox:
            return None

        _, bank_top, _, bank_bottom = self._bbox_rect(bank_bbox)
        ifsc_left, ifsc_top, _, _ = self._bbox_rect(ifsc_bbox)
        if ifsc_top <= bank_bottom:
            return None

        band_height = max(ifsc_top - bank_bottom, 1)
        pad = band_height * pad_ratio
        y_start = bank_bottom - pad
        y_end = ifsc_top + pad
        x_limit = width_est * x_limit_ratio

        candidates = []
        for line in lines:
            raw_txt = line["text"].strip()
            if not raw_txt:
                continue
            if self._is_overlay_mark(raw_txt) or self._is_annotation_label(raw_txt):
                continue
            if self._is_probable_garbage(raw_txt, line["confidence"]):
                continue
            if any(kw in raw_txt.lower() for kw in self.non_field_keywords):
                continue

            txt = raw_txt
            if self.try_match_ifsc(raw_txt):
                stripped = re.sub(r'IFS?C?\s*:?\s*[A-Z0-9]{6,11}', '', raw_txt, flags=re.IGNORECASE)
                stripped = re.sub(r'SWIFT\s*:?\s*[A-Z0-9]*', '', stripped, flags=re.IGNORECASE)
                stripped = stripped.strip(' :,-')
                if len(stripped) < 5:
                    continue
                txt = stripped

            cy = self._row_center(line)
            cx = self._bbox_left(line["bbox"])
            if not (y_start <= cy <= y_end) or cx > x_limit:
                continue
            candidates.append({**line, "text": txt})

        if not candidates:
            return None

        candidates.sort(key=self._row_center)
        text = ", ".join(c["text"].strip() for c in candidates if c["text"].strip())
        xs = [p[0] for c in candidates for p in c["bbox"]]
        ys = [p[1] for c in candidates for p in c["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(c["confidence"] for c in candidates) / len(candidates)
        return {"text": text, "bbox": bbox, "confidence": conf}

    def merge_header_block(self, lines, img_height, max_gap_ratio=0.10, y_limit=0.25,
                            height_ratio_thresh=0.55):
        all_xs = [p[0] for l in lines for p in l["bbox"]]
        width_est = max(all_xs) if all_xs else 1
        x_limit = 0.58
        candidates = [
            l for l in lines
            if self._row_center(l) / img_height < y_limit
            and self._bbox_left(l["bbox"]) / width_est < x_limit
            and not self._is_annotation_label(l["text"])
            and not self._is_overlay_mark(l["text"])
            and not self._is_probable_garbage(l["text"], l["confidence"])
            and (not self.is_address(l["text"]) or self._has_bank_evidence(l["text"]))
            and not self.try_match_ifsc(l["text"])
            and not self.branch_code_re.search(l["text"])
            and not self._looks_like_person_name(l["text"])
            and not any(kw in l["text"].lower() for kw in self.non_field_keywords)
        ]
        bankish = [l for l in candidates if self._has_bank_evidence(l["text"])]
        if not bankish:
            return None, []
        candidates = bankish

        anchor = max(candidates, key=self._row_height)
        anchor_height = max(self._row_height(anchor), 1)

        block = [anchor]
        remaining = [l for l in candidates if l is not anchor]

        changed = True
        while changed and remaining:
            changed = False
            for l in list(remaining):
                gap = min(abs(self._row_center(l) - self._row_center(b)) for b in block)
                if gap / img_height > max_gap_ratio:
                    continue
                txt = l["text"]
                height_ok = self._row_height(l) >= anchor_height * height_ratio_thresh
                keyword_continuation = bool(
                    self.bank_continuation_re.search(txt) or "bank" in txt.lower()
                )
                if height_ok or keyword_continuation:
                    block.append(l)
                    remaining.remove(l)
                    changed = True

        block.sort(key=self._row_center)
        raw_text, bbox = self._bank_name_parts_from_block(block)
        if not raw_text or not bbox:
            return None, []
        text = self._clean_bank_name(raw_text) or raw_text
        if not self._looks_like_bank_name(text):
            return None, []
        conf = sum(b["confidence"] for b in block) / len(block)
        return {"text": text, "bbox": bbox, "confidence": conf}, block

    def _overlay_exclusion_zones(self, extracted):
        zones = []
        for item in extracted:
            if self._is_overlay_mark(item["text"]) or self._is_annotation_label(item["text"]):
                xs = [p[0] for p in item["bbox"]]
                ys = [p[1] for p in item["bbox"]]
                zones.append((min(xs), min(ys), max(xs), max(ys)))
        return zones

    def _in_exclusion_zone(self, bbox, zones, overlap_thresh=0.3):
        if not zones:
            return False
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
        b_area = max(bx1 - bx0, 1) * max(by1 - by0, 1)
        for zx0, zy0, zx1, zy1 in zones:
            ix0, iy0 = max(bx0, zx0), max(by0, zy0)
            ix1, iy1 = min(bx1, zx1), min(by1, zy1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / b_area > overlap_thresh:
                    return True
        return False

    def _looks_like_payer_name_text(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return False
        tl = t.lower()
        block_kw = ('rupees', 'signature', 'valid for', 'a/c', 'account',
                    'ifsc', 'branch', 'cheque', 'chq', 'date', 'dd mm yyyy')
        if any(kw in tl for kw in block_kw):
            return False
        if self._is_overlay_mark(t) or self._is_annotation_label(t):
            return False
        return True

    def extract_payer_name(self, extracted, img_height):
        pay_anchor = None
        for item in extracted:
            t = item["text"].strip().lower().rstrip(':')
            if t != 'pay':
                continue
            y_ratio = self._row_center(item) / img_height if img_height else 0
            if y_ratio > 0.6:
                continue
            if pay_anchor is None or self._row_center(item) < self._row_center(pay_anchor):
                pay_anchor = item

        if pay_anchor is None:
            return None

        anchor_y = self._row_center(pay_anchor)
        anchor_h = max(self._row_height(pay_anchor), 1)
        anchor_right = self._bbox_rect(pay_anchor["bbox"])[2]

        same_row = [
            it for it in extracted
            if it is not pay_anchor
            and abs(self._row_center(it) - anchor_y) < anchor_h * 1.1
            and self._bbox_left(it["bbox"]) >= anchor_right - anchor_h * 0.5
        ]
        same_row.sort(key=lambda it: self._bbox_left(it["bbox"]))

        name_parts = []
        for it in same_row:
            t = it["text"].strip()
            tl = t.lower()
            if re.match(r'^or\b', tl) or 'bearer' in tl or re.search(r'\border\b', tl):
                break
            if not self._looks_like_payer_name_text(t):
                continue
            name_parts.append(it)

        if not name_parts:
            return None

        text = " ".join(p["text"].strip() for p in name_parts).strip(' .:-')
        if not text:
            return None
        xs = [p[0] for it in name_parts for p in it["bbox"]]
        ys = [p[1] for it in name_parts for p in it["bbox"]]
        bbox = [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]
        conf = sum(it["confidence"] for it in name_parts) / len(name_parts)
        return {"text": text, "bbox": bbox, "confidence": conf}