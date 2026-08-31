import numpy as np


class GeometryUtilsMixin:
    """Bounding-box and point-ordering helpers shared across field extraction."""

    def _order_points(self, pts):
        pts = pts.reshape(4, 2).astype("float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _bbox_rect(self, bbox):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return min(xs), min(ys), max(xs), max(ys)

    def _bbox_left(self, bbox):
        return self._bbox_rect(bbox)[0]

    def _bbox_right(self, bbox):
        return self._bbox_rect(bbox)[2]

    def _row_center(self, item):
        ys = [p[1] for p in item["bbox"]]
        return sum(ys) / len(ys)

    def _row_height(self, item):
        ys = [p[1] for p in item["bbox"]]
        return max(ys) - min(ys)