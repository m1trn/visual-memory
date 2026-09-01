"""Silhouettes for the display: YOLO11-seg masks, decoded in numpy.

For drawing only. Masked crops were measured to make re-identification WORSE
(held-out AUROC 0.979 -> 0.962: the person embedder was trained on rectangles
with background, and a cut-out silhouette is outside its training
distribution), so nothing here touches the embedding path. What an outline
buys is on screen: two people who overlap get one box each but two clearly
separate shapes.

YOLO11-seg emits ``(1, 4 + C + 32, A)`` predictions plus a ``(1, 32, H/4,
W/4)`` prototype bank. Each detection's 32 coefficients are a recipe for
mixing the prototypes into that instance's mask, which is why one small
forward pass yields a mask per object rather than one per pixel class.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from vision_memory.detector import Detection, YoloOnnxDetector, _nms

_PROTOTYPES = 32


@dataclass(frozen=True)
class Segmented:
    """A detection with the silhouette that produced it, in frame coordinates."""

    detection: Detection
    mask: np.ndarray  # bool, frame-sized


class YoloSegmenter(YoloOnnxDetector):
    """A detector that also returns each object's mask.

    Subclasses the detector so letterboxing, decoding, NMS and the class map
    stay in one place; ``detect`` keeps working unchanged for callers that only
    want boxes.
    """

    def detect_masks(self, frame_bgr: np.ndarray) -> list[Segmented]:
        """Detections with per-object masks, or plain boxes if the model has no mask head."""
        blob, scale, pad = self._letterbox(frame_bgr)
        outputs = self._session.run(None, {self._input_name: blob})
        if len(outputs) < 2:
            return [Segmented(d, _box_mask(d, frame_bgr.shape[:2])) for d in self.detect(frame_bgr)]

        preds = outputs[0][0].T                      # (A, 4 + C + 32)
        proto = outputs[1][0]                        # (32, h, w)
        n_class = preds.shape[1] - 4 - _PROTOTYPES
        boxes, scores, class_ids = self._decode(preds[:, : 4 + n_class])
        if len(boxes) == 0:
            return []
        keep_rows = self._kept_rows(preds[:, 4 : 4 + n_class])
        coeffs = preds[keep_rows, 4 + n_class :]

        size = self.cfg.input_size
        height, width = frame_bgr.shape[:2]
        out: list[Segmented] = []
        for i in _nms(boxes, scores, class_ids, self.cfg.iou_threshold):
            box = (boxes[i] - np.array([pad[0], pad[1], pad[0], pad[1]])) / scale
            det = Detection(tuple(float(v) for v in box), float(scores[i]), int(class_ids[i]),
                            self.names[int(class_ids[i])])
            m = _sigmoid(coeffs[i] @ proto.reshape(_PROTOTYPES, -1)).reshape(proto.shape[1:])
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
            # undo the letterbox, then back to frame size
            y1, y2 = pad[1], size - pad[1] if pad[1] else size
            x1, x2 = pad[0], size - pad[0] if pad[0] else size
            m = cv2.resize(m[y1:y2, x1:x2], (width, height), interpolation=cv2.INTER_LINEAR)
            mask = np.zeros((height, width), dtype=bool)
            bx1, by1 = max(int(box[0]), 0), max(int(box[1]), 0)
            bx2, by2 = min(int(box[2]), width), min(int(box[3]), height)
            if bx2 - bx1 < 2 or by2 - by1 < 2:
                continue
            # The mask is only meaningful inside its own box: prototypes are
            # shared, so a distant object of the same kind lights up too.
            mask[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2] > 0.5
            out.append(Segmented(det, _largest_piece(mask)))
        return out

    def _kept_rows(self, class_scores: np.ndarray) -> np.ndarray:
        """Row indices that survived ``_decode``'s confidence and class filters."""
        ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_scores)), ids]
        keep = scores >= self.cfg.conf_threshold
        if self.cfg.classes is not None:
            keep &= np.isin(ids, self.cfg.classes)
        return np.flatnonzero(keep)


def outline(mask: np.ndarray) -> list[np.ndarray]:
    """Contours of a mask, ready for ``cv2.polylines``."""
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(found)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _box_mask(det: Detection, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in det.box)
    mask[max(y1, 0): max(y2, 0), max(x1, 0): max(x2, 0)] = True
    return mask


def _largest_piece(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob.

    A segmenter under motion blur can attach a stray patch - a limb of the
    person behind, a bit of background - to an otherwise good silhouette. The
    body is the large piece; the strays are small and detached.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest
