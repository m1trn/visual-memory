"""Object detection: BGR frame -> list of boxes in original pixel coordinates.

Backends implement ``Detector``; the first is a YOLO11 ONNX model run through
onnxruntime with numpy pre/post-processing (letterbox, decode, NMS).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import onnxruntime as ort

from vision_memory.config import DetectorConfig

_PAD_VALUE = 114  # YOLO letterbox fill (gray), matches training


@dataclass(frozen=True)
class Detection:
    """One detected object; ``box`` is (x1, y1, x2, y2) in source-frame pixels."""

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    label: str

    def crop(self, frame: np.ndarray, upper_fraction: float = 1.0) -> np.ndarray:
        """Slice this box out of ``frame`` (HWC), clamped to the image.

        ``upper_fraction`` keeps only the top of the box. For people that is
        head and torso, which carries the clothing that tells one person from
        another; legs are largely generic and are the first thing hidden when
        somebody walks in front. Measured over 500 frames, cropping to the top
        60% drops the similarity between *different* people from 0.604 to 0.482
        while barely moving same-person similarity, so the two distributions
        stop overlapping.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.box
        if upper_fraction < 1.0:
            y2 = y1 + (y2 - y1) * upper_fraction
        return frame[max(int(y1), 0) : min(int(y2), h), max(int(x1), 0) : min(int(x2), w)]


class Detector(Protocol):
    """Anything that turns a BGR frame into detections."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


class YoloOnnxDetector:
    """YOLO11 (Ultralytics export) via onnxruntime.

    Expects input ``(1, 3, S, S)`` RGB float in [0, 1] and output
    ``(1, 4 + C, A)``: per-anchor cx, cy, w, h followed by C class scores.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        opts = ort.SessionOptions()
        if cfg.num_threads > 0:
            opts.intra_op_num_threads = cfg.num_threads
        self._session = ort.InferenceSession(cfg.model_path, opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        meta = self._session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"])

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run the model on one frame and return NMS-filtered detections."""
        blob, scale, pad = self._letterbox(frame_bgr)
        raw = self._session.run(None, {self._input_name: blob})[0][0]  # (4 + C, A)
        boxes, scores, class_ids = self._decode(raw.T)
        if len(boxes) == 0:
            return []
        boxes = (boxes - np.array([pad[0], pad[1], pad[0], pad[1]])) / scale
        keep = _nms(boxes, scores, class_ids, self.cfg.iou_threshold)
        return [
            Detection(tuple(float(v) for v in boxes[i]), float(scores[i]), int(class_ids[i]), self.names[int(class_ids[i])])
            for i in keep
        ]

    def _letterbox(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize keeping aspect ratio, pad to a square. Returns (blob, scale, (pad_x, pad_y))."""
        size = self.cfg.input_size
        h, w = frame_bgr.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = round(h * scale), round(w * scale)
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
        canvas = np.full((size, size, 3), _PAD_VALUE, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), scale, (pad_x, pad_y)

    def _decode(self, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(A, 4 + C) -> confidence-filtered xyxy boxes, scores, class ids (letterbox space)."""
        class_scores = preds[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(preds)), class_ids]
        mask = scores >= self.cfg.conf_threshold
        if self.cfg.classes is not None:
            mask &= np.isin(class_ids, self.cfg.classes)
        cx, cy, w, h = preds[mask, 0], preds[mask, 1], preds[mask, 2], preds[mask, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        return boxes, scores[mask], class_ids[mask]


def _nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy per-class NMS. Boxes of different classes are offset so they never overlap."""
    offset = class_ids[:, None] * (boxes.max() + 1.0)
    b = boxes + offset
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]
    return keep
