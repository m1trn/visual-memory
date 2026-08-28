"""How an object's appearance is described: learned features plus clothing colour.

The frozen encoder is good at shape and texture but, on 30-pixel-wide pedestrian
crops, two strangers in dark coats look much alike. Colour is the complementary
evidence — and specifically colour *by body band*, because "dark jacket, blue
jeans" and "dark jacket, dark trousers" are different people even when the
learned features cannot separate them.

Measured on this footage, matching the same person across a three-second gap
against provably-different people:

    DINOv2 on the torso crop        AUROC 0.912, 62.1% recall at 5% false accepts
    colour bands alone              AUROC 0.900, 63.2%
    both, fused                     AUROC 0.930, 68.8%

The two halves are scaled by ``sqrt(1 - w)`` and ``sqrt(w)`` before being
concatenated, which makes the fused vector's cosine exactly the weighted sum of
the two cosines. Everything downstream — the index, memory, re-identification,
the tracker's appearance veto — keeps working on plain dot products and needs to
know nothing about this.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import onnxruntime as ort

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:  # only for the signature; appearance does not depend on detection
    from vision_memory.detector import Detection

from vision_memory.config import AppearanceConfig
from vision_memory.encoder import Encoder

_HSV_BINS = (8, 8, 4)  # hue is what identifies clothing; value is mostly lighting
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def colour_bands(crop_bgr: np.ndarray, bands: int) -> np.ndarray:
    """Unit-norm HSV histogram per horizontal band of a crop, concatenated."""
    if crop_bgr.size == 0:
        raise ValueError("colour_bands needs a non-empty crop")
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    edges = [round(height * k / bands) for k in range(bands + 1)]
    parts = []
    for lo, hi in zip(edges, edges[1:]):
        band = hsv[lo:hi] if hi > lo else hsv
        hist = cv2.calcHist([band], [0, 1, 2], None, list(_HSV_BINS), [0, 180, 0, 256, 0, 256])
        parts.append(hist.flatten())
    return _unit(np.concatenate(parts).astype(np.float32))


def fuse(deep: np.ndarray, colour: np.ndarray, weight: float) -> np.ndarray:
    """Combine unit-norm halves so the result's cosine is their weighted sum."""
    return np.concatenate([
        np.sqrt(1.0 - weight) * np.asarray(deep, dtype=np.float32),
        np.sqrt(weight) * np.asarray(colour, dtype=np.float32),
    ]).astype(np.float32)


class ReidNet:
    """A network trained for person re-identification, run through onnxruntime.

    A general-purpose encoder describes what a crop looks like; this describes
    what makes one person distinguishable from another, which is a different
    question and the one being asked here. Measured on this footage at a
    three-second gap: recall at 5% false accepts rises from 62.1% to 83.7%, and
    it is roughly ten times faster because the network is far smaller.
    """

    def __init__(self, model_path: str, num_threads: int = 0) -> None:
        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        spec = self._session.get_inputs()[0]
        self._input = spec.name
        # These models are commonly exported with a fixed batch; short batches
        # are padded and the padding discarded.
        self._batch = spec.shape[0] if isinstance(spec.shape[0], int) else 0
        self._h, self._w = int(spec.shape[2]), int(spec.shape[3])
        # Some exports leave the feature as (N, C, 1, 1). The width is the
        # product of everything after the batch axis, and the output is
        # flattened to match, so dim and the vectors agree.
        out_shape = self._session.get_outputs()[0].shape[1:]
        self.dim = int(np.prod([int(d) for d in out_shape])) if all(
            isinstance(d, int) for d in out_shape) else int(out_shape[0])

    def encode_batch(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Embed RGB crops into unit-norm vectors, shape ``(N, dim)``."""
        prepared = (np.stack([self._prepare(c) for c in crops_rgb]) if len(crops_rgb)
                    else np.empty((0, 3, self._h, self._w), np.float32))
        out = []
        step = self._batch or len(prepared) or 1
        for start in range(0, len(prepared), step):
            chunk = prepared[start : start + step]
            if self._batch and len(chunk) < self._batch:
                padded = np.zeros((self._batch, 3, self._h, self._w), np.float32)
                padded[: len(chunk)] = chunk
                got = self._session.run(None, {self._input: padded})[0][: len(chunk)]
                got = np.asarray(got).reshape(len(chunk), -1)
            else:
                got = self._session.run(None, {self._input: chunk})[0]
                got = np.asarray(got).reshape(len(chunk), -1)
            out.append(np.asarray(got, dtype=np.float32))
        if not out:
            return np.empty((0, self.dim), np.float32)
        v = np.concatenate(out)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def _prepare(self, rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(rgb, (self._w, self._h)).astype(np.float32) / 255.0
        return ((img - _IMAGENET_MEAN) / _IMAGENET_STD).transpose(2, 0, 1)


class Embedder(Protocol):
    """Anything that turns RGB crops into unit-norm vectors."""

    dim: int

    def encode_batch(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray: ...


def build_describer(cfg: AppearanceConfig) -> RoutedDescriber | AppearanceDescriber:
    """The appearance pipeline described by the config, ready to describe boxes."""
    general_embedder, general_upper = build_embedder(dataclasses.replace(cfg, model="encoder"))
    general = AppearanceDescriber(general_embedder, cfg, general_upper)
    if cfg.model != "reid":
        return general
    specialist_embedder, specialist_upper = build_embedder(cfg)
    specialist = AppearanceDescriber(
        specialist_embedder, dataclasses.replace(cfg, colour_weight=0.0), specialist_upper
    )
    return RoutedDescriber(specialist, general, frozenset(cfg.specialist_labels))


def build_embedder(cfg: AppearanceConfig) -> tuple[Embedder, float]:
    """The configured appearance model, and how much of a box it should see.

    A re-identification network is trained on whole-body crops at a fixed aspect
    ratio, so it gets the entire box. A general encoder does better on the top of
    the box alone, where a person's identity lives and where an occluder
    interferes least.
    """
    if cfg.model == "reid":
        return ReidNet(cfg.reid_model_path), 1.0
    if cfg.model == "encoder":
        from vision_memory.config import load_encoder_config
        return Encoder(load_encoder_config()), cfg.deep_upper_fraction
    raise ValueError(f"unknown appearance model: {cfg.model!r}")


class AppearanceDescriber:
    """Turns a frame and a box into the vector the rest of the system stores."""

    def __init__(self, encoder: Embedder, cfg: AppearanceConfig,
                 upper_fraction: float | None = None) -> None:
        self.encoder = encoder
        self.cfg = cfg
        self.upper_fraction = (cfg.deep_upper_fraction if upper_fraction is None
                               else upper_fraction)

    @property
    def dim(self) -> int:
        """Width of the fused vector."""
        if self.cfg.colour_weight <= 0.0:
            return self.encoder.dim
        return self.encoder.dim + int(np.prod(_HSV_BINS)) * self.cfg.colour_bands

    def describe(self, frame_bgr: np.ndarray, boxes: list[np.ndarray]) -> dict[int, np.ndarray]:
        """Describe each box, skipping any too small to carry a usable signal.

        Returns a mapping from position in ``boxes`` to its vector, so callers
        can tell which boxes were skipped rather than silently mis-aligning.
        """
        crops_deep, crops_colour, kept = [], [], []
        height, width = frame_bgr.shape[:2]
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(v) for v in box)
            upper = int(round(y1 + (y2 - y1) * self.upper_fraction))
            x1i, y1i = max(int(round(x1)), 0), max(int(round(y1)), 0)
            x2i, y2i = min(int(round(x2)), width), min(int(round(y2)), height)
            upper = min(max(upper, y1i), y2i)
            if x2i - x1i < self.cfg.min_crop_px or upper - y1i < self.cfg.min_crop_px:
                continue
            crops_deep.append(cv2.cvtColor(frame_bgr[y1i:upper, x1i:x2i], cv2.COLOR_BGR2RGB))
            crops_colour.append(frame_bgr[y1i:y2i, x1i:x2i])
            kept.append(index)
        if not kept:
            return {}

        vectors = self.encoder.encode_batch(crops_deep)
        if self.cfg.colour_weight <= 0.0:
            return dict(zip(kept, vectors))
        return {
            index: fuse(deep, colour_bands(colour, self.cfg.colour_bands), self.cfg.colour_weight)
            for index, deep, colour in zip(kept, vectors, crops_colour)
        }


class RoutedDescriber:
    """Describes each object with whichever model is good at its kind of thing.

    A person re-identification network is trained to separate people and is far
    better at it than a general encoder, but it collapses everything else: on
    assorted objects it rates unrelated images at 0.454 where the general
    encoder gives 0.031, and it retrieves 4 of 6 known pairs against 6 of 6.
    Measured the other way round, on people, the re-id network reaches 0.869
    AUROC against the general encoder's 0.683. Neither is the right answer for
    both, so each object goes to the model that can actually see it.

    The two vectors occupy separate slices of one combined vector, zero
    elsewhere. Within a kind, cosine is exactly the model's own cosine; across
    kinds it is zero, which is correct, since a person and a car are never the
    same object. Everything downstream keeps working on plain dot products.
    """

    def __init__(self, specialist: AppearanceDescriber, general: AppearanceDescriber,
                 specialist_labels: frozenset[str]) -> None:
        self.specialist = specialist
        self.general = general
        self.specialist_labels = specialist_labels

    @property
    def dim(self) -> int:
        return self.specialist.dim + self.general.dim

    def describe(self, frame_bgr: np.ndarray, boxes: Sequence[np.ndarray],
                 labels: Sequence[str]) -> dict[int, np.ndarray]:
        """Describe each box using the model suited to its label."""
        groups: dict[bool, list[int]] = {True: [], False: []}
        for index, label in enumerate(labels):
            groups[label in self.specialist_labels].append(index)

        out: dict[int, np.ndarray] = {}
        for is_specialist, indices in groups.items():
            if not indices:
                continue
            model = self.specialist if is_specialist else self.general
            described = model.describe(frame_bgr, [boxes[i] for i in indices])
            offset = 0 if is_specialist else self.specialist.dim
            for local, vector in described.items():
                slot = np.zeros(self.dim, dtype=np.float32)
                slot[offset : offset + len(vector)] = vector
                out[indices[local]] = slot
        return out


def describe_detections(
    describer: "RoutedDescriber | AppearanceDescriber",
    frame_bgr: np.ndarray,
    detections: Sequence["Detection"],
    min_score: float = 0.0,
) -> dict[int, np.ndarray]:
    """Embed the detections worth embedding, keyed by position in ``detections``.

    The one path from boxes to vectors. Seven scripts previously each built
    their own call, and three times a change was measured in one and left stale
    in another, so there is deliberately nothing left to keep in sync.

    ``min_score`` should be the tracker's ``min_exemplar_confidence``. Nothing
    below it can ever be consumed: the tracker reads appearance only in its
    high-confidence association pass, and the exemplar gate refuses weaker
    crops. Profiled on the demo, embedding every box down to the detector's
    0.1 floor cost 2.7 s per detector frame - dozens of surfboards, skis and
    fragments of people - against 0.2 s for detection itself.
    """
    keep = [i for i, d in enumerate(detections) if float(d.score) >= min_score]
    if not keep:
        return {}
    boxes = [detections[i].box for i in keep]
    if isinstance(describer, RoutedDescriber):
        out = describer.describe(frame_bgr, boxes, [detections[i].label for i in keep])
    else:
        # A plain describer uses one model for everything and takes no labels.
        out = describer.describe(frame_bgr, boxes)
    return {keep[local]: vector for local, vector in out.items()}


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v if norm == 0.0 else (v / norm).astype(np.float32)
