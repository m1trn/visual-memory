import numpy as np

from vision_memory.detector import Detection, _nms


def test_nms_suppresses_same_class_keeps_other_class() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert _nms(boxes, scores, np.array([0, 0, 0]), 0.5) == [0, 2]
    # same overlap, different classes -> both survive
    assert _nms(boxes, scores, np.array([0, 1, 0]), 0.5) == [0, 1, 2]


def test_crop_clamps_to_frame() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    d = Detection(box=(-5.0, 10.0, 40.0, 25.0), score=1.0, class_id=0, label="x")
    assert d.crop(frame).shape == (10, 30, 3)
