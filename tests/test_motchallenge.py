"""The labelled-data reader, and proof the metrics respond to identity errors."""

from __future__ import annotations

import numpy as np
import pytest

from vision_memory.motchallenge import find_sequences, load_sequence, metrics_module

pytest.importorskip("motmetrics")
mm = metrics_module()


def write_sequence(tmp_path, rows: str, length: int = 3):
    """A minimal on-disk sequence: seqinfo.ini plus gt/gt.txt, no images."""
    seq = tmp_path / "MOT-TEST"
    (seq / "gt").mkdir(parents=True)
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT-TEST\nimDir=img1\nframeRate=30\n"
        f"seqLength={length}\nimWidth=640\nimHeight=480\n",
        encoding="utf-8",
    )
    (seq / "gt" / "gt.txt").write_text(rows, encoding="utf-8")
    return seq


def test_labels_become_xyxy_boxes_with_stable_identities(tmp_path) -> None:
    seq = write_sequence(
        tmp_path,
        "1,7,10,20,30,40,1,1,1.0\n"
        "1,9,100,100,10,10,1,1,1.0\n"
        "2,7,12,22,30,40,1,1,1.0\n",
    )
    s = load_sequence(seq)
    assert s.name == "MOT-TEST" and s.length == 3 and s.fps == 30

    one = s.truth[1]
    assert np.allclose(one.boxes[0], [10, 20, 40, 60])  # x,y,w,h -> xyxy
    # The same person carries the same number into the next frame; that column
    # is the whole reason for using this data.
    assert 7 in set(one.ids) and 7 in set(s.truth[2].ids)


def test_unlabelled_and_non_pedestrian_rows_are_dropped(tmp_path) -> None:
    seq = write_sequence(
        tmp_path,
        "1,1,10,10,20,20,1,1,1.0\n"   # kept
        "1,2,30,30,20,20,0,1,1.0\n"   # keep flag clear
        "1,3,50,50,20,20,1,7,1.0\n",  # class 7 is not an upright pedestrian
    )
    s = load_sequence(seq)
    assert list(s.truth[1].ids) == [1]


def test_mostly_hidden_boxes_are_excluded_from_scoring(tmp_path) -> None:
    seq = write_sequence(
        tmp_path,
        "1,1,10,10,20,20,1,1,1.0\n"
        "1,2,30,30,20,20,1,1,0.05\n",
    )
    s = load_sequence(seq)
    assert list(s.visible(1, 0.25).ids) == [1]
    assert list(s.visible(1, 0.0).ids) == [1, 2]
    # A frame with no labels at all must still answer, not raise.
    assert len(s.visible(99, 0.25).ids) == 0


def test_missing_ground_truth_is_not_an_error(tmp_path) -> None:
    seq = write_sequence(tmp_path, "")
    (seq / "gt" / "gt.txt").unlink()
    assert load_sequence(seq).truth == {}


def test_find_sequences_accepts_a_single_directory_or_a_tree(tmp_path) -> None:
    seq = write_sequence(tmp_path, "1,1,10,10,20,20,1,1,1.0\n")
    assert find_sequences(seq) == [seq]
    assert find_sequences(tmp_path) == [seq]


def _accumulate(truth_ids, hyp_ids, boxes):
    """Score a hypothesis whose boxes sit exactly on the labelled ones."""
    acc = mm.MOTAccumulator(auto_id=False)
    for frame, (t, h) in enumerate(zip(truth_ids, hyp_ids), start=1):
        d = mm.distances.iou_matrix(boxes, boxes, max_iou=0.5)
        acc.update(list(t), list(h), d, frameid=frame)
    return mm.metrics.create().compute(acc, metrics=["idf1", "num_switches"])


def test_the_metrics_actually_notice_a_swapped_identity() -> None:
    """Guards the scoring wiring: a perfect run and a swap must not look alike."""
    boxes = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], dtype=np.float64)
    steady = _accumulate([[1, 2]] * 4, [[1, 2]] * 4, boxes)
    swapped = _accumulate([[1, 2]] * 4, [[1, 2], [1, 2], [2, 1], [2, 1]], boxes)

    assert steady["num_switches"].iloc[0] == 0
    assert swapped["num_switches"].iloc[0] > 0
    assert swapped["idf1"].iloc[0] < steady["idf1"].iloc[0]
