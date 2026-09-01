# Vision Memory

A local visual memory for video: it recognises objects it has seen before,
finds ones that look alike, and flags ones that do not belong.

All three come from a single idea. One frozen vision encoder turns every
tracked object into a vector, and each capability is a distance computation in
that one space:

| question | how it is answered |
| --- | --- |
| *Have I seen this before?* | nearest stored identity, above a calibrated boundary |
| *What looks like this?* | k-nearest neighbours over memory |
| *Is this normal?* | distance to the distribution of normal objects |
| *Where is the red backpack?* | the same, in a second space shared with text |

The encoder is never fine-tuned. A mutable encoder would invalidate every
vector already stored, so the backbone stays frozen and everything is built
around that constraint.

Runs entirely on a laptop CPU with integrated graphics. No GPU, no cloud, no
external services.

```
video -> detect -> track -> crop -> encode -> {memory, search, anomaly}
```

## Results

Scored against hand-labelled MOTChallenge identities, on a sequence used for
no tuning decision (MOT17-09):

| | IDF1 | identity switches |
| --- | --- | --- |
| tracker alone | 65.1% | 22 |
| **with re-identification** | **81.3%** | **9** |

IDF1 is the MOT benchmark's identity F1, computed by `motmetrics`. The tracker
row is the floor: what the numbers look like with re-identification switched
off. Every threshold in `configs/default.yaml` was chosen by running the whole
system end to end and reading these two columns, never by argument.

Honest limits, both measured rather than assumed:

- **The detector is the bottleneck, not the matching.** On MOT17-02 it finds
  69% of labelled people; the median labelled person is 39 px wide in a
  1920 px frame. Nobody can be re-identified who was never detected.
- **The scope is people.** On VisDrone drone footage the tracker reaches 31.0%
  IDF1 on vehicles and re-identification does not improve it. Cars viewed from
  above are close to identical, and the detector finds 26% of them. Objects are
  tracked and stored correctly; identifying *a particular car* is beyond these
  features.

## Live

```bash
python scripts/live_demo.py --source 0        # webcam, or a video file path
```

Three modes, switched with `1` `2` `3`, all of them at camera rate:

1. **Memory** — every object outlined and numbered, keeping its number when it
   leaves and returns.
2. **Search** — click an object; the closest matches in memory are ranked.
3. **Anomaly** — objects shaded by how unusual they are, with a heatmap showing
   *where* on the object is unusual rather than only how much.
4. **Language** — press `/` and type a description; the objects in memory that
   match it are highlighted. "a person in a red jacket" finds the person in the
   red jacket, and "a bicycle" correctly finds nothing when there is no bicycle.

Detection, embedding, segmentation and heatmaps each run on their own thread,
and the display projects boxes forward with optical flow between pipeline
passes, so the picture stays smooth while a pass takes about a second.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate     # source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Model files are not in the repository (they are large, and several carry their
own licences). `data/models/` needs:

| file | what it is | where from |
| --- | --- | --- |
| `yolo11n_768.onnx` | detector | export YOLO11n from `ultralytics` at `imgsz=768` |
| `yolo11n_seg_768.onnx` | display silhouettes | same, from `yolo11n-seg` |
| `person_reid_youtu_2021nov.onnx` | person appearance | OpenCV Zoo |

DINOv2 ViT-S/14, the general-object encoder, is fetched by `torch.hub` on first
use. `ultralytics` is only ever used to produce the ONNX files once, in a
throwaway environment; the project itself never imports it and runs the
detector through `onnxruntime` with its own decoding and NMS.

## Layout

```
src/vision_memory/     one module per pipeline stage
  encoder.py           frozen DINOv2, global and per-patch descriptors
  appearance.py        routes people to a specialist embedder, objects to DINOv2
  detector.py          YOLO11 ONNX: letterbox, decode, NMS, all in numpy
  segmenter.py         instance masks, for the display only
  tracker.py           ByteTrack-style: Kalman, two-pass association, lifecycle
  memory.py            SQLite identities joined to a FAISS exemplar index
  reid.py              pair mining and learned verification boundaries
  reidentifier.py      binding a track to a stored identity, and unbinding it
  search.py            the FAISS index
  anomaly.py           kNN / Mahalanobis / IsolationForest / OC-SVM
  heatmap.py           per-patch anomaly, so it can say where
  language.py          a CLIP space shared with text, for search by description
  engine.py            one pipeline, three reads
scripts/               entrypoints: demos, benchmarks, evaluation, calibration
tests/                 pytest
configs/default.yaml   every tunable, with the measurement behind it
```

## Notes on some choices

- **The embedder is split by what is being described.** A person-specific
  re-identification model beats a general one badly on people (held-out IDF1
  66.6% to 81.3%), but it only knows people. Objects go to DINOv2 with colour
  statistics; the two occupy disjoint slices, so a person is never compared
  against a suitcase.
- **Two boundaries, not one.** The threshold for "are these two crops the same
  person" does not transfer to "is this track the person behind that stored
  record", because the second maximises over several stored views and sits much
  higher. Using one boundary for both caused false merges.
- **Masks are for looking at, not for measuring.** Cropping to the silhouette
  before embedding sounds obviously right and made re-identification worse
  (held-out AUROC 0.979 to 0.962): the embedder was trained on rectangles with
  background in them.
- **Language search gets its own space, deliberately.** The identity vectors
  are trained so two people in similar coats land apart; text search needs the
  opposite, since "a red backpack" must match every red backpack. One space
  cannot do both, so CLIP vectors live in a second index keyed by identity and
  nothing about the identity path changes.
- **Several trained heads were built and rejected.** A logistic pair head, an
  MLP verifier, and a 768→128 projection all matched plain cosine or lost to
  it end to end. The projection is kept as a config option for its six-fold
  size reduction, and is off by default.

## Licence

MIT. Model weights are covered by their own licences.
