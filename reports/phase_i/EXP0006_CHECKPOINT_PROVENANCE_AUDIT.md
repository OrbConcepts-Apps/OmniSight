# EXP-0006 Checkpoint Provenance Audit

Zero-live-call, zero-internet-fetch audit. Every fact below was extracted directly from
local repository files this session (`research/provenance.py`,
`research/provenance_registry_exp0006.json` is the machine-readable snapshot) — nothing
copied from an earlier report without independent re-verification, nothing inferred from
a filename.

## Artifact identification — three distinct forms found

Confirmed, per the eligibility audit's own instruction not to assume the `.pt` file is
what's actually shipped:

| Form | Path | SHA-256 |
|---|---|---|
| **TRAINING-SOURCE CHECKPOINT** | `benchmark/models/yolov8m-oiv7.pt` | `21ffa3718c577ac23e708e4c0544c49a20682efa03914d5f816166b54e8fd3fe` |
| **CONVERTED DEPLOYMENT MODEL** | `ios/OmniSightApp/ScanningData.mlpackage/Data/com.apple.CoreML/model.mlmodel` | `f56055a0d89e8a21c705c1a929127df3c42d454b5ec791aa84293de5b39c4ba0` |
| **ACTUALLY BUNDLED APP ARTIFACT (weights)** | `.../ScanningData.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `06f3c282a285b88f3ce0ffbde6e6f9da5da9634092f8f6b596b3791e093befe9` |

A fourth candidate form (`assets/models/yolov8_quantized.onnx`, referenced in the repo's
very first bulk-import commit) **no longer exists** in the working tree — superseded by
the current `.mlpackage`, confirmed absent by search; not part of the current shipped
artifact.

**Git tracking status**: the `.pt` file is `.gitignore`d (`benchmark/models/*.pt`) —
**never committed**, provisioned locally outside version control. The `.mlpackage`
(both `model.mlmodel` and `weights/weight.bin`) **is** tracked directly in git, no LFS,
committed in this repo's initial bulk-import commit (`eb56679`, "Final stabilization
pt 2", 2026-04-30) — that commit is itself a squashed history import with no
per-file incremental provenance trail of its own.

## Training-source checkpoint — read directly via `torch.load()`

`benchmark/models/yolov8m-oiv7.pt`'s own embedded dict (Ultralytics' standard checkpoint
format always stores `date`/`version`/`license`/`docs`/`train_args`):

- `date`: **`2023-08-24T18:32:24.194402`** — VERIFIED (read directly, not inferred)
- `version`: **`8.0.157`** (Ultralytics version used for the ORIGINAL TRAINING run) — VERIFIED
- `license`: **`AGPL-3.0 License (https://ultralytics.com/license)`** — VERIFIED
- `train_args.task`: `detect`; `train_args.model`: `yolov8m.yaml`; `train_args.data`:
  `/usr/src/ultralytics/ultralytics/cfg/datasets/open-images-v7.yaml` — VERIFIED (this
  is the real training-time dataset config path, embedded in the checkpoint itself)
- `train_args.epochs`: `50`; `optimizer`: `SGD`; `lr0`: `0.01`; `seed`: `0`;
  `deterministic`: `True`; `pretrained`: `True`; `batch`: `120`; `device`: `2` — the full
  real training recipe, VERIFIED, previously marked entirely UNKNOWN in the EXP-0006
  preregistration's `training_config` — this closes a real, previously-open prerequisite.
- The 2023-08-24 training date **predates this project's own timeline (2026) by ~2.5
  years**, and no local training infrastructure/GPU history exists that could have
  produced it — this strongly SUPPORTS (does not, by itself, cryptographically prove)
  that this is Ultralytics' own officially-released pretrained checkpoint, not a
  locally-produced or covertly fine-tuned artifact.

## Deployed CoreML artifact — read directly via raw byte/string extraction

`coremltools`' native proto-reading libraries are unavailable on Windows (confirmed:
`ImportError: No module named 'coremltools.libcoremlpython'`), so `model.mlmodel`'s
embedded metadata was extracted by scanning the raw protobuf bytes for printable-ASCII
runs (`research.provenance.extract_mlmodel_strings`) — a deterministic, dependency-free
technique, not a claim of full protobuf parsing:

- Embedded description: **`"Ultralytics YOLOv8m model trained on
  /usr/src/ultralytics/ultralytics/cfg/datasets/open-images-v7.yaml"`** — VERIFIED,
  matches the `.pt` checkpoint's own `train_args.data` exactly.
- `com.github.apple.coremltools.conversion_date`: **`2026-04-25`**
  (`2026-04-25T19:33:05.010338`) — VERIFIED. This is the **conversion** date, ~2.5 years
  after the original 2023 training date — consistent with "an old, stock checkpoint
  loaded and exported later," not a contradiction.
- `com.github.apple.coremltools.source_dialect`: `TorchScript`; embedded `torch==2.11.0`;
  Ultralytics export version `8.4.41` (distinct from the training-time `8.0.157` —
  expected: conversion tooling was updated between training and export) — VERIFIED.
- Export args embedded: `{'batch': 1, 'half': False, 'int8': False, 'dynamic': False,
  'nms': True}` — VERIFIED (NMS baked into the CoreML graph, no separate NMS code
  needed in Swift, matching `OMNISIGHT_ARCHITECTURE.md`'s own independent finding).
- License string: **`AGPL-3.0 License (https://ultralytics.com/license)`** — VERIFIED,
  embedded directly in the exported model, matching the `.pt` checkpoint's license.
- Class taxonomy: the full 601-class Open Images V7 label map is embedded verbatim in
  the model spec (index 381 = `'Person'`) — VERIFIED, matches production's documented
  601-class OIV7 vocabulary exactly.

## Provenance chain

```
Ultralytics official YOLOv8m/OIV7 training run (2023-08-24, ultralytics 8.0.157)
  -> benchmark/models/yolov8m-oiv7.pt   [TRAINING-SOURCE CHECKPOINT, VERIFIED contents]
  -> coremltools TorchScript->CoreML export (2026-04-25, ultralytics 8.4.41, torch 2.11.0)
  -> ios/.../model.mlmodel + weights/weight.bin   [DEPLOYED/BUNDLED ARTIFACT, VERIFIED contents]
```

The link between the two artifacts is classified **SUPPORTED, not VERIFIED**: both
independently embed identical dataset-config path, identical license string, and the
same architecture/class-taxonomy — extremely strong circumstantial agreement — but this
audit did not (and could not, without functioning `coremltools` conversion on this
Windows machine) cryptographically prove byte-for-byte weight identity across the two
different file formats. This is an honest precision distinction, not a gap glossed over.

## License audit

- **Weight/architecture code license**: `AGPL-3.0 License (https://ultralytics.com/license)`
  — VERIFIED, embedded identically in both the `.pt` checkpoint and the CoreML export.
  Per Ultralytics' own terms, commercial closed-source use requires an Ultralytics
  Enterprise license — a fact already surfaced during EXP-0005 (`research/_exp0005_preregister.py`),
  now independently re-confirmed by direct extraction rather than citation.
- **Training data license**: Open Images V7 annotations are CC BY 4.0, images are
  individually CC BY 2.0-licensed per Google's own disclaimer — documented in
  `docs/DATASETS.md` (existing, pre-verified evidence, re-cited not re-fetched).
- **`weights/weight.bin`'s own license**: UNKNOWN as a standalone artifact (it is a raw
  tensor blob with no embedded text metadata of its own); governed by `model.mlmodel`'s
  license, which IS verified.
- **Source URL** (`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-oiv7.pt`,
  recorded in `benchmark/diagnostics/model_variant_eval.py`, claimed HTTP-302-verified
  during EXP-0005): classified **SUPPORTED**, not VERIFIED — documented in-repo from a
  real prior experiment, but not re-fetched live this session (no internet access
  authorized). No cached raw HTTP response exists locally to inspect deterministically.

## Causal-control implication (EXP-0006 section 7 answer)

**Yes, the registered matched-control design remains scientifically defensible.**
Both the CONTROL and INTERVENTION arms are specified to start from
`benchmark/models/yolov8m-oiv7.pt` — the exact same file, same SHA-256
(`21ffa3718c...`). Since both arms fork from byte-identical weights, whatever happened
during that checkpoint's original 2023 training (now substantially VERIFIED, not
UNKNOWN) does not confound the CONTROL-vs-INTERVENTION comparison itself — only the
broader claim of generalizing beyond this lab's specific starting checkpoint, which was
never the causal claim EXP-0006 makes. The intended IV (training-data composition)
remains isolated. No blocker found.

## Unknown external facts still requiring future verification (not fetched this session)

1. Live re-confirmation of the exact source URL / release asset (HTTP HEAD or GET,
   confirming the 302/200 and exact byte match against `benchmark/models/yolov8m-oiv7.pt`).
2. Whether Ultralytics has published any errata/revision to the specific `v8.3.0`
   release asset since it was originally fetched (checksums can occasionally change on
   a re-tagged release, rare but not impossible).
3. Any formal legal confirmation of Ultralytics' AGPL-3.0 terms as they'd apply to
   OmniSight's specific distribution model, if OmniSight is ever distributed as
   closed-source (an Enterprise license question, not a provenance question).

## Provenance registry artifact

Machine-readable snapshot: `research/provenance_registry_exp0006.json` (generated by
`research.provenance.build_exp0006_checkpoint_registry()`, regenerable deterministically
at any time by re-running it against the same local files).
