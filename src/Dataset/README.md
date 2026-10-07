# Dataset

`annotations.py` validates the actual lowercase ViCoS keys: `image`, `frame`,
`position`, `pose1`, `pose2`. Image IDs retain leading zeros. An absent or null
pose is allowed; present poses must contain 17 finite numeric triples. Raw
confidence values outside [0,1] are retained. `LABEL_MAP_10` explicitly merges
the 18 role labels into ten categories; unknown labels fail validation.

`audit.py` reports labels, missing poses, confidence range, duplicate IDs, frame
number discrepancies, optional image existence and per-prefix class coverage.
Malformed records appear in the report and make the audit CLI exit nonzero.
Training refuses malformed annotations instead of silently removing examples.

`temporal.py` implements the default single-view temporal holdout. It selects one
source prefix per normalized class, excludes other views and incidental class
occurrences, and assigns chronological per-class 70/15/15 provisional partitions.
It removes an embargo of ±75 frames around every cross-partition boundary within
each prefix, including transitions between classes, then verifies a separation
greater than 150 original frames and at least 20 examples per class per partition.
An insufficient dataset fails explicitly; no random split or gap reduction is
applied. Ratios after exclusions can differ from the provisional ratios.

`PREPARE_SPLITS.sh` saves `Data/splits/temporal_split.json`. Its schema-2 manifest
records every assignment and exclusion reason, parameters, selected sources,
boundaries, separation checks, class counts, taxonomy and annotation hash. This
is evaluation within known videos, without a claim of independent matches or
athletes. The default sources and rationale are documented in the root README.

`splits.py` retains the explicit `--method recording-group --groups ...` method
for verified recording metadata. It keeps all views of each recording together
and requires every class in every partition. It also writes both kinds of
manifest atomically and refuses to replace an existing file.

`download.py` downloads directly on the remote environment and extracts under
`Data/images`. It preserves archives and existing valid files and reports
progress. `DOWNLOAD_IMAGES.sh` downloads, extracts, and audits image paths;
default bootstrap, smoke tests, and plain `DOWNLOAD_DATA.sh` do not download images.
JSON parsing
loads the full annotation array into RAM; the in-memory representation requires
considerably more memory than the JSON file size.

Primary sources: [ViCoS dataset](https://www2.vicos.si/resources/jiujitsu/),
[author label definitions](https://github.com/ValterH/automatic-positions-detection-and-scoring-in-jiu-jitsu/blob/main/src/scripts/evaluate_positions.py).
