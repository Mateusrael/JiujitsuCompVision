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

`splits.py` requires an explicit, verified mapping from each observed image prefix
to a recording and train/val/test split. All views of one recording must remain
together. Every split must contain every category. The manifest records the
annotation hash, assignments, taxonomy, grouping evidence and class counts.
It never creates a random frame split. If recording-level coverage cannot be
satisfied, the evaluation design needs revision before generating results.

`download.py` downloads directly on the remote environment and extracts under
`Data/images`. It preserves archives and existing valid files. JSON parsing
loads the full annotation array into RAM; the in-memory representation requires
considerably more memory than the JSON file size.

Primary sources: [ViCoS dataset](https://www2.vicos.si/resources/jiujitsu/),
[author label definitions](https://github.com/ValterH/automatic-positions-detection-and-scoring-in-jiu-jitsu/blob/main/src/scripts/evaluate_positions.py).
