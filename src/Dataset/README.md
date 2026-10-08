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

`multiview.py` implements the default `multiview-sections` method from the
versioned `docs/multiview_sections.json` plan. It uses all 16 sources in six
related camera groups and assigns contiguous sections throughout each timeline.
The target among retained images is 80% train, 10% validation and 10% test,
with approximately half of each evaluation partition reserved for each type:

- `held_out_view`: evaluate one camera while other views of the section train.
- `unseen_moment`: withhold all available camera views of the section.

The deterministic assignment process balances candidate sections for both the
image population and the pose population, using retained annotation counts and
pose availability rather than model metrics. It targets 80/10/10 separately
within every class as well as overall. Per-class fractions divide by that
class's retained train + validation + test examples, rather than all examples
in a partition. Pose examples require both athletes'
supplied poses in training, validation and test; image examples retain samples
with one supplied pose. Pose held-out-view examples also need an eligible other
view in training. These eligibility rules change sample counts without moving
a section's temporal role between modalities.

Each athlete counts as present when at least one of their supplied joints has
positive confidence. This does not impose a higher confidence threshold or
require every joint to be observed.

The plan rotates held-out cameras and defines sections on an estimated common
reference clock. Each source maps original frame numbers with
`reference = (frame - offset) / scale`. Class transitions and annotation gaps
provide the affine timing landmarks; exact camera synchronization has not been
independently verified. Temporal buffers exclude frames near section boundaries.
The default width is 150 shared reference frames, split evenly around each
boundary touching an evaluation section. Held-out-view samples require a
same-class training counterpart from another camera within three reference
frames; missing counterparts are excluded, with pose eligibility checked
separately for the pose population.
Counts after these exclusions determine actual fractions, which need not equal
the targets exactly. The optional plan field `max_class_deviation` limits the
absolute deviation of every class's train/val/test fractions from their targets
in both populations. The versioned default plan sets `0.025`, limiting each
deviation to 2.5 percentage points; custom plans that omit the field use `0.04`
(4 percentage points). The builder fails if whole-section allocation
cannot meet that bound alongside class coverage and temporal separation; it
does not relax the limit automatically. Candidate sections must then be revised.
Both tests remain within known recordings and athletes.

Frames missing an athlete's pose can occupy boundary buffers without sacrificing
an otherwise usable pose example. They are still image examples, so the shared
image split must retain its temporal separation. Missing-pose intervals do not
by themselves establish an annotation gap or a synchronization landmark.
`pose_buffer_audit` reports all buffered annotated images, how many already lack
an athlete, and how many contain both poses. These counts distinguish existing
pose ineligibility from additional complete-pose examples removed by buffers.

`PREPARE_SPLITS.sh` creates `Data/splits/multiview_sections.json`. Its schema-3
manifest records the full plan, image and pose assignments, evaluation types,
exclusion reasons, pooled/per-type class counts for both populations, taxonomy,
and annotation hash. Loaders rebuild derived fields before accepting it.
`balance_audit` records each population's per-class fractions, worst absolute
deviation, RMS deviation, and permitted maximum, with deviations expressed in
percentage points. The preparation command prints counts and within-class
percentages, both deviation summaries, and the pose buffer counts.
`--section-plan` accepts a custom
JSON plan; choose a new output path for any revision. A changed split requires
fresh training because previous checkpoints may have seen newly held-out images.

The manifest audits final retained gaps, nearest paired-view distances, and
retained held-out examples per camera for both populations. Its distribution
audit checks validation/test examples together in the early, middle and late
thirds of substantial group/class stretches, separately for images and eligible
poses. Both populations use the same estimated reference clock. Each supported
third requires at least 20 evaluation examples. Image checks apply when the class
has at least 600 annotations, at least six sections with 20 examples each, and
at least 20 released examples in all three thirds. Within those same thirds,
pose checks apply individually wherever at least 20 complete-pose examples are
available; an unsupported pose third does not disable checks on the other thirds.
The manifest stores
image results in `distribution_audit.coverage` and pose results in
`distribution_audit.pose_coverage`. General class-coverage checks still apply
where a timeline distribution constraint is unsupported.

`temporal.py` retains the explicit `single-view-temporal` method. It selects one
source prefix per normalized class, excludes other views and incidental class
occurrences, and assigns chronological per-class 70/15/15 provisional partitions.
It removes an embargo of ±75 frames around every cross-partition boundary within
each prefix, including transitions between classes, then verifies a separation
greater than 150 original frames and at least 20 examples per class per partition.
An insufficient dataset fails explicitly; no random split or gap reduction is
applied. Ratios after exclusions can differ from the provisional ratios.

Its schema-2 manifest records every assignment and exclusion reason, parameters, selected sources,
boundaries, separation checks, class counts, taxonomy and annotation hash. This
is evaluation within known videos, without a claim of independent matches or
athletes. Existing temporal manifests remain usable when selected explicitly.

`splits.py` retains the explicit `--method recording-group --groups ...` method
for verified recording metadata. It keeps all views of each recording together
and requires every class in every partition. It also writes manifests for all
methods atomically and refuses to replace an existing file.

`download.py` downloads directly on the remote environment and extracts under
`Data/images`. It preserves archives and existing valid files and reports
progress. `DOWNLOAD_IMAGES.sh` downloads, extracts, and audits image paths;
default bootstrap, smoke tests, and plain `DOWNLOAD_DATA.sh` do not download images.
JSON parsing
loads the full annotation array into RAM; the in-memory representation requires
considerably more memory than the JSON file size.

Primary sources: [ViCoS dataset](https://www2.vicos.si/resources/jiujitsu/),
[author label definitions](https://github.com/ValterH/automatic-positions-detection-and-scoring-in-jiu-jitsu/blob/main/src/scripts/evaluate_positions.py).
