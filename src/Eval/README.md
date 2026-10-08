# Eval

`evaluate.evaluate(args)` loads the saved architecture/weights and scores only the
shared manifest's held-out `test` partition. The checkpoint's annotation hash,
split hash, class order, and raw-label mapping must match. Prefer `best.pt`, whose
epoch was selected using validation macro F1. Do not tune settings against test
results; keep test evaluation for the final comparison.

With the default multiview section manifest, results include pooled test metrics
and separate scores for `held_out_view` and `unseen_moment`. The former evaluates
camera views of sections that training saw from other cameras. The latter
evaluates sections withheld from every available camera. Validation uses the
same two categories on different sections; test examples never select epochs
or hyperparameters. Each subset receives its own loss, accuracy, macro F1,
per-class metrics, and confusion matrix.

Pose models evaluate only examples with both athletes' supplied poses. The
image classifier includes single-pose images, so its sample population differs.
The manifest coordinates section assignments and reports counts for both
populations. Scores are saved under `metrics`, with the subset breakdown under
`metrics.by_evaluation_type`; the output also records sample eligibility counts.

The model is reconstructed from its checkpoint, including all attention
dimensions, block/head counts, separate attention and block MLP dropout rates,
and pooling choices. Evaluation does not
select a new architecture or download pretrained weights. `pose`, `pose-wide`
and `pose-attention` all receive the same normalized 102 pose features; `image`
receives the corresponding full frames. Every model returns ten logits, and the
largest logit determines the predicted class. Pose results assume supplied
skeletons and do not evaluate pose extraction or temporal video modeling.

Evaluation disables dropout for all four models and never mirrors examples or
swaps athlete slots, regardless of the saved training augmentation settings.

The test batch progress bar is enabled by default and shows running loss and
accuracy, batch counts, throughput, elapsed time and ETA. `--no-progress` hides
it. Final accuracy and macro F1 are still printed and saved.

Evaluation defaults to uncompiled execution independently of the training run.
`--compile` wraps the entire reconstructed model with `torch.compile` after its
checkpoint weights are loaded. `--compile-mode` accepts `default`,
`reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs`; nondefault
modes require `--compile`. The first batches may take longer to compile.
The output's `execution` field records compilation, mode and progress settings.

The default plan uses all 16 source videos in six related camera groups and
distributes held-out sections throughout their timelines. Source-specific
frame boundaries and temporal buffers coordinate views. Alignment is estimated
from annotation landmarks, not independently verified frame synchronization;
report that limitation alongside the metrics. Neither subset establishes
performance on independent matches or new athletes. Excluded examples are
never evaluated. Previous checkpoints must not be reused after changing split
assignments because training may have exposed the new test frames.

`metrics.classification_metrics(targets, predictions, class_names)` uses only the
standard library. Results include accuracy, macro F1 over all declared classes,
per-class precision/recall/F1/support, and the confusion matrix (true classes in
rows, predictions in columns). Undefined precision/recall/F1 are zero. Output is
`diagnostics/test_metrics.json`, or an explicit `--output` path.
