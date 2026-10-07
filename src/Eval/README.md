# Eval

`evaluate.evaluate(args)` loads the saved architecture/weights and scores only the
shared manifest's held-out `test` partition. The checkpoint's annotation hash,
split hash, class order, and raw-label mapping must match. Prefer `best.pt`, whose
epoch was selected using validation macro F1. Do not tune settings against test
results; keep test evaluation for the final comparison.

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

The default temporal manifest measures held-out segments of selected known
videos. Its unused frame gaps and single source per class reduce nearby-frame
and additional-view overlap; they do not establish performance on independent
matches or new athletes. Report this scope alongside the metrics. Excluded
examples are never evaluated.

`metrics.classification_metrics(targets, predictions, class_names)` uses only the
standard library. Results include accuracy, macro F1 over all declared classes,
per-class precision/recall/F1/support, and the confusion matrix (true classes in
rows, predictions in columns). Undefined precision/recall/F1 are zero. Output is
`diagnostics/test_metrics.json`, or an explicit `--output` path.
