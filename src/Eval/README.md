# Eval

`evaluate.evaluate(args)` loads the saved architecture/weights and scores only the
shared manifest's held-out `test` partition. The checkpoint's annotation hash,
split hash, class order, and raw-label mapping must match. Prefer `best.pt`, whose
epoch was selected using validation macro F1. Do not tune settings against test
results; keep test evaluation for the final comparison.

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
