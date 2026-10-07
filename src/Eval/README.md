# Eval

`evaluate.evaluate(args)` loads the saved architecture/weights and scores only the
shared manifest's held-out `test` partition. The checkpoint's annotation hash,
split hash, class order, and raw-label mapping must match. Prefer `best.pt`, whose
epoch was selected using validation macro F1. Do not tune settings against test
results; keep test evaluation for the final comparison.

`metrics.classification_metrics(targets, predictions, class_names)` uses only the
standard library. Results include accuracy, macro F1 over all declared classes,
per-class precision/recall/F1/support, and the confusion matrix (true classes in
rows, predictions in columns). Undefined precision/recall/F1 are zero. Output is
`diagnostics/test_metrics.json`, or an explicit `--output` path.
