# Train

`engine.train(args)` trains only on `train`, selects the best checkpoint by
validation macro F1 on `val`, and never scores `test`. AdamW uses learning rate
0.001 and weight decay 0.0001 by default. Defaults are 20 epochs, batch size 128,
two DataLoader workers, seed 42, CUDA. An unavailable GPU is an error; choose
`--device cpu` explicitly for a small test. Both classifiers use the same split
and unweighted cross entropy. Start image fine tuning with a smaller learning
rate if needed, by choosing it at the start of a new run.

The default shared manifest is `Data/splits/temporal_split.json`; records marked
`excluded` never enter training or validation. Setup, image download and split
preparation do not start training. Model architectures and optimization defaults
are unchanged by the temporal evaluation protocol.

Every fresh run requires a new `--run-dir`. Outputs are `config_start.json`,
`checkpoints/best.pt`, `checkpoints/last.pt`, and `diagnostics/metrics.jsonl`.
The resolved configuration and provenance capture model, labels, exact annotation
and split hashes, Git state when available, environment, and device. Git is optional.

`--resume <run>/checkpoints/last.pt` resumes that exact run. Omitted settings use
the checkpoint values. Explicit changes to seed, batch size, learning rate,
workers, pretrained/fine-tune/swap settings are rejected. `--epochs` means total
target epochs and can be increased; model, annotation/split hashes, taxonomy,
architecture, and image directory must match. Device can be changed explicitly.
Each continuation writes its own config record. Checkpoints save optimizer,
epoch, best validation score, Python/Torch/CUDA and loader-generator random state.
No NumPy random generator is used. Completed-epoch resume restores these states;
bitwise agreement across different devices or software versions is not guaranteed.

Checkpoints are written atomically using a sibling temporary file. Load only
trusted project checkpoints: restoring full Python random state uses PyTorch
deserialization with `weights_only=False`.
