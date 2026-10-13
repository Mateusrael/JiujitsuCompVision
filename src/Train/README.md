# Train

`engine.train(args)` trains only on `train`, selects the best checkpoint by
validation macro F1 on `val`, and never scores `test`. AdamW uses learning rate
0.001 and weight decay 0.0001 by default. Defaults are 20 epochs, batch size 128,
two DataLoader workers, seed 42, CUDA. An unavailable GPU is an error; choose
`--device cpu` explicitly for a small test. All five classifiers share temporal
section roles and use unweighted cross entropy. Pose models require both
athletes' supplied poses in training, validation and test; image models retain
single-pose examples. Report the different eligible populations with results.
Start image fine tuning with a smaller learning rate if needed, by choosing it
at the start of a new run.

For the default multiview split, validation also reports `held_out_view` and
`unseen_moment` separately. The first evaluates other views of sections present
in training; the second withholds every available view of a section. Best
checkpoint selection uses macro F1 computed from the pooled validation images,
not an average of the two subset scores. Both subsets are validation data and
can guide model choices; neither includes test images. Complete subset metrics
are saved under `val.by_evaluation_type` in each epoch's metrics record.

Set optimizer settings with `--lr` and `--weight-decay`. Weight decay accepts
finite nonnegative values, including zero to disable it. AdamW applies it to all
trainable parameters, including biases and normalization parameters. The selected
value is recorded in the run configuration and optimizer checkpoint; changing it
requires a new run, as with learning rate. Runs created before this CLI option
used the same fixed default of 0.0001.

Progress reporting uses `tqdm.auto`: an overall epoch bar and separate batch bars
for training and validation. They show counts, percentage, throughput, elapsed
time and ETA. Batch postfixes show sample-weighted running loss and accuracy;
the epoch postfix and completed-epoch summary show training loss and accuracy,
validation loss and accuracy, and validation macro F1. These metrics are also
saved in `diagnostics/metrics.jsonl` under `train` and `val`; existing runs already
contain them. Training metrics average predictions made during parameter updates
with training augmentations/dropout active. Validation scores the final epoch
weights in evaluation mode without random augmentation. The completed-epoch
summary is printed and flushed. `--no-progress` hides the
bars; `--progress` is the default. Progress does not alter sampling or metrics.
`train_epoch` and `score_model` accept keyword arguments `progress`, `description`
and `position` for display control. `score_model` also supplies the held-out test
evaluation bar.

`--compile` enables `torch.compile` for the complete model; the default is
`--no-compile`. `--compile-mode` accepts `default`, `reduce-overhead`,
`max-autotune`, or `max-autotune-no-cudagraphs`. Nondefault modes require enabled
compilation. The original model owns the optimizer parameters and supplies the
checkpoint state; its compiled wrapper is used for training and validation.
Initial batches and new shapes may incur compilation overhead. Errors propagate
instead of triggering a project-specific fallback to uncompiled execution.

The default shared manifest is `Data/splits/multiview_sections.json`; records marked
`excluded` never enter training or validation. Setup, image download and split
preparation do not start training. Model architectures and optimization defaults
are independent of the section evaluation protocol. Start a new run after
changing split assignments. Resuming a checkpoint trained on the legacy split
would expose some new validation/test images through its earlier training;
annotation and split hash checks reject such a mismatch.

Fresh runs select `--model pose`, `pose-wide`, `pose-attention`, `image`, or
`image-dinov3`.
All three pose models update every parameter from random initialization and use
the same pose loader and athlete-swap augmentation. Image runs train a randomly
initialized 10-class head on a frozen pretrained ResNet-18 by default.
`--image-training head` keeps that default. `--image-training layer4` trains the
last residual stage and classifier while freezing the initial convolution and
stages 1–3, including their BatchNorm statistics. `--image-training full` updates
all backbone parameters. The classifier always trains; BatchNorm in trainable
stages updates during training. All modes use the same `--lr` for their trainable
parameters. Startup prints trainable/total counts and stores these counts in the
configuration alongside the selected mode.

`--fine-tune` and `--no-fine-tune` remain aliases for `full` and `head`; the CLI
rejects combining these aliases with `--image-training`. Image training modes
are rejected for pose models. Old head/full checkpoints restore their original
mode when the explicit `image_training` field is absent.

DINOv3 runs (`image-dinov3`) instead use the official pretrained ViT-B/16 with
all backbone weights frozen. Only a new 768-to-10 linear head trains, with 7,690
trainable parameters. The backbone remains in evaluation mode during head
training. Only `--image-training head` is supported; `layer4`, `full`, and
`--fine-tune` are rejected. It shares the ResNet image loader, letterboxing,
normalization, sample population and augmentation. This preprocessing preserves
the complete frame and deliberately differs from the official processor's direct resize.

Install the pinned runtime requirements, request access on the
[official DINOv3 model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
then run `.venv/bin/hf auth login` interactively in the training
environment before the first pretrained run. Bootstrap does not download model
weights. `--no-pretrained` enables offline smoke checks with random weights;
it is not a meaningful frozen-feature baseline. Full project checkpoints save
the frozen backbone as well as the head, so resume requires no Hub connection.
The same epoch, optimizer, metrics and whole-model compile options apply.
CUDA compilation and speedup for DINOv3 must be verified on the DGX.

`--horizontal-flip-prob` defaults to 0.0 for every model. Set it to 0.5 to mirror
half of training examples on average. Poses reflect centered x and exchange
left/right joint slots within each athlete; images mirror the full frame before
resize-and-pad. This is independent of the pose athlete-swap choice, which
remains enabled with probability 0.5 by default. There is no vertical flip.
Validation and test datasets never use either random augmentation.

`--dropout` defaults to 0.0 for every model. It follows each hidden ReLU in the
MLPs, precedes the image classifier on its 512 ResNet pooled features or 768
DINOv3 CLS features, and acts within the attention blocks. Frozen image backbones
still use classifier-input dropout
during training. Validation and evaluation disable dropout. Its probability
must be finite and in [0, 1); mirroring accepts [0, 1].

Attention runs default to embedding width 128, four heads, four blocks, MLP
hidden width 512, no dropout, and mean pooling over observed joints. Configure
these at run creation with `--attention-dim`, `--attention-heads`,
`--attention-layers`, `--attention-mlp-dim`, and
`--attention-pooling`. Pooling accepts `mean` or `cls`; the latter prepends a
learned classification token. These options are rejected for other models.
They change the model, not the shared input preprocessing or split.
`--attention-dropout` controls attention weights and attention output before
its residual addition. `--attention-mlp-dropout` independently controls dropout
after GELU and on MLP output before its residual addition. Both default to zero;
`--dropout` supplies a shared fallback for omitted branch flags. Explicit branch
values override the fallback, including zero. Both branch flags require
`--model pose-attention` and finite probabilities in [0, 1).

Every fresh run requires a new `--run-dir`. Outputs are `config_start.json`,
`checkpoints/best.pt`, `checkpoints/last.pt`, and `diagnostics/metrics.jsonl`.
The resolved configuration and provenance capture model, labels, exact annotation
and split hashes, Git state when available, environment, and device. Git is optional.

Use `bash launchers/EXPORT_METRICS.sh` to collect metrics and configurations from
all runs into one ZIP. It preserves run names and raw metric logs,
adds a comparison CSV and a manifest, and does not read checkpoints or evaluate
models. See the root README for the archive contents and the limits of snapshots
taken during active training.

`--resume <run>/checkpoints/last.pt` resumes that exact run. Omitted settings use
the checkpoint values. Explicit changes to seed, batch size, learning rate, weight decay,
workers, pretrained/image-training/fine-tune/swap settings, or any attention architecture setting
are rejected, as are changes to dropout or horizontal-mirror probability. Both
effective attention dropout rates are saved and restored independently.
`--epochs` means total
target epochs and can be increased; model, annotation/split hashes, taxonomy,
architecture, and image directory must match. Device can be changed explicitly.
Compilation, compile mode and progress can also be changed on resume, with
omitted flags restoring the saved choices. Switching compilation on or off does
not change the checkpoint format, but can change floating point and RNG behavior.
Each continuation writes its own config record. Checkpoints save optimizer,
epoch, best validation score, Python/Torch/CUDA and loader-generator random state.
No NumPy random generator is used. Completed-epoch resume restores these states;
bitwise agreement across different devices or software versions is not guaranteed.

Checkpoints are written atomically using a sibling temporary file. Load only
trusted project checkpoints: restoring full Python random state uses PyTorch
deserialization with `weights_only=False`.
